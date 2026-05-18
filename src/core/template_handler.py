from dataclasses import dataclass, field
from typing import Optional, Union
from pathlib import Path
import base64
import io
import re
import yaml


def _inject_labels_into_schema(node, labels_list: list):
    """Recursively replace the sentinel string "{labels}" with the labels list."""
    if isinstance(node, dict):
        return {k: _inject_labels_into_schema(v, labels_list) for k, v in node.items()}
    if isinstance(node, list):
        return [_inject_labels_into_schema(item, labels_list) for item in node]
    if node == "{labels}":
        return labels_list
    return node


@dataclass
class OutputField:
    """
    Defines a single expected field in the model's output.

    Attributes:
        name: Field name, used as the key in ParsedOutput.fields.
        pattern: Regex pattern with exactly one capture group. If None, the full
                 output text is used as the field value (no extraction performed).
        is_list: If True, use re.findall to collect all matches (in order).
                 If False, use re.search to get the first match only.
        required: If True, a missing match marks the overall parse as failed
                  and reduces format_quality. Default True.

    Examples:
        # Single required field with a tag delimiter
        OutputField(name="answer", pattern=r"<answer>(.*?)</answer>")

        # Optional list of reasoning steps
        OutputField(name="steps", pattern=r"<step>(.*?)</step>", is_list=True, required=False)

        # Take the full output as-is (no regex)
        OutputField(name="response")
    """
    name: str
    pattern: Optional[str] = None
    is_list: bool = False
    required: bool = True


@dataclass
class ParsedOutput:
    """
    Result of parsing a model output.

    Attributes:
        success: True if all required fields were found.
        format_quality: Fraction of required fields successfully extracted (0.0–1.0).
                        1.0 when there are no required fields.
        fields: Dict mapping field name -> extracted value.
                Values are str (is_list=False) or list[str] (is_list=True), or None if not found.
    """
    success: bool
    format_quality: float
    fields: dict


def _encode_image(pil_image) -> str:
    buf = io.BytesIO()
    pil_image.convert("RGB").save(buf, format="JPEG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


class TemplateHandler:
    """
    Handles prompt formatting and structured output parsing for LLM interactions.

    Prompt templates are defined as a list of message dicts (role + content),
    where content strings may contain {placeholder} slots for runtime substitution.
    The last message with role "assistant" is treated as the SFT target and is only
    included when format() is called with include_last_assistant=True.

    Output parsing uses a list of OutputField definitions, each with an optional
    regex pattern (with one capture group) applied to the model's raw output.

    Templates can be defined directly in code or loaded from a YAML file:

        messages:
          - role: system
            content: "You are a helpful assistant."
          - role: user
            content: "Question: {question}"
          - role: assistant          # optional SFT target
            content: "<answer>{answer}</answer>"

        output_fields:
          - name: answer
            pattern: "<answer>(.*?)</answer>"
            required: true
          - name: steps
            pattern: "<step>(.*?)</step>"
            is_list: true
            required: false

    Usage:
        handler = TemplateHandler.from_yaml("templates/my_template.yaml")

        # Inference: format without the assistant message
        messages = handler.format(question="What is 2+2?")

        # SFT: format including the target assistant message
        messages = handler.format(include_last_assistant=True, question="What is 2+2?", answer="4")

        # Parse model output
        result = handler.parse("<answer>4</answer>")
        result.success          # True
        result.format_quality   # 1.0
        result.fields["answer"] # "4"
    """

    def __init__(
        self,
        messages: list,
        output_fields: Optional[list] = None,
        structured_output_schema: Optional[dict] = None,
    ):
        """
        Args:
            messages: List of message dicts with "role" and "content" keys.
                      Content strings may contain {placeholder} slots.
            output_fields: List of OutputField instances. If empty or None,
                           parse() returns the full output under key "output".
        """
        self.messages = messages
        self.output_fields: list[OutputField] = output_fields or []
        self.structured_output_schema: Optional[dict] = structured_output_schema

    @classmethod
    def from_yaml(cls, path: Union[str, Path], labels: Optional[list] = None) -> "TemplateHandler":
        """
        Load a TemplateHandler from a YAML file.

        The YAML must have a "messages" key and optionally an "output_fields" key.
        See class docstring for the expected format.

        If `labels` is provided, the placeholder `{labels}` in any string message
        content is replaced at load time with the comma-separated label list.
        """
        with open(path, "r") as f:
            data = yaml.safe_load(f)

        messages = data["messages"]

        if labels:
            labels_str = ", ".join(labels)
            for msg in messages:
                if isinstance(msg.get("content"), str):
                    msg["content"] = msg["content"].replace("{labels}", labels_str)

        output_fields = []
        for fd in data.get("output_fields", []):
            output_fields.append(
                OutputField(
                    name=fd["name"],
                    pattern=fd.get("pattern", None),
                    is_list=fd.get("is_list", False),
                    required=fd.get("required", True),
                )
            )

        schema = data.get("structured_output")
        if schema is not None and labels:
            schema = _inject_labels_into_schema(schema, list(labels))

        return cls(messages=messages, output_fields=output_fields, structured_output_schema=schema)

    def format(self, include_last_assistant: bool = False, **kwargs) -> list:
        """
        Build the message list with placeholders filled.

        Args:
            include_last_assistant: If True and the last message has role "assistant",
                                    include it (used for SFT target formatting).
            **kwargs: Values for {placeholder} slots. String values replace {name}
                      tokens; PIL.Image values are base64-encoded when referenced
                      by an image content block.

        Returns:
            List of {"role": ..., "content": ...} dicts in OpenAI API format.
            Content is a plain string for text-only messages or a list of typed
            blocks for multimodal messages.
        """
        messages = []
        for i, msg in enumerate(self.messages):
            is_last = i == len(self.messages) - 1
            if is_last and msg["role"] == "assistant" and not include_last_assistant:
                continue
            raw_content = msg["content"]
            if isinstance(raw_content, str):
                content = raw_content.format(**kwargs)
            else:
                content = []
                for block in raw_content:
                    if block["type"] == "text":
                        content.append({"type": "text", "text": block["text"].format(**kwargs)})
                    elif block["type"] == "image":
                        image = kwargs[block["variable"]]
                        content.append({"type": "image_url", "image_url": {"url": _encode_image(image)}})
            messages.append({"role": msg["role"], "content": content})
        return messages

    def parse(self, text: str) -> ParsedOutput:
        """
        Extract defined output fields from a model-generated text.

        For each OutputField:
        - pattern=None: the field receives the full text (always succeeds).
        - is_list=False: re.search, returns the first capture group or None.
        - is_list=True: re.findall, returns all matches in order (list may be empty).

        All regex patterns are matched with re.DOTALL so "." spans newlines.

        Args:
            text: Raw model output string.

        Returns:
            ParsedOutput with success, format_quality, and fields dict.
        """
        if not self.output_fields:
            return ParsedOutput(success=True, format_quality=1.0, fields={"output": text})

        fields = {}
        required_total = sum(1 for f in self.output_fields if f.required)
        required_found = 0
        success = True

        for f in self.output_fields:
            if f.pattern is None:
                fields[f.name] = [text] if f.is_list else text
                if f.required:
                    required_found += 1
            elif f.is_list:
                matches = re.findall(f.pattern, text, re.DOTALL)
                fields[f.name] = matches if matches else None
                if f.required:
                    if matches:
                        required_found += 1
                    else:
                        success = False
            else:
                match = re.search(f.pattern, text, re.DOTALL)
                if match:
                    fields[f.name] = match.group(1)
                    if f.required:
                        required_found += 1
                else:
                    fields[f.name] = None
                    if f.required:
                        success = False

        format_quality = required_found / required_total if required_total > 0 else 1.0
        return ParsedOutput(success=success, format_quality=format_quality, fields=fields)
