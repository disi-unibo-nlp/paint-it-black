from pathlib import Path
import yaml

_DEFAULT_PATH = Path(__file__).resolve().parents[2] / "config" / "labels.yaml"


def load_labels(path=None) -> list[str]:
    """Load the canonical PHI label list from config/labels.yaml."""
    p = Path(path) if path else _DEFAULT_PATH
    with open(p) as f:
        return [e["name"] for e in yaml.safe_load(f)["labels"]]


def load_label_guidelines(path=None) -> list[tuple[str, str]]:
    """Load PHI labels with their one-line descriptions from config/labels.yaml."""
    p = Path(path) if path else _DEFAULT_PATH
    with open(p) as f:
        return [(e["name"], e["description"]) for e in yaml.safe_load(f)["labels"]]
