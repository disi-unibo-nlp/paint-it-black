import io
import logging

from tqdm import tqdm

logger = logging.getLogger(__name__)


def run_ocr(dataset, logger=None) -> list[str]:
    """
    Convert all page images in a HuggingFace dataset to markdown via Docling OCR.

    Imports are deferred so this module is importable even when docling is not installed.

    Args:
        dataset: HF Dataset with an "image" column of PIL Images.
        logger: Logger to use; falls back to the module logger if None.

    Returns:
        List of markdown strings, one per sample, in dataset order.
    """
    from docling.document_converter import DocumentConverter
    from docling.datamodel.base_models import DocumentStream

    converter = DocumentConverter()
    markdowns = []

    for i, row in enumerate(tqdm(dataset, desc="OCR", unit="page")):
        buf = io.BytesIO()
        row["image"].save(buf, format="PNG")
        buf.seek(0)
        stream = DocumentStream(name=f"page_{i}.png", stream=buf)
        result = converter.convert(stream)
        markdowns.append(result.document.export_to_markdown())

    return markdowns
