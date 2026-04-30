from pathlib import Path
import yaml

_DEFAULT_PATH = Path(__file__).resolve().parents[2] / "config" / "labels.yaml"


def load_labels(path=None) -> list[str]:
    """Load the canonical PHI label list from config/labels.yaml."""
    p = Path(path) if path else _DEFAULT_PATH
    with open(p) as f:
        return yaml.safe_load(f)["labels"]
