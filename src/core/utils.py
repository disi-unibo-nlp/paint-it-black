from typing import Any, Dict, Optional, List
import argparse
import logging
import sys
import yaml
from pathlib import Path
import random
import numpy as np

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


class ConfigArgumentParser:
    """
    A wrapper around argparse that integrates YAML config files.

    Priority order (highest to lowest):
    1. Explicit command-line arguments
    2. YAML config file values
    3. Default values specified in add_argument()

    Usage:
        parser = ConfigArgumentParser(description="My script")
        parser.add_argument("--run_name", type=str, default="run")
        parser.add_argument("--lr", type=float, default=1e-4)
        args = parser.parse_args()
        # run with: python3 src/script.py --config my_config [--lr 5e-5]
    """

    def __init__(
        self,
        description: str = "",
        config_arg_name: str = "--config",
        config_dir: Path = Path("./config"),
    ):
        self.parser = argparse.ArgumentParser(description=description)
        self.config_arg_name = config_arg_name
        self._arguments = {}
        self.config_dir = config_dir

        self.parser.add_argument(
            config_arg_name,
            type=str,
            default=None,
            help="YAML config file name (relative to config/ dir, .yaml extension optional)",
        )

    def add_argument(self, *args, **kwargs):
        if "dest" in kwargs:
            dest = kwargs["dest"]
        else:
            dest = args[0].lstrip("-").replace("-", "_")
            if args[0].startswith("--"):
                dest = args[0].lstrip("--").replace("-", "_")
            elif args[0].startswith("-"):
                dest = args[0].lstrip("-")
            else:
                dest = args[0]

        self._arguments[dest] = {
            "type": kwargs.get("type", str),
            "choices": kwargs.get("choices", None),
            "default": kwargs.get("default", None),
            "action": kwargs.get("action", "store"),
            "nargs": kwargs.get("nargs", None),
        }
        return self.parser.add_argument(*args, **kwargs)

    def _validate_config(self, config: Dict[str, Any]) -> None:
        config_key = self.config_arg_name.lstrip("-").replace("-", "_")
        for key, value in config.items():
            if key == config_key:
                continue
            if key not in self._arguments:
                available = sorted(k for k in self._arguments if k != config_key)
                raise ValueError(
                    f"Config key '{key}' is not a registered argument. "
                    f"Available: {available}"
                )
            spec = self._arguments[key]
            if spec["action"] in ("store_true", "store_false"):
                if not isinstance(value, bool):
                    raise ValueError(f"Config key '{key}' expects bool, got {type(value).__name__}")
                continue
            if value is None:
                continue
            if spec["type"] is not None and spec["nargs"] is None:
                expected = spec["type"]
                if expected == float and isinstance(value, (int, float)):
                    pass
                elif not isinstance(value, expected):
                    raise ValueError(
                        f"Config key '{key}' expects {expected.__name__}, got {type(value).__name__}"
                    )
            if spec["choices"] is not None and value not in spec["choices"]:
                raise ValueError(
                    f"Config key '{key}' = '{value}' is not a valid choice: {spec['choices']}"
                )
            if spec["nargs"] is not None:
                if not isinstance(value, list):
                    raise ValueError(f"Config key '{key}' expects a list, got {type(value).__name__}")
                if spec["type"] is not None:
                    for item in value:
                        if not isinstance(item, spec["type"]):
                            raise ValueError(
                                f"Config key '{key}' list items expect {spec['type'].__name__}, "
                                f"got {type(item).__name__} for item '{item}'"
                            )

    def parse_args(self, args: Optional[List[str]] = None) -> argparse.Namespace:
        initial_args, _ = self.parser.parse_known_args(args)
        config_key = self.config_arg_name.lstrip("-").replace("-", "_")
        config_path = getattr(initial_args, config_key)

        config_values = {}
        if config_path:
            config_path = Path(config_path)
            if not config_path.suffix or config_path.suffix != ".yaml":
                config_path = Path(str(config_path) + ".yaml") if not config_path.suffix else config_path

            resolved = None
            if config_path.is_absolute():
                resolved = config_path
            else:
                candidates = [self.config_dir / config_path]
                if config_path.parts and config_path.parts[0] == self.config_dir.name:
                    remaining = config_path.parts[1:]
                    if remaining:
                        candidates.append(self.config_dir / Path(*remaining))
                candidates.append(config_path)
                for c in candidates:
                    if c.exists():
                        resolved = c
                        break
                if resolved is None:
                    resolved = candidates[0]

            if not resolved.exists():
                raise FileNotFoundError(f"Config file not found: {resolved}")

            with open(resolved, "r") as f:
                config_values = yaml.safe_load(f) or {}
            self._validate_config(config_values)

        for key, value in config_values.items():
            if key in self._arguments:
                self.parser.set_defaults(**{key: value})

        return self.parser.parse_args(args)


def init_logger(name: Optional[str] = None, level: str = "INFO", output_path: Optional[str] = None) -> logging.Logger:
    """
    Initialize a logger with console output and optional file output.

    Args:
        name: Logger name (use __name__ in modules). None returns root logger.
        level: Log level string ("DEBUG", "INFO", "WARNING", "ERROR").
        output_path: If provided, also write logs to this file path.

    Returns:
        Configured Logger instance.
    """
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="[%(levelname)s] %(name)s - %(message)s",
        force=True,
    )

    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, level.upper()))
    formatter = logging.Formatter("[%(levelname)s] %(name)s - %(message)s")

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(output_path, encoding="utf-8", mode="w")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger.propagate = False
    return logger


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility across random, numpy, and optionally torch."""
    random.seed(seed)
    np.random.seed(seed)
    if _TORCH_AVAILABLE:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def make_output_dir(run_name: str, base: str = "./output") -> Path:
    """
    Create and return the output directory for a run.

    Args:
        run_name: Name of the run (becomes the subdirectory name).
        base: Base output directory (default: ./output).

    Returns:
        Path to the created directory.

    Example:
        out_dir = make_output_dir(args.run_name, base="./output/dataprep")
        # creates ./output/dataprep/my_run/ and returns that Path
    """
    out_dir = Path(base) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir
