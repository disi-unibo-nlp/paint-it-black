"""
Tests for render_predictions.py.

Covers:
- render_sample() output size and type
- Header strip and legend are present
- GT and pred bboxes are drawn (pixel-level smoke test)
- Empty GT / empty preds edge cases
- _draw_box clamping of out-of-range coords
- main() CLI: creates renders directory, clears stale PNGs, produces correct count
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image as PILImage

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from inference.render_predictions import render_sample, _draw_box


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _white_image(w=200, h=300):
    return PILImage.fromarray(np.full((h, w, 3), 255, dtype=np.uint8))


def _sample_meta(idx=0, parse_success=True):
    return {
        "idx": idx,
        "page": 1,
        "total_pages": 3,
        "doc_type": "lab",
        "source_pdf": "test.pdf",
        "parse_success": parse_success,
    }


def _gt():
    return [
        {"label": "NAME:PATIENT", "text": "John Doe",  "bboxes": [[0.1, 0.05, 0.2, 0.4]]},
        {"label": "DATE",         "text": "2024-01-01", "bboxes": [[0.3, 0.1, 0.35, 0.5]]},
    ]


def _preds():
    return [
        {"label": "NAME:PATIENT", "text": "John Doe",  "bboxes": [[0.11, 0.05, 0.21, 0.41]]},
        {"label": "DATE",         "text": "2024-01-02", "bboxes": [[0.55, 0.6, 0.65, 0.9]]},
    ]


# ── render_sample tests ───────────────────────────────────────────────────────

class TestRenderSample:
    def test_returns_pil_image(self):
        img = _white_image()
        result = render_sample(img, _gt(), _preds(), _sample_meta())
        assert isinstance(result, PILImage.Image)

    def test_output_taller_than_input(self):
        """Output must be taller due to header + legend strips."""
        img = _white_image(w=200, h=300)
        result = render_sample(img, _gt(), _preds(), _sample_meta())
        assert result.height > img.height

    def test_output_same_width_as_input(self):
        img = _white_image(w=200, h=300)
        result = render_sample(img, _gt(), _preds(), _sample_meta())
        assert result.width == img.width

    def test_output_not_all_white(self):
        """GT and/or pred bboxes must leave non-white pixels."""
        img = _white_image()
        result = render_sample(img, _gt(), _preds(), _sample_meta())
        arr = np.array(result)
        assert arr.min() < 240, "Expected non-white pixels from annotations or header"

    def test_empty_gt_works(self):
        img = _white_image()
        result = render_sample(img, [], _preds(), _sample_meta())
        assert isinstance(result, PILImage.Image)

    def test_empty_preds_works(self):
        img = _white_image()
        result = render_sample(img, _gt(), [], _sample_meta())
        assert isinstance(result, PILImage.Image)

    def test_both_empty_works(self):
        img = _white_image()
        result = render_sample(img, [], [], _sample_meta())
        assert isinstance(result, PILImage.Image)

    def test_parse_fail_flag_accepted(self):
        img = _white_image()
        result = render_sample(img, [], [], _sample_meta(parse_success=False))
        assert isinstance(result, PILImage.Image)

    def test_input_image_not_mutated(self):
        img = _white_image()
        arr_before = np.array(img).copy()
        render_sample(img, _gt(), _preds(), _sample_meta())
        assert np.array_equal(np.array(img), arr_before)

    def test_rgba_input_converted(self):
        """RGBA input should be accepted and converted to RGB."""
        rgba = PILImage.fromarray(np.full((100, 100, 4), 255, dtype=np.uint8), mode="RGBA")
        result = render_sample(rgba, [], [], _sample_meta())
        assert result.mode == "RGB"

    def test_multiple_bboxes_per_entity(self):
        gt = [{"label": "NAME:PATIENT", "text": "A B", "bboxes": [
            [0.1, 0.1, 0.2, 0.3],
            [0.4, 0.1, 0.5, 0.3],
        ]}]
        result = render_sample(_white_image(), gt, [], _sample_meta())
        assert isinstance(result, PILImage.Image)


# ── _draw_box edge cases ──────────────────────────────────────────────────────

class TestDrawBox:
    def _make_draw(self, w=100, h=100):
        img = PILImage.new("RGBA", (w, h), (255, 255, 255, 255))
        return img, ImageDraw_helper(img)

    def test_clamped_out_of_range_bbox(self):
        """Coords > 1 or < 0 should not raise."""
        from PIL import ImageDraw
        img = PILImage.new("RGBA", (100, 100), (255, 255, 255, 255))
        draw = ImageDraw.Draw(img, "RGBA")
        _draw_box(draw, [-0.1, -0.1, 1.5, 1.5], 100, 100, (255, 0, 0), "LABEL", True)

    def test_zero_size_bbox(self):
        """A degenerate zero-size bbox should not raise."""
        from PIL import ImageDraw
        img = PILImage.new("RGBA", (100, 100), (255, 255, 255, 255))
        draw = ImageDraw.Draw(img, "RGBA")
        _draw_box(draw, [0.5, 0.5, 0.5, 0.5], 100, 100, (0, 255, 0), "X", False)


# ── main() CLI integration test ───────────────────────────────────────────────

class TestMainCLI:
    def _make_results_json(self, tmp_path, n_samples=3):
        data = {
            "args": {
                "input_dataset": "PLACEHOLDER",
                "input_split": "base",
                "from_hub": False,
            },
            "metrics": {},
            "per_sample": [
                {
                    "idx": i,
                    "page": i + 1,
                    "total_pages": n_samples,
                    "doc_type": "lab",
                    "source_pdf": f"doc{i}.pdf",
                    "predictions": [
                        {"label": "DATE", "text": "2024", "bboxes": [[0.1, 0.1, 0.2, 0.3]]}
                    ],
                    "raw_output": "...",
                    "parse_success": True,
                }
                for i in range(n_samples)
            ],
        }
        p = tmp_path / "results.json"
        p.write_text(json.dumps(data))
        return p

    def _make_fake_dataset(self, n=3):
        from datasets import Dataset
        from datasets import Image as HFImage

        rows = [
            {
                "image": PILImage.fromarray(np.full((100, 80, 3), 200, dtype=np.uint8)),
                "page": i + 1,
                "total_pages": n,
                "doc_type": "lab",
                "source_pdf": f"doc{i}.pdf",
                "annotations": [
                    {"label": "NAME:PATIENT", "text": "Alice", "bboxes": [[0.05, 0.05, 0.15, 0.4]]}
                ],
            }
            for i in range(n)
        ]
        return Dataset.from_list(rows).cast_column("image", HFImage())

    def test_renders_correct_count(self, tmp_path):
        from unittest.mock import patch

        results_path = self._make_results_json(tmp_path, n_samples=3)
        fake_ds      = self._make_fake_dataset(3)

        with patch("inference.render_predictions._load_dataset", return_value=fake_ds):
            from inference.render_predictions import main
            import sys as _sys
            _sys.argv = ["render_predictions.py", "--results", str(results_path)]
            main()

        renders = list((results_path.parent / "renders").glob("*.png"))
        assert len(renders) == 3

    def test_limit_respected(self, tmp_path):
        from unittest.mock import patch

        results_path = self._make_results_json(tmp_path, n_samples=5)
        fake_ds      = self._make_fake_dataset(5)

        with patch("inference.render_predictions._load_dataset", return_value=fake_ds):
            from inference.render_predictions import main
            import sys as _sys
            _sys.argv = ["render_predictions.py", "--results", str(results_path), "--limit", "2"]
            main()

        renders = list((results_path.parent / "renders").glob("*.png"))
        assert len(renders) == 2

    def test_stale_renders_cleared(self, tmp_path):
        from unittest.mock import patch

        results_path = self._make_results_json(tmp_path, n_samples=2)
        fake_ds      = self._make_fake_dataset(2)

        renders_dir = results_path.parent / "renders"
        renders_dir.mkdir()
        stale = renders_dir / "9999.png"
        stale.write_bytes(b"fake")

        with patch("inference.render_predictions._load_dataset", return_value=fake_ds):
            from inference.render_predictions import main
            import sys as _sys
            _sys.argv = ["render_predictions.py", "--results", str(results_path)]
            main()

        assert not stale.exists(), "Stale render should have been deleted"
        assert len(list(renders_dir.glob("*.png"))) == 2

    def test_custom_output_dir(self, tmp_path):
        from unittest.mock import patch

        results_path = self._make_results_json(tmp_path, n_samples=1)
        fake_ds      = self._make_fake_dataset(1)
        custom_dir   = tmp_path / "my_renders"

        with patch("inference.render_predictions._load_dataset", return_value=fake_ds):
            from inference.render_predictions import main
            import sys as _sys
            _sys.argv = [
                "render_predictions.py",
                "--results", str(results_path),
                "--output_dir", str(custom_dir),
            ]
            main()

        assert custom_dir.exists()
        assert len(list(custom_dir.glob("*.png"))) == 1

    def test_missing_results_file_exits(self, tmp_path):
        import sys as _sys
        _sys.argv = ["render_predictions.py", "--results", str(tmp_path / "nope.json")]

        from inference.render_predictions import main
        with pytest.raises(SystemExit):
            main()


# helper so _draw_box tests don't import ImageDraw at module level
class ImageDraw_helper:
    def __init__(self, img):
        from PIL import ImageDraw
        self._draw = ImageDraw.Draw(img, "RGBA")

    def __getattr__(self, name):
        return getattr(self._draw, name)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
