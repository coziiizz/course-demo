import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "scripts" / "lab5_checker.py"


def load_checker_module():
    spec = importlib.util.spec_from_file_location("lab5_checker", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_notebook(path: Path, code_cells: list[str]) -> None:
    notebook = {
        "cells": [
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": cell.splitlines(keepends=True),
            }
            for cell in code_cells
        ],
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path.write_text(json.dumps(notebook, ensure_ascii=False), encoding="utf-8")


class Lab5CheckerTests(unittest.TestCase):
    def test_detects_ellipsis_placeholder(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            notebook_path = Path(tmpdir) / "placeholder.ipynb"
            write_notebook(notebook_path, ["value = ...\n"])

            checker = load_checker_module()
            report = checker.check_notebook(notebook_path)

            self.assertFalse(report.ok)
            self.assertTrue(any(issue.kind == "placeholder" for issue in report.issues))

    def test_passes_simple_notebook_without_placeholders(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            notebook_path = Path(tmpdir) / "simple.ipynb"
            write_notebook(notebook_path, ["value = 1\nprint(value)\n"])

            checker = load_checker_module()
            report = checker.check_notebook(notebook_path)

            self.assertTrue(report.ok)
            self.assertEqual(report.issues, [])

    def test_cli_returns_non_zero_for_failed_check(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            notebook_path = Path(tmpdir) / "placeholder.ipynb"
            write_notebook(notebook_path, ["value = ...\n"])

            completed = subprocess.run(
                [sys.executable, str(MODULE_PATH), str(notebook_path)],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("placeholder", completed.stdout.lower())


if __name__ == "__main__":
    unittest.main()
