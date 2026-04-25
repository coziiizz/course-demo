#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


DEFAULT_TARGET = Path("docs/lab5")


@dataclass(frozen=True)
class Issue:
    kind: str
    message: str
    cell_index: int | None = None
    line: int | None = None

    def render(self) -> str:
        location = []
        if self.cell_index is not None:
            location.append(f"cell {self.cell_index}")
        if self.line is not None:
            location.append(f"line {self.line}")
        prefix = f"[{self.kind}]"
        if location:
            prefix = f"{prefix} ({', '.join(location)})"
        return f"{prefix} {self.message}"


@dataclass
class NotebookReport:
    path: Path
    issues: list[Issue] = field(default_factory=list)
    checks_run: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.issues

    def add_issue(
        self,
        kind: str,
        message: str,
        *,
        cell_index: int | None = None,
        line: int | None = None,
    ) -> None:
        self.issues.append(
            Issue(kind=kind, message=message, cell_index=cell_index, line=line)
        )


class EllipsisVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.lines: list[int] = []

    def visit_Constant(self, node: ast.Constant) -> None:
        if node.value is Ellipsis:
            self.lines.append(node.lineno)
        self.generic_visit(node)


def load_code_cells(notebook_path: Path) -> list[tuple[int, str]]:
    raw = json.loads(notebook_path.read_text(encoding="utf-8"))
    cells: list[tuple[int, str]] = []
    for index, cell in enumerate(raw.get("cells", []), start=1):
        if cell.get("cell_type") != "code":
            continue
        source = cell.get("source", "")
        if isinstance(source, list):
            source = "".join(source)
        cells.append((index, source))
    return cells


def iter_notebooks(targets: Iterable[Path]) -> list[Path]:
    notebooks: list[Path] = []
    for target in targets:
        if target.is_dir():
            notebooks.extend(sorted(target.rglob("*.ipynb")))
        elif target.suffix == ".ipynb":
            notebooks.append(target)
    return notebooks


def should_skip_import(node: ast.stmt) -> bool:
    if isinstance(node, ast.ImportFrom):
        return node.module == "transformers"
    if isinstance(node, ast.Import):
        return any(alias.name == "transformers" for alias in node.names)
    return False


def build_module(
    source: str,
    *,
    class_names: set[str],
    function_names: set[str] | None = None,
) -> tuple[ast.Module, set[str]]:
    function_names = function_names or set()
    tree = ast.parse(source)
    selected_nodes: list[ast.stmt] = []
    seen: set[str] = set()

    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            if not should_skip_import(node):
                selected_nodes.append(node)
        elif isinstance(node, ast.ClassDef) and node.name in class_names:
            selected_nodes.append(node)
            seen.add(node.name)
        elif isinstance(node, ast.FunctionDef) and node.name in function_names:
            selected_nodes.append(node)
            seen.add(node.name)

    module = ast.Module(body=selected_nodes, type_ignores=[])
    ast.fix_missing_locations(module)
    missing = (class_names | function_names) - seen
    return module, missing


def fake_tokenizer_factory(torch_module):
    class FakeTokenizer:
        pad_token_id = 0
        unk_token_id = 1

        def __call__(
            self,
            text: str,
            *,
            truncation: bool = True,
            padding: str = "max_length",
            max_length: int = 64,
            return_tensors: str = "pt",
        ):
            tokens = self.encode(text, add_special_tokens=True)[:max_length]
            attention = [1] * len(tokens)
            if padding == "max_length" and len(tokens) < max_length:
                pad_size = max_length - len(tokens)
                tokens = tokens + [self.pad_token_id] * pad_size
                attention = attention + [0] * pad_size
            if return_tensors != "pt":
                raise ValueError("FakeTokenizer only supports return_tensors='pt'")
            return {
                "input_ids": torch_module.tensor([tokens], dtype=torch_module.long),
                "attention_mask": torch_module.tensor(
                    [attention], dtype=torch_module.long
                ),
            }

        def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
            pieces = text.split()
            ids = [((len(piece) + idx) % 23) + 2 for idx, piece in enumerate(pieces)]
            if add_special_tokens:
                return [101] + ids + [102]
            return ids

        def get_vocab(self) -> dict[str, int]:
            return {f"tok_{idx}": idx for idx in range(128)}

    class FakeAutoTokenizer:
        @classmethod
        def from_pretrained(cls, model_name: str):
            return FakeTokenizer()

    return FakeAutoTokenizer


def fake_model_factory(torch_module):
    import torch.nn as nn

    class FakeBertModel(nn.Module):
        def __init__(self, hidden_size: int = 16):
            super().__init__()
            self.config = type("Config", (), {"hidden_size": hidden_size})()
            self.embedding = nn.Embedding(256, hidden_size)

        def forward(self, input_ids, attention_mask=None):
            embedded = self.embedding(input_ids)
            return type("Output", (), {"last_hidden_state": embedded})()

    class FakeAutoModel:
        @classmethod
        def from_pretrained(cls, model_name: str):
            return FakeBertModel()

    return FakeAutoModel


def smoke_test_transformer(report: NotebookReport, source: str) -> None:
    try:
        import torch
    except ImportError as exc:
        report.add_issue("dependency", f"smoke test needs torch: {exc}")
        return

    required_classes = {
        "PositionalEncoding",
        "MultiHeadSelfAttention",
        "TransformerEncoderLayer",
        "TransformerEncoderClassifier",
    }
    module, missing = build_module(source, class_names=required_classes)
    if missing:
        report.add_issue(
            "smoke",
            "missing class definitions: " + ", ".join(sorted(missing)),
        )
        return

    namespace = {
        "__builtins__": __builtins__,
        "AutoTokenizer": fake_tokenizer_factory(torch),
        "pad_idx": 0,
    }

    try:
        exec(compile(module, str(report.path), "exec"), namespace)
        positional = namespace["PositionalEncoding"](d_model=8, max_len=32)
        encoded = positional(torch.zeros(2, 5, 8))
        assert encoded.shape == (2, 5, 8)

        attention = namespace["MultiHeadSelfAttention"](d_model=8, n_heads=2)
        x = torch.randn(2, 5, 8)
        mask = torch.ones(2, 1, 1, 5, dtype=torch.long)
        output, attn = attention(x, mask=mask, return_attn=True)
        assert output.shape == (2, 5, 8)
        assert attn.shape == (2, 2, 5, 5)

        layer = namespace["TransformerEncoderLayer"](d_model=8, n_heads=2, d_ff=16)
        layer_out, layer_attn = layer(x, mask=mask, return_attn=True)
        assert layer_out.shape == (2, 5, 8)
        assert layer_attn.shape == (2, 2, 5, 5)

        classifier_cls = namespace["TransformerEncoderClassifier"]
        input_ids = torch.tensor(
            [[5, 7, 9, 0, 0], [4, 8, 6, 3, 0]], dtype=torch.long
        )
        attention_mask = (input_ids != 0).long()
        for pooling in ("mean", "max"):
            model = classifier_cls(
                vocab_size=128,
                d_model=8,
                n_heads=2,
                d_ff=16,
                num_layers=2,
                num_classes=4,
                pooling=pooling,
            )
            logits, last_attn = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_attn=True,
            )
            assert logits.shape == (2, 4)
            assert last_attn.shape == (2, 2, 5, 5)
        report.checks_run.append("transformer-smoke")
    except Exception as exc:
        report.add_issue("smoke", f"transformer smoke test failed: {exc}")


def smoke_test_bert(report: NotebookReport, source: str) -> None:
    try:
        import torch
    except ImportError as exc:
        report.add_issue("dependency", f"smoke test needs torch: {exc}")
        return

    required_classes = {"AGNewsDataset", "BERTClassifier"}
    module, missing = build_module(source, class_names=required_classes)
    if missing:
        report.add_issue(
            "smoke",
            "missing class definitions: " + ", ".join(sorted(missing)),
        )
        return

    namespace = {
        "__builtins__": __builtins__,
        "AutoTokenizer": fake_tokenizer_factory(torch),
        "AutoModel": fake_model_factory(torch),
    }

    try:
        exec(compile(module, str(report.path), "exec"), namespace)
        tokenizer = namespace["AutoTokenizer"].from_pretrained("bert-base-uncased")
        dataset = namespace["AGNewsDataset"](
            ["hello world", "deep learning course"],
            [0, 1],
            tokenizer,
            max_length=8,
        )
        sample = dataset[0]
        assert sample["input_ids"].shape == (8,)
        assert sample["attention_mask"].shape == (8,)

        model_cls = namespace["BERTClassifier"]
        input_ids = torch.tensor(
            [[101, 5, 7, 102, 0, 0], [101, 9, 3, 4, 102, 0]], dtype=torch.long
        )
        attention_mask = (input_ids != 0).long()
        for pooling in ("cls", "mean", "attention"):
            model = model_cls("bert-base-uncased", num_labels=4, pooling=pooling)
            logits = model(input_ids, attention_mask)
            assert logits.shape == (2, 4)
        report.checks_run.append("bert-smoke")
    except Exception as exc:
        report.add_issue("smoke", f"bert smoke test failed: {exc}")


def check_notebook(notebook_path: Path) -> NotebookReport:
    report = NotebookReport(path=notebook_path)
    try:
        code_cells = load_code_cells(notebook_path)
    except Exception as exc:
        report.add_issue("load", f"failed to load notebook: {exc}")
        return report

    report.checks_run.append("syntax")
    report.checks_run.append("placeholder")

    combined_source = "\n\n".join(source for _, source in code_cells)
    for cell_index, source in code_cells:
        try:
            tree = ast.parse(source)
        except SyntaxError as exc:
            report.add_issue(
                "syntax",
                exc.msg,
                cell_index=cell_index,
                line=exc.lineno,
            )
            continue

        visitor = EllipsisVisitor()
        visitor.visit(tree)
        for line in visitor.lines:
            report.add_issue(
                "placeholder",
                "found Python ellipsis placeholder `...`",
                cell_index=cell_index,
                line=line,
            )

    if any(issue.kind in {"syntax", "placeholder"} for issue in report.issues):
        return report

    notebook_name = notebook_path.name.lower()
    if notebook_name == "transformer.ipynb":
        smoke_test_transformer(report, combined_source)
    elif notebook_name == "bert.ipynb":
        smoke_test_bert(report, combined_source)

    return report


def render_report(report: NotebookReport, repo_root: Path) -> str:
    try:
        display_path = report.path.relative_to(repo_root)
    except ValueError:
        display_path = report.path

    status = "PASS" if report.ok else "FAIL"
    lines = [f"[{status}] {display_path}"]
    if report.checks_run:
        lines.append("  checks: " + ", ".join(report.checks_run))
    for issue in report.issues:
        lines.append("  - " + issue.render())
    return "\n".join(lines)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check lab5 notebooks for placeholder code and CPU smoke-testability."
    )
    parser.add_argument(
        "targets",
        nargs="*",
        type=Path,
        default=[DEFAULT_TARGET],
        help="Notebook files or directories to check. Defaults to docs/lab5.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    repo_root = Path.cwd()
    notebooks = iter_notebooks(args.targets)
    if not notebooks:
        print("No notebook files found.", file=sys.stderr)
        return 2

    reports = [check_notebook(path) for path in notebooks]
    for report in reports:
        print(render_report(report, repo_root))

    failed = sum(not report.ok for report in reports)
    passed = len(reports) - failed
    print(
        f"Summary: checked {len(reports)} notebook(s), "
        f"{passed} passed, {failed} failed."
    )
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
