from __future__ import annotations

import ast
import py_compile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _parse_module(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"))


def _find_class(tree: ast.Module, name: str) -> ast.ClassDef:
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    raise AssertionError(f"class not found: {name}")


def _constant_values(node: ast.AST) -> set[object]:
    return {child.value for child in ast.walk(node) if isinstance(child, ast.Constant)}


def _name_values(node: ast.AST) -> set[str]:
    return {child.id for child in ast.walk(node) if isinstance(child, ast.Name)}


def _identifier_values(node: ast.AST) -> set[str]:
    values = _name_values(node)
    values.update(child.attr for child in ast.walk(node) if isinstance(child, ast.Attribute))
    return values


def test_cnn_lstm_v2_model_file_exists_and_compiles() -> None:
    module_path = PROJECT_ROOT / "packages" / "uav_fault" / "uav_fault" / "models" / "cnn" / "cnn_lstm_v2.py"
    assert module_path.exists()
    py_compile.compile(str(module_path), doraise=True)


def test_cnn_lstm_v2_declares_multiscale_skip_and_groupnorm() -> None:
    module_path = PROJECT_ROOT / "packages" / "uav_fault" / "uav_fault" / "models" / "cnn" / "cnn_lstm_v2.py"
    tree = _parse_module(module_path)
    class_node = _find_class(tree, "CNNLSTMv2Classifier")

    constants = _constant_values(class_node)
    names = _identifier_values(class_node)

    assert 3 in constants
    assert 7 in constants
    assert 15 in constants
    assert 32 in constants
    assert 128 in constants
    assert "GroupNorm" in names
    assert "LSTM" in names
    assert "Linear" in names

