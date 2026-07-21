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


def test_train_transfer3_supports_cnn_lstm_v2_option() -> None:
    module_path = PROJECT_ROOT / "packages" / "uav_fault" / "uav_fault" / "models" / "train_transfer3.py"
    tree = _parse_module(module_path)

    choices: list[str] | None = None
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Attribute) or node.func.attr != "add_argument":
            continue
        if not node.args:
            continue
        first_arg = node.args[0]
        if not isinstance(first_arg, ast.Constant) or first_arg.value != "--model":
            continue
        for kw in node.keywords:
            if kw.arg == "choices":
                choices = list(ast.literal_eval(kw.value))
                break
    assert choices is not None
    assert "cnn_lstm_v2" in choices

    build_model = None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "build_model":
            build_model = node
            break
    assert build_model is not None
    constants = _constant_values(build_model)
    names = _identifier_values(build_model)
    assert "cnn_lstm_v2" in constants
    assert "CNNLSTMv2Classifier" in names
