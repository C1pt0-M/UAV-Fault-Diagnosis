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


def _find_function(tree: ast.Module, name: str) -> ast.FunctionDef:
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"function not found: {name}")


def _identifier_values(node: ast.AST) -> set[str]:
    values = {child.id for child in ast.walk(node) if isinstance(child, ast.Name)}
    values.update(child.attr for child in ast.walk(node) if isinstance(child, ast.Attribute))
    return values


def _constant_values(node: ast.AST) -> set[object]:
    return {child.value for child in ast.walk(node) if isinstance(child, ast.Constant)}


def test_cnn_lstm_v2_dg_file_exists_and_compiles() -> None:
    module_path = PROJECT_ROOT / "packages" / "uav_fault" / "uav_fault" / "models" / "cnn" / "cnn_lstm_v2_dg.py"
    assert module_path.exists()
    py_compile.compile(str(module_path), doraise=True)


def test_cnn_lstm_v2_dg_declares_domain_head_and_grl_contract() -> None:
    module_path = PROJECT_ROOT / "packages" / "uav_fault" / "uav_fault" / "models" / "cnn" / "cnn_lstm_v2_dg.py"
    tree = _parse_module(module_path)
    class_node = _find_class(tree, "CNNLSTMv2DGClassifier")

    constants = _constant_values(tree)
    identifiers = _identifier_values(tree)

    assert "ReverseLayerF" in identifiers
    assert 3 in constants
    assert "domain_head" in identifiers
    assert "forward_features" in _identifier_values(class_node)
    forward_fn = _find_function(class_node, "forward")
    arg_names = [arg.arg for arg in forward_fn.args.args]
    assert "return_domain" in arg_names


def test_train_transfer3_dg_exists_and_supports_cnn_lstm_v2_dg() -> None:
    module_path = PROJECT_ROOT / "packages" / "uav_fault" / "uav_fault" / "models" / "train_transfer3_dg.py"
    assert module_path.exists()
    py_compile.compile(str(module_path), doraise=True)

    tree = _parse_module(module_path)
    identifiers = _identifier_values(tree)
    constants = _constant_values(tree)

    assert "CNNLSTMv2DGClassifier" in identifiers
    assert "domain_lambda" in constants or "domain-lambda" in constants
    assert "domain_warmup_epochs" in identifiers or "domain-warmup-epochs" in constants

    build_model = _find_function(tree, "build_model")
    assert "cnn_lstm_v2_dg" in _constant_values(build_model)
    assert "CNNLSTMv2DGClassifier" in _identifier_values(build_model)
