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


def test_bilstm_normalproto_file_exists_and_compiles() -> None:
    module_path = PROJECT_ROOT / "packages" / "uav_fault" / "uav_fault" / "models" / "cnn" / "bilstm_normalproto.py"
    assert module_path.exists()
    py_compile.compile(str(module_path), doraise=True)


def test_bilstm_normalproto_declares_normal_center_and_feature_interface() -> None:
    module_path = PROJECT_ROOT / "packages" / "uav_fault" / "uav_fault" / "models" / "cnn" / "bilstm_normalproto.py"
    tree = _parse_module(module_path)
    class_node = _find_class(tree, "BiLSTMNormalProtoClassifier")

    identifiers = _identifier_values(class_node)
    constants = _constant_values(class_node)

    assert "normal_center" in identifiers
    assert "Parameter" in identifiers
    assert "forward_features" in identifiers
    assert "LSTM" in identifiers
    assert "Linear" in identifiers
    assert 128 in constants


def test_train_transfer3_bilstm_normalproto_exists_and_supports_expected_args() -> None:
    module_path = PROJECT_ROOT / "packages" / "uav_fault" / "uav_fault" / "models" / "train_transfer3_bilstm_normalproto.py"
    assert module_path.exists()
    py_compile.compile(str(module_path), doraise=True)

    tree = _parse_module(module_path)
    identifiers = _identifier_values(tree)
    constants = _constant_values(tree)

    assert "BiLSTMNormalProtoClassifier" in identifiers
    assert "lambda_normal" in identifiers or "lambda-normal" in constants
    assert "normal_margin" in identifiers or "normal-margin" in constants

    build_model = _find_function(tree, "build_model")
    assert "bilstm_normalproto" in _constant_values(build_model)
    assert "BiLSTMNormalProtoClassifier" in _identifier_values(build_model)
