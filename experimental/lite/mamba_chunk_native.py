"""Derive a private HF native SSD callable; no global monkeypatch or scan change."""

import ast
import hashlib
import inspect
import textwrap

from mamba_chunk_contractions import chunk_product_sum

NATIVE_SHA = "37e1eeb82f801616ed330e3e3bb11066202aa8aa4f5a9a9cb8c7c5133a83873e"


def transform(source):
    tree = ast.parse(textwrap.dedent(source))
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef))
    native_source = ast.get_source_segment(textwrap.dedent(source), function)
    if hashlib.sha256(native_source.encode()).hexdigest() != NATIVE_SHA:
        raise ValueError("Unexpected native SSD function source")
    function.decorator_list = []
    changed = []
    output = []
    c_product = None
    for statement in function.body:
        if not (isinstance(statement, ast.Assign) and len(statement.targets) == 1
                and isinstance(statement.targets[0], ast.Name)):
            output.append(statement)
            continue
        name, value = statement.targets[0].id, statement.value
        if name == "C_times_states":
            if not isinstance(value, ast.BinOp) or not isinstance(value.op, ast.Mult):
                raise ValueError("Unexpected C/state contraction")
            c_product = value
            continue
        if (name in ("G", "Y_diag", "states") and isinstance(value, ast.Call)
                and isinstance(value.func, ast.Attribute) and value.func.attr == "sum"
                and isinstance(value.func.value, ast.BinOp)
                and isinstance(value.func.value.op, ast.Mult)):
            product = value.func.value
            if len(value.keywords) != 1 or value.keywords[0].arg != "dim":
                raise ValueError("Unexpected native reduction")
            statement.value = ast.Call(
                func=ast.Name(id="chunk_product_sum", ctx=ast.Load()),
                args=[product.left, product.right, value.keywords[0].value], keywords=[])
            changed.append(name)
        elif name == "Y_off":
            if (c_product is None or not isinstance(value, ast.BinOp)
                    or not isinstance(value.left, ast.Call)
                    or not isinstance(value.left.func, ast.Attribute)
                    or not isinstance(value.left.func.value, ast.Name)
                    or value.left.func.value.id != "C_times_states"
                    or value.left.func.attr != "sum"
                    or len(value.left.args) != 1 or value.left.keywords
                    or ast.literal_eval(value.left.args[0]) != -1):
                raise ValueError("Unexpected off-diagonal contraction")
            value.left = ast.Call(
                func=ast.Name(id="chunk_product_sum", ctx=ast.Load()),
                args=[c_product.left, c_product.right, ast.Constant(value=-1)], keywords=[])
            changed.append("C_times_states")
        output.append(statement)
    if changed != ["G", "Y_diag", "states", "C_times_states"]:
        raise ValueError(f"Unexpected native contraction inventory: {changed}")
    function.body = output
    return ast.fix_missing_locations(tree)


def make_native_scan(original):
    native = getattr(original, "__wrapped__", original)
    tree = transform(inspect.getsource(native))
    namespace = {**native.__globals__, "chunk_product_sum": chunk_product_sum}
    exec(compile(tree, "isolated_chunk_native", "exec"), namespace)
    return namespace[native.__name__]
