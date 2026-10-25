import ast
import subprocess
import sys
from importlib.util import resolve_name
from pathlib import Path

import pytest

import windsr

ROOT = Path(windsr.__file__).parent
SHARED = {"grid", "periods"}
ALLOWED = {
    "data": {"data"} | SHARED,
    "model": {"model", "data"} | SHARED,
    "train": {"train", "model", "data"} | SHARED,
    "evaluate": {"evaluate", "data"} | SHARED,
    "grid": set(),
    "periods": set(),
    "windsr": set(),
}


def module_name(path: Path) -> str:
    parts = path.relative_to(ROOT.parent).with_suffix("").parts
    return ".".join(parts[:-1] if parts[-1] == "__init__" else parts)


def part(module: str) -> str:
    """windsr.data.store -> data, windsr.train -> train, windsr -> windsr."""
    return module.split(".")[1] if "." in module else module


def windsr_imports(path: Path):
    """(line, statement, imported name) for every import of windsr in the file, relative imports resolved."""
    name = module_name(path)
    package = name if path.name == "__init__.py" else name.rpartition(".")[0]
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            targets = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            base = resolve_name("." * node.level + (node.module or ""), package) if node.level else node.module
            targets = [f"{base}.{a.name}" for a in node.names]
        else:
            continue
        for t in targets:
            if t.split(".")[0] == "windsr":
                yield node.lineno, ast.unparse(node), t


def test_each_part_imports_only_what_the_layout_allows():
    broken = []
    for path in sorted(ROOT.rglob("*.py")):
        own = part(module_name(path))
        if own == "__main__":
            continue
        allowed = ALLOWED.get(own, set())
        for line, statement, target in windsr_imports(path):
            if part(target) not in allowed:
                broken.append(f"{path.relative_to(ROOT.parent)}:{line}: {statement} imports {part(target)}, "
                              f"{own} may import {sorted(allowed)}")
    assert not broken, "\n".join(broken)


@pytest.mark.parametrize("module", ["windsr.evaluate.report", "windsr.data.build", "windsr.__main__"])
def test_module_loads_without_torch(module):
    r = subprocess.run([sys.executable, "-c", f"import sys, {module}; print('torch' in sys.modules)"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "False", f"importing {module} loads torch"
