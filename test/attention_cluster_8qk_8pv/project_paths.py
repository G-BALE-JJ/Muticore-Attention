"""Resolve relocated project artifacts without rewriting historical evidence."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MIGRATIONS = (
    ('Muticore-Attention', 'test/history/legacy-primary'),
    ('test/history/Muticore-Attention', 'test/history/legacy-primary'),
    ('baseline', 'test'),
    ('src/sst/elements/golem/tests', 'test/golem'),
)


def resolve_project_path(value, root=ROOT):
    root = Path(root)
    path = Path(value)
    if not path.is_absolute():
        path = root / path
    roots = [root]
    history = root / 'test/history'
    if history.is_dir():
        # Feature worktrees share the public archive; old metadata can contain
        # absolute paths from the public workspace rather than this worktree.
        roots.append(history.resolve().parent.parent)
    for origin in roots:
        for old, new in MIGRATIONS:
            try:
                relative = path.relative_to(origin / old)
            except ValueError:
                continue
            return root / new / relative
    return path
