"""Build a temporary host-only scalar oracle once per verifier process."""
from functools import lru_cache
from pathlib import Path
import subprocess
import tempfile


@lru_cache(maxsize=1)
def compiled():
    here=Path(__file__).resolve().parent
    directory=tempfile.TemporaryDirectory(prefix='golem-dense-oracle-')
    target=Path(directory.name)/'reference'
    subprocess.run(['g++','-O3','-std=c++17','-ffp-contract=off','-I',str(here.parents[1]/'src'),
                    str(here/'dense_linear_reference.cpp'),'-o',str(target)],check=True)
    return directory,target


def executable():
    return compiled()[1]
