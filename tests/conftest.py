import glob
import subprocess
import sys

import pytest


@pytest.fixture(scope="session", autouse=True)
def build_python_modules():
    built = (
        (glob.glob("cripser/_cripser*.so") or glob.glob("cripser/_cripser*.pyd"))
        and (glob.glob("cripser/tcripser*.so") or glob.glob("cripser/tcripser*.pyd"))
    )
    if not built:
        subprocess.run([sys.executable, 'setup.py', 'build_ext', '--inplace'], check=True)
