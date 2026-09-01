import subprocess
import sys
from pathlib import Path


def test_transport_metadata_import_does_not_require_torch():
    root = Path(__file__).resolve().parents[2]
    process = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; sys.modules['torch']=None; "
            "import rosetta.transport.token_metadata; import rosetta.transport.audit",
        ],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert process.returncode == 0, process.stderr
