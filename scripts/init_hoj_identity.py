"""One-shot local service identity. Never rotates or prints an existing key."""
import argparse
import os
from pathlib import Path
import re
import secrets


def initialize(path: Path, uid=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o400)
    except FileExistsError:
        if path.stat().st_size > 128 or not re.fullmatch(r"[a-f0-9]{64}", path.read_text(encoding="ascii").strip()):
            raise ValueError("Existing service identity is invalid; it was not replaced") from None
    else:
        with os.fdopen(descriptor, "w", encoding="ascii") as output:
            output.write(secrets.token_hex(32))
            output.flush()
            os.fsync(output.fileno())
    if uid is not None:
        os.chown(path, uid, uid)
        os.chmod(path, 0o400)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    parser.add_argument("--uid", type=int)
    args = parser.parse_args()
    initialize(args.path, args.uid)
