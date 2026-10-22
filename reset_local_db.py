"""Remove only the local Chroma database created by this practical."""

import shutil
from pathlib import Path

STORE_PATH = Path(__file__).parent / "chroma_store"


def main() -> None:
    """Delete the known classroom Chroma directory when it exists."""
    if STORE_PATH.exists():
        shutil.rmtree(STORE_PATH)
        print(f"Removed local Chroma store: {STORE_PATH}")
    else:
        print("No local Chroma store exists; nothing to remove.")


if __name__ == "__main__":
    main()
