from __future__ import annotations

import argparse
from pathlib import Path

from src.retrieval.modal_client import upload_index_to_modal_volume


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload the local FAISS index artifacts into a Modal Volume.")
    parser.add_argument(
        "--local-index-dir",
        type=Path,
        default=Path("data/indexes/default"),
    )
    parser.add_argument("--volume-name", type=str, default="nlp-videoqa-index")
    parser.add_argument("--remote-index-subdir", type=str, default="indexes/default")
    args = parser.parse_args()

    upload_index_to_modal_volume(
        volume_name=args.volume_name,
        local_index_dir=args.local_index_dir,
        remote_index_subdir=args.remote_index_subdir,
    )
    print(
        f"Uploaded {args.local_index_dir} to Modal Volume {args.volume_name!r} "
        f"under /{args.remote_index_subdir}"
    )


if __name__ == "__main__":
    main()
