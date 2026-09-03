#!/usr/bin/env python3
"""
Rebuild the TorchScript checkpoints that are too large to store directly in git.

GitHub rejects any blob over 100 MB and Git LFS is not usable through the
anonymisation proxy, so `seacast_scripted.pt` (186 MB) and
`graph_fm_scripted.pt` (124 MB) ship as xz-compressed siblings, split into
<95 MB parts where the compressed form is still over the limit:

    seacast_scripted.pt.xz
    graph_fm_scripted.pt.xz.part-aa
    graph_fm_scripted.pt.xz.part-ab

Run this once after cloning, before starting the backend:

    python3 scripts/restore_models.py

Nothing outside the standard library is required. `graphcast_scripted.pt` is
25 MB and is committed directly, so it is not handled here.

Restored files are byte-identical to the originals; each is verified against
the committed .sha256 digest and the run fails loudly if it does not match.
"""

from __future__ import annotations

import argparse
import hashlib
import lzma
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_CKPT_DIR = _REPO / "backend" / "data" / "paper_checkpoints"

# Files rebuilt by this script. Each is stored as either `<name>.xz` or, when
# the compressed form still exceeds the blob limit, `<name>.xz.part-*`.
_TARGETS = ("seacast_scripted.pt", "graph_fm_scripted.pt")

_CHUNK = 8 * 1024 * 1024


def _sources(target: Path) -> list[Path]:
    """The compressed input(s) for `target`, in concatenation order."""
    single = target.with_suffix(target.suffix + ".xz")
    if single.exists():
        return [single]
    return sorted(target.parent.glob(target.name + ".xz.part-*"))


def _expected_digest(target: Path) -> str | None:
    digest_file = target.with_suffix(target.suffix + ".sha256")
    if not digest_file.exists():
        return None
    return digest_file.read_text().strip().split()[0]


def _decompress(sources: list[Path], dest: Path) -> str:
    """Stream the concatenated parts through xz into `dest`; return its sha256."""
    sha = hashlib.sha256()
    decompressor = lzma.LZMADecompressor()
    with open(dest, "wb") as out:
        for part in sources:
            with open(part, "rb") as fh:
                while chunk := fh.read(_CHUNK):
                    plain = decompressor.decompress(chunk)
                    if plain:
                        sha.update(plain)
                        out.write(plain)
    if not decompressor.eof:
        raise RuntimeError(
            f"compressed stream for {dest.name} ended early — a part is missing or truncated"
        )
    return sha.hexdigest()


def restore(name: str, force: bool = False) -> bool:
    """Rebuild one checkpoint. Returns True if it was written, False if skipped."""
    target = _CKPT_DIR / name
    sources = _sources(target)

    if not sources:
        if target.exists():
            print(f"  {name}: no compressed parts, file already present — skipping")
            return False
        raise FileNotFoundError(
            f"{name}: neither the file nor its .xz parts exist under {_CKPT_DIR}"
        )

    expected = _expected_digest(target)

    if target.exists() and not force:
        if expected is None:
            print(f"  {name}: already present (no digest to check) — skipping")
            return False
        actual = hashlib.sha256()
        with open(target, "rb") as fh:
            while chunk := fh.read(_CHUNK):
                actual.update(chunk)
        if actual.hexdigest() == expected:
            print(f"  {name}: already restored and verified — skipping")
            return False
        print(f"  {name}: present but digest does not match — rebuilding")

    total_mb = sum(p.stat().st_size for p in sources) / 1048576
    print(f"  {name}: decompressing {len(sources)} part(s), {total_mb:.1f} MB …")

    # Write to a temporary name so an interrupted run never leaves a half-written
    # checkpoint that a later invocation would treat as present.
    tmp = target.with_suffix(target.suffix + ".tmp")
    try:
        digest = _decompress(sources, tmp)
        if expected is not None and digest != expected:
            raise RuntimeError(
                f"{name}: sha256 mismatch after decompression\n"
                f"  expected {expected}\n  got      {digest}"
            )
        tmp.replace(target)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise

    size_mb = target.stat().st_size / 1048576
    suffix = " (sha256 verified)" if expected is not None else ""
    print(f"  {name}: restored, {size_mb:.1f} MB{suffix}")
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument(
        "--force",
        action="store_true",
        help="rebuild even if the checkpoint is already present and verified",
    )
    args = ap.parse_args()

    print(f"Restoring checkpoints into {_CKPT_DIR}")
    written = 0
    for name in _TARGETS:
        written += restore(name, force=args.force)
    print(f"Done — {written} checkpoint(s) written, {len(_TARGETS) - written} already in place.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:  # noqa: BLE001 — surface a clean message, not a traceback
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)
