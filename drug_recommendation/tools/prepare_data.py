#!/usr/bin/env python3

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONTROL = ROOT / "data/controls/ctl_vehicle_default164_ctrl1.h5ad"
PARTS = ROOT / "data/controls/parts"
PART_MANIFEST = PARTS / "manifest.json"
BLOCK_BYTES = 8 * 1024 * 1024
PART_BYTES = 1024 * 1024 * 1024


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(BLOCK_BYTES), b""):
            digest.update(block)
    return digest.hexdigest()


def expected_control():
    manifest = json.loads((ROOT / "data_manifest.json").read_text())
    return next(entry for entry in manifest["files"] if entry["path"] == CONTROL.relative_to(ROOT).as_posix())


def split_control():
    expected = expected_control()
    if CONTROL.stat().st_size != expected["size"]:
        raise ValueError("Control file size does not match data_manifest.json")
    PARTS.mkdir(parents=True, exist_ok=True)
    if PART_MANIFEST.exists():
        raise FileExistsError("Parts already exist; use verify to check them.")
    entries = []
    whole_digest = hashlib.sha256()
    with CONTROL.open("rb") as incoming:
        remaining = expected["size"]
        while remaining:
            name = f"{CONTROL.name}.part{len(entries):03d}"
            destination = PARTS / name
            temporary = destination.with_name(destination.name + ".tmp")
            part_digest = hashlib.sha256()
            size = min(PART_BYTES, remaining)
            todo = size
            with temporary.open("wb") as outgoing:
                while todo:
                    block = incoming.read(min(BLOCK_BYTES, todo))
                    if not block:
                        raise EOFError("Control file ended before expected size")
                    outgoing.write(block)
                    part_digest.update(block)
                    whole_digest.update(block)
                    todo -= len(block)
            temporary.replace(destination)
            entries.append({"name": name, "size": size, "sha256": part_digest.hexdigest()})
            remaining -= size
            print(f"Created {name} ({size} bytes)", flush=True)
    if whole_digest.hexdigest() != expected["sha256"]:
        raise ValueError("Control file checksum does not match data_manifest.json")
    data = {"file": CONTROL.relative_to(ROOT).as_posix(), "size": expected["size"], "sha256": expected["sha256"], "parts": entries}
    temporary_manifest = PART_MANIFEST.with_suffix(".json.tmp")
    temporary_manifest.write_text(json.dumps(data, indent=2) + "\n")
    temporary_manifest.replace(PART_MANIFEST)


def load_parts_manifest():
    data = json.loads(PART_MANIFEST.read_text())
    expected = expected_control()
    if any(data[key] != expected[key] for key in ("size", "sha256")):
        raise ValueError("Parts manifest does not match data_manifest.json")
    names = [entry["name"] for entry in data["parts"]]
    if not names or len(set(names)) != len(names):
        raise ValueError("Parts manifest is empty or contains duplicate parts")
    for name in names:
        if Path(name).name != name or name in (".", ".."):
            raise ValueError("Invalid part name in manifest")
    if sum(entry["size"] for entry in data["parts"]) != data["size"]:
        raise ValueError("Parts manifest has an inconsistent total size")
    return data


def restore_control():
    expected = expected_control()
    if CONTROL.exists():
        if CONTROL.stat().st_size != expected["size"]:
            raise ValueError("Existing control file has an unexpected size; inspect it before replacing it")
        print("Complete control file already exists. Use verify for a full checksum check.")
        return
    data = load_parts_manifest()
    temporary = CONTROL.with_suffix(".h5ad.tmp")
    whole_digest = hashlib.sha256()
    created = False
    try:
        with temporary.open("xb") as outgoing:
            created = True
            for entry in data["parts"]:
                part = PARTS / entry["name"]
                if part.stat().st_size != entry["size"]:
                    raise ValueError(f"Wrong size: {entry['name']}. Run git lfs pull first.")
                part_digest = hashlib.sha256()
                with part.open("rb") as incoming:
                    for block in iter(lambda: incoming.read(BLOCK_BYTES), b""):
                        outgoing.write(block)
                        part_digest.update(block)
                        whole_digest.update(block)
                if part_digest.hexdigest() != entry["sha256"]:
                    raise ValueError(f"Checksum mismatch: {entry['name']}")
                print(f"Restored {entry['name']}", flush=True)
        if whole_digest.hexdigest() != data["sha256"]:
            raise ValueError("Restored control checksum mismatch")
        temporary.replace(CONTROL)
    except Exception:
        if created and temporary.exists():
            temporary.unlink()
        raise
    print(f"Restored and verified {CONTROL.relative_to(ROOT)}", flush=True)


def verify():
    manifest = json.loads((ROOT / "data_manifest.json").read_text())
    for entry in manifest["files"]:
        path = ROOT / entry["path"]
        if not path.is_file() or path.stat().st_size != entry["size"] or sha256(path) != entry["sha256"]:
            raise ValueError(f"Missing or damaged file: {entry['path']}")
        print(f"Verified {entry['path']}", flush=True)
    if PART_MANIFEST.exists():
        data = load_parts_manifest()
        for entry in data["parts"]:
            path = PARTS / entry["name"]
            if not path.is_file() or path.stat().st_size != entry["size"] or sha256(path) != entry["sha256"]:
                raise ValueError(f"Missing or damaged part: {entry['name']}")
            print(f"Verified {entry['name']}", flush=True)


def main():
    parser = argparse.ArgumentParser(description='Split/restore the complete control H5AD without changing any bytes.')
    parser.add_argument("action", choices=["split", "restore", "verify"])
    args = parser.parse_args()
    {"split": split_control, "restore": restore_control, "verify": verify}[args.action]()


if __name__ == "__main__":
    main()
