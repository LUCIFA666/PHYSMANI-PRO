"""Cache sampled position tracks from a pinned official GSCollision revision."""

import concurrent.futures
import hashlib
import io
import json
import re
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import h5py
import numpy as np
import requests
import torch

from dataset.constants import INDEX_OBJ, OBJPART
from utils.geometry_utils import farthest_point_sampling_indices

REVISION = "4db8e745c1c4f77cdfd74334110ba1156ac8270e"
BASE = f"https://huggingface.co/datasets/lishiqianhugh/GSCollision/resolve/{REVISION}/"
DEST = ROOT / "data/quality_smoke"
FRAMES = list(range(0, 61, 2))
KEYPOINTS = 256
LOCAL = threading.local()


def fetch(path, start=None, end=None):
    if not hasattr(LOCAL, "session"):
        LOCAL.session = requests.Session()
    headers = {"Range": f"bytes={start}-{end}"} if start is not None else {}
    error = None
    for attempt in range(6):
        try:
            # A distinct query for each range avoids stale intermediary range caches.
            query = f"?download=true&ngff_range={start}_{end}"
            response = LOCAL.session.get(BASE+path+query, headers=headers, timeout=(15, 45))
            response.raise_for_status()
            content = response.content
            if start is not None:
                match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", ""))
                if response.status_code == 200:
                    total = len(content)
                    assert end < total
                    return content[start:end+1], total
                assert match and int(match[1]) == start and int(match[2]) == end, "unexpected HTTP byte range"
                assert len(content) == end-start+1, "truncated byte range"
                return content, int(match[3])
            return content, len(content)
        except Exception as exc:
            # Do not log signed redirect URLs.
            error = type(exc).__name__
            print("RETRY", path, attempt+1, error, flush=True)
            time.sleep(min(2**attempt, 8))
    raise RuntimeError(f"cannot download {path}: {error}")


def layout(path):
    header, size = fetch(path, 0, 65535)
    virtual = io.BytesIO(header+bytes(size-len(header)))
    with h5py.File(virtual, "r") as file:
        dataset = file["pos"]
        assert dataset.chunks is None and dataset.compression is None
        assert dataset.ndim == 2 and dataset.shape[1] == 3
        return {"offset": dataset.id.get_offset(), "shape": list(dataset.shape),
                "dtype": dataset.dtype.str, "bytes": dataset.size*dataset.dtype.itemsize,
                "file_bytes": size}


def read_positions(path, spec):
    content, size = fetch(path, spec["offset"], spec["offset"]+spec["bytes"]-1)
    assert size == spec["file_bytes"], "frame layout changed"
    array = np.frombuffer(content, dtype=spec["dtype"]).reshape(spec["shape"])
    assert np.isfinite(array).all(), f"non-finite positions in {path}"
    return array, hashlib.sha256(content).hexdigest()


def main():
    torch.set_num_threads(4)
    DEST.mkdir(parents=True, exist_ok=True)
    configs = {}
    for group in ("3_0", "3_9"):
        path = DEST/f"{group}.json"
        if not path.exists():
            path.write_bytes(fetch(f"scene_configs/{group}.json")[0])
        configs[group] = json.loads(path.read_text())
    training = sorted(configs["3_0"], key=lambda name: int(name.split("_")[0]))[:10]
    testing = sorted(configs["3_9"], key=lambda name: int(name.split("_")[0]))[:2]
    selections = [("train", "3_0", name) for name in training[:8]]
    selections += [("validation", "3_0", name) for name in training[8:10]]
    selections += [("test", "3_9", name) for name in testing]
    manifest = {"dataset": "lishiqianhugh/GSCollision", "revision": REVISION,
                "seed": 0, "frame_indices": FRAMES, "original_dt": 0.02,
                "sampled_dt": 0.04, "keypoints_per_object": KEYPOINTS,
                "sampling": "upstream farthest_point_sampling_indices, seed=scene index",
                "scene_selection": "first 8 train / next 2 validation from 3_0; first 2 test from 3_9",
                "split_selections": selections, "scenes": []}
    (DEST/"manifest.json").write_text(json.dumps(manifest, indent=2))
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        for number, (split, group, name) in enumerate(selections):
            target = DEST/f"{group}_{name}.npz"
            provenance = target.with_suffix(".json")
            if target.exists() and provenance.exists():
                entry = json.loads(provenance.read_text())
                assert entry["split"] == split and entry["frame_indices"] == FRAMES
                manifest["scenes"].append(entry)
                print("CACHED", split, name, flush=True)
                continue
            path = f"mpm/{group}/{name}"
            first = layout(f"{path}/0000.h5")
            last = layout(f"{path}/{FRAMES[-1]:04d}.h5")
            assert first == last, "frame layouts differ"
            raw, initial_hash = read_positions(f"{path}/0000.h5", first)
            ids = configs[group][name]["scene_object_idxs"]
            objects = [INDEX_OBJ[i] for i in ids]
            counts = [OBJPART[obj] for obj in objects]
            assert sum(counts) == len(raw), f"object partition does not match {name}"
            object_ids = torch.repeat_interleave(torch.arange(len(counts)), torch.tensor(counts)).unsqueeze(-1)
            torch.manual_seed(number)
            indices = farthest_point_sampling_indices(torch.from_numpy(raw.astype(np.float32)), object_ids, KEYPOINTS).numpy()
            tracks = {0: raw[indices].astype(np.float32).reshape(len(objects), KEYPOINTS, 3)}
            hashes = {"0": initial_hash}
            futures = {pool.submit(read_positions, f"{path}/{frame:04d}.h5", first): frame for frame in FRAMES[1:]}
            completed = 1
            for future in concurrent.futures.as_completed(futures):
                frame = futures[future]
                positions, digest = future.result()
                tracks[frame] = positions[indices].astype(np.float32).reshape(len(objects), KEYPOINTS, 3)
                hashes[str(frame)] = digest
                completed += 1
                if completed % 10 == 0:
                    print("FRAMES", split, name, completed, "/", len(FRAMES), flush=True)
            points = np.stack([tracks[frame] for frame in FRAMES])
            assert np.isfinite(points).all()
            movement = np.linalg.norm(points[-1]-points[0], axis=-1).mean()
            assert movement > 1e-3, "static sequence cannot test dynamics"
            np.savez_compressed(target, points=points, indices=indices, frames=np.array(FRAMES), dt=np.float32(0.04))
            entry = {"split": split, "group": group, "name": name, "source": path,
                     "cache": str(target.relative_to(ROOT)), "frame_indices": FRAMES, "objects": objects,
                     "layout": first, "position_block_sha256": hashes,
                     "sample_indices_sha256": hashlib.sha256(indices.tobytes()).hexdigest(),
                     "mean_point_displacement": float(movement),
                     "cache_sha256": hashlib.sha256(target.read_bytes()).hexdigest()}
            provenance.write_text(json.dumps(entry, indent=2))
            manifest["scenes"].append(entry)
            (DEST/"manifest.json").write_text(json.dumps(manifest, indent=2))
            print("READY", split, name, objects, "mean movement", round(float(movement),4), flush=True)
    print("DATA READY", len(manifest["scenes"]), DEST/"manifest.json", flush=True)


if __name__ == "__main__":
    main()
