"""
add_yolo_to_dataset.py
─────────────────────
Post-processing script that reads an existing LeRobot dataset, runs your
YOLO detector on every camera frame, and writes the resulting feature
vectors back as a new column: `observation.yolo_features`.

Steps performed:
  1. Load the dataset from a local snapshot.
  2. For each episode, open the corresponding video file with PyAV (AV1 support).
  3. Run YOLO frame-by-frame and collect a (N, 16) feature array.
  4. Add the new column to the underlying HuggingFace dataset.
  5. Update meta/info.json so the schema stays consistent.
  6. Recompute per-feature statistics (mean/std/min/max).
  7. Save locally and optionally push back to the Hub.

Usage:
    python add_yolo_to_dataset.py
"""

import json
import shutil
from pathlib import Path

import av
import cv2
import numpy as np
from datasets import Dataset
from tqdm import tqdm
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from lerobot.datasets.lerobot_dataset import LeRobotDataset

# ── Your detector ─────────────────────────────────────────────────────────────
from yolo_extract import YOLODetector, extract_feature_vector

WORK_DIR = Path('.').resolve()
WEIGHTS  = WORK_DIR / "backup/yolov4-tiny-custom_final.weights"
CFG_FILE = WORK_DIR / "custom_cfg/yolov4-tiny-custom.cfg"

DETECTOR = YOLODetector(WEIGHTS, CFG_FILE)

# ── Configuration ─────────────────────────────────────────────────────────────
REPO_ID       = "emiliano-ng/so101-pilares004"
LOCAL_ROOT    = WORK_DIR / "dataset/test_single_episode"
CAMERA_KEY    = "front"
FEATURE_DIM   = 16
FEATURE_NAMES = [
    "base_cx", "base_cy", "base_bw", "base_bh", "base_conf", "base_area", "base_valid",
    "col_cx",  "col_cy",  "col_bw",  "col_bh",  "col_conf",  "col_area",  "col_valid",
    "delta_x", "delta_y",
]
OUTPUT_KEY = "observation.yolo_features"
PUSH_TO_HUB = False


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Load dataset
# ─────────────────────────────────────────────────────────────────────────────

def load_dataset(repo_id: str, root=None) -> LeRobotDataset:
    if root is None:
        root = LOCAL_ROOT
    print(f"[dataset] Loading from {root} …")
    dataset = LeRobotDataset(repo_id, root=root)
    print(f"[dataset] {len(dataset)} frames across "
          f"{dataset.meta.total_episodes} episodes")
    return dataset


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Locate video files
# ─────────────────────────────────────────────────────────────────────────────

def video_path_for_episode(dataset: LeRobotDataset, episode_idx: int) -> Path:
    video_key = f"observation.images.{CAMERA_KEY}"
    root = Path(dataset.root)
    try:
        relative = Path(dataset.meta.get_video_file_path(episode_idx, video_key))
        # get_video_file_path returns a relative path — make it absolute
        path = root / relative
    except AttributeError:
        matches = sorted(root.rglob(f"*{video_key}*/{episode_idx:06d}.mp4"))
        if not matches:
            raise FileNotFoundError(
                f"Cannot locate video for episode {episode_idx}, key {video_key}.\n"
                f"Searched under: {root}\n"
                "Check that CAMERA_KEY matches your recording config."
            )
        path = matches[0]

    if not path.exists():
        raise FileNotFoundError(
            f"Video path resolved to {path} but file does not exist.\n"
            f"dataset.root = {root}"
        )

    return path

# ─────────────────────────────────────────────────────────────────────────────
# 3.  Decode video frames with PyAV (supports AV1, H.264, H.265, …)
# ─────────────────────────────────────────────────────────────────────────────

def iter_video_frames_av(video_path: Path):
    """
    Yield BGR uint8 numpy frames using PyAV / FFmpeg.
    Handles AV1 and any other codec OpenCV cannot decode.
    Install with:  pip install av
    """
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        stream.codec_context.thread_type = "AUTO"
        for packet in container.demux(stream):
            for frame in packet.decode():
                bgr = cv2.cvtColor(
                    frame.to_ndarray(format="rgb24"),
                    cv2.COLOR_RGB2BGR
                )
                yield bgr


def process_episode(video_path: Path, expected_frames: int) -> np.ndarray:
    features = []
    try:
        for frame in iter_video_frames_av(video_path):
            detections = DETECTOR.detect(frame)
            feat = extract_feature_vector(detections, frame.shape)
            features.append(feat)
    except Exception as e:
        print(f"  [error] Could not read {video_path}: {e}")

    n_read = len(features)
    if n_read != expected_frames:
        print(f"  [warn] video has {n_read} frames, "
              f"dataset expects {expected_frames}. Aligning …")

    if n_read == 0:
        return np.zeros((expected_frames, FEATURE_DIM), dtype=np.float32)

    features_arr = np.array(features, dtype=np.float32)   # (n_read, 16)

    if n_read < expected_frames:
        pad = np.zeros((expected_frames - n_read, FEATURE_DIM), dtype=np.float32)
        features_arr = np.vstack([features_arr, pad])
    else:
        features_arr = features_arr[:expected_frames]

    return features_arr   # (expected_frames, 16)


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Build the full feature column aligned to the dataset index
# ─────────────────────────────────────────────────────────────────────────────

def build_feature_column(dataset: LeRobotDataset) -> list[list[float]]:
    """
    Returns a list of length len(dataset), each element being a 16-float list.
    The order matches the global frame index in dataset.hf_dataset.
    """
    all_features: list[np.ndarray] = []

    for ep_idx in tqdm(range(dataset.meta.total_episodes),
                       desc="Processing episodes"):
        ep_data  = dataset.hf_dataset.filter(
            lambda row, i=ep_idx: row["episode_index"] == i
        )
        n_frames = len(ep_data)

        video_path = video_path_for_episode(dataset, ep_idx)
        ep_feats   = process_episode(video_path, n_frames)
        all_features.append(ep_feats)

    stacked = np.concatenate(all_features, axis=0)   # (N_total, 16)

    assert len(stacked) == len(dataset), (
        f"Feature count {len(stacked)} != dataset length {len(dataset)}"
    )

    return stacked.tolist()


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Patch the HuggingFace dataset with the new column
# ─────────────────────────────────────────────────────────────────────────────

def save_to_parquet(dataset: LeRobotDataset,
                    feature_column: list[list[float]]) -> None:
    root = Path(dataset.root)

    # ── Build a lookup: episode_index → list of feature rows ─────────────────
    ep_indices   = np.array(dataset.hf_dataset["episode_index"])
    features_arr = np.array(feature_column, dtype=object)   # keep as list-of-lists

    # Group rows by episode
    from collections import defaultdict
    ep_features: dict[int, list] = defaultdict(list)
    for i, ep_idx in enumerate(ep_indices):
        ep_features[int(ep_idx)].append(feature_column[i])

    # ── 1. data/ parquets (per-episode or single file) ────────────────────────
    data_parquets = sorted((root / "data").rglob("*.parquet"))
    print(f"[parquet] data/ → {len(data_parquets)} file(s)")

    offset = 0
    for p in data_parquets:
        df = pd.read_parquet(p)
        if "episode_index" not in df.columns:
            print(f"  [skip] {p.relative_to(root)} — no episode_index")
            continue

        n = len(df)
        chunk = feature_column[offset : offset + n]
        if OUTPUT_KEY in df.columns:
            df = df.drop(columns=[OUTPUT_KEY])
        df[OUTPUT_KEY] = chunk

        shutil.copy(p, p.with_suffix(".parquet.bak"))
        pq.write_table(pa.Table.from_pandas(df), p)
        print(f"  ✓ data  {p.relative_to(root)}  ({n} rows)")
        offset += n

    # ── 2. meta/episodes/ parquets (episode-level metadata) ──────────────────
    meta_ep_dir = root / "meta" / "episodes"
    if meta_ep_dir.exists():
        meta_parquets = sorted(meta_ep_dir.rglob("*.parquet"))
        print(f"[parquet] meta/episodes/ → {len(meta_parquets)} file(s)")

        for p in meta_parquets:
            df = pd.read_parquet(p)

            # These files have one row per episode, not per frame.
            # They don't store per-frame features — but the visualiser
            # reads their schema to know what columns exist.
            # Add a dummy placeholder column so the schema is consistent.
            if "episode_index" in df.columns:
                # Per-episode file: one row per episode — store feature stats
                if OUTPUT_KEY not in df.columns:
                    # Add a representative value (mean of episode's features)
                    def ep_mean(ep_idx):
                        rows = ep_features.get(int(ep_idx), [])
                        if not rows:
                            return [0.0] * FEATURE_DIM
                        return np.array(rows, dtype=np.float32).mean(axis=0).tolist()

                    df[OUTPUT_KEY] = df["episode_index"].apply(ep_mean)
                    shutil.copy(p, p.with_suffix(".parquet.bak"))
                    pq.write_table(pa.Table.from_pandas(df), p)
                    print(f"  ✓ meta  {p.relative_to(root)}  ({len(df)} episodes)")
            else:
                print(f"  [skip] {p.relative_to(root)} — unknown schema")
    else:
        print("[parquet] meta/episodes/ not found — skipping")

    # ── 3. meta/episodes.jsonl (some versions use this instead) ──────────────
    episodes_jsonl = root / "meta" / "episodes.jsonl"
    if episodes_jsonl.exists():
        print("[meta] episodes.jsonl found — no schema change needed there")



# ─────────────────────────────────────────────────────────────────────────────
# 6.  Update meta/info.json
# ─────────────────────────────────────────────────────────────────────────────

def update_info_json(dataset: LeRobotDataset) -> None:
    info_path = Path(dataset.root) / "meta" / "info.json"
    if not info_path.exists():
        print(f"[meta] info.json not found at {info_path} — skipping update.")
        return

    with open(info_path) as f:
        info = json.load(f)

    info.setdefault("features", {})[OUTPUT_KEY] = {
        "dtype": "float32",
        "shape": [FEATURE_DIM],
        "names": FEATURE_NAMES,
    }

    shutil.copy(info_path, info_path.with_suffix(".json.bak"))
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)
    print(f"[meta] Updated {info_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 7.  Recompute statistics for the new feature
# ─────────────────────────────────────────────────────────────────────────────

def update_stats_json(dataset: LeRobotDataset,
                      feature_column: list[list[float]]) -> None:
    stats_path = Path(dataset.root) / "meta" / "stats.json"
    if not stats_path.exists():
        print(f"[meta] stats.json not found at {stats_path} — skipping.")
        return

    arr = np.array(feature_column, dtype=np.float32)   # (N, 16)

    new_stats = {
        "mean": arr.mean(axis=0).tolist(),
        "std":  arr.std(axis=0).tolist(),
        "min":  arr.min(axis=0).tolist(),
        "max":  arr.max(axis=0).tolist(),
    }

    with open(stats_path) as f:
        stats = json.load(f)

    stats[OUTPUT_KEY] = new_stats
    shutil.copy(stats_path, stats_path.with_suffix(".json.bak"))
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"[meta] Updated {stats_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 8.  Save locally and optionally push to Hub
# ─────────────────────────────────────────────────────────────────────────────

def save_and_push(dataset: LeRobotDataset, hf_patched: Dataset) -> None:
    root = Path(dataset.root)

    print("[dataset] Saving patched dataset locally …")
    hf_patched.save_to_disk(str(root / "data"))
    hf_patched.to_parquet(str(root / "data" / "train-00000-of-00001.parquet"))

    if PUSH_TO_HUB:
        print("[dataset] Pushing to Hub …")
        hf_patched.push_to_hub(
            repo_id=REPO_ID,
            token=None,   # uses cached token from `huggingface-cli login`
        )
        print(f"[dataset] Pushed to https://huggingface.co/datasets/{REPO_ID}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    dss = [
        WORK_DIR / "dataset/so101-pilares-blanco/angel01/",
        WORK_DIR / "dataset/so101-pilares-blanco/angel02/",
        WORK_DIR / "dataset/so101-pilares-blanco/angel03/",
        WORK_DIR / "dataset/so101-pilares-blanco/angel04/",
    ]

    for i in dss:
        dataset = load_dataset(REPO_ID,i)
        print(dataset.features)
    
        print("[yolo] Running detector over all episodes …")
        feature_column = build_feature_column(dataset)
    
        save_to_parquet(dataset, feature_column)
    
        update_info_json(dataset)
        update_stats_json(dataset, feature_column)

    
        print("\n[done] Dataset updated successfully.")
        print(f"       New column : {OUTPUT_KEY}  shape=({FEATURE_DIM},)")


if __name__ == "__main__":
    main()
