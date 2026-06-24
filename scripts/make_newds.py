"""
make_test_dataset.py  —  extracts episode 0 into a new dataset for fast testing
"""
import json, shutil
from pathlib import Path
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

SRC  = Path("/home/angel/VISUAL/Pruebaconda/SO101/dataset/so101-pilares-blanco/angel02")
DST  = Path("/home/angel/VISUAL/Pruebaconda/SO101/dataset/test_single_episode")
EPISODE = 0

# ── 1. Parquet: keep only rows where episode_index == EPISODE ────────────────
DST.mkdir(parents=True, exist_ok=True)

src_parquets = sorted((SRC / "data" ).rglob("*.parquet"))
rows = []
for p in src_parquets:
    print(p)
    df = pd.read_parquet(p)
    print(df["episode_index"])
    rows.append(df[df["episode_index"] == EPISODE])

ep_df = pd.concat(rows, ignore_index=True)
# Reset frame_index to start from 0
ep_df["frame_index"] = range(len(ep_df))

out_data = DST / "data"
out_data.mkdir(parents=True, exist_ok=True)
pq.write_table(pa.Table.from_pandas(ep_df),
               out_data / "train-00000-of-00001.parquet")
print(f"[parquet] {len(ep_df)} rows written")

# ── 2. Videos: copy only the episode 0 video file(s) ────────────────────────
for src_video in SRC.rglob(f"*episode_{EPISODE:06d}*.mp4"):
    rel  = src_video.relative_to(SRC)
    dest = DST / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(src_video, dest)
    print(f"[video]   copied {rel}")

# Also handle file-NNN.mp4 naming (some LeRobot versions use file index not episode index)
# Find the video by looking at what the src dataset points to
try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    src_ds   = LeRobotDataset("emiliano-ng/so101-pilares004", root=SRC)
    vid_path = Path(src_ds.meta.get_video_file_path(
        EPISODE, f"observation.images.front"
    ))
    src_vid  = SRC / vid_path
    dst_vid  = DST / vid_path
    dst_vid.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(src_vid, dst_vid)
    print(f"[video]   copied {vid_path}")
except Exception as e:
    print(f"[video]   fallback copy done (LeRobot path: {e})")

# ── 3. Meta: copy and patch info.json + stats.json ──────────────────────────
(DST / "meta").mkdir(parents=True, exist_ok=True)

for meta_file in (SRC / "meta").glob("*.json"):
    shutil.copy(meta_file, DST / "meta" / meta_file.name)

# Patch info.json: set total_episodes = 1
info_path = DST / "meta" / "info.json"
with open(info_path) as f:
    info = json.load(f)
info["total_episodes"] = 1
info["total_frames"]   = len(ep_df)
with open(info_path, "w") as f:
    json.dump(info, f, indent=2)

print(f"\n[done]  Test dataset at {DST}")
print(f"        {len(ep_df)} frames, 1 episode")
