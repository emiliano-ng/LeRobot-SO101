"""
check_yolo_column.py  —  confirm the feature is actually in the parquet
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT       = Path("/home/angel/VISUAL/Pruebaconda/SO101/dataset/so101-pilares-blanco/angel01")
OUTPUT_KEY = "observation.yolo_features"

# ── 1. Check every parquet file ───────────────────────────────────────────────
print("── Parquet column audit ─────────────────────────────────────────")
for p in sorted(ROOT.rglob("*.parquet")):
    df = pd.read_parquet(p)
    has = OUTPUT_KEY in df.columns
    val = df[OUTPUT_KEY].iloc[0] if has else None
    print(f"  {'✓' if has else '✗'}  {p.relative_to(ROOT)}")
    if has:
        print(f"       row[0] type : {type(val)}")
        print(f"       row[0] len  : {len(val)}")
        print(f"       row[0] val  : {val[:4]} …")

# ── 2. Load the data parquet and print stats ──────────────────────────────────
print("\n── Feature statistics ───────────────────────────────────────────")
data_pqs = sorted((ROOT / "data").rglob("*.parquet"))
chunks   = [pd.read_parquet(p, columns=["episode_index", OUTPUT_KEY])
            for p in data_pqs]
df_all   = pd.concat(chunks, ignore_index=True)

arr = np.array(df_all[OUTPUT_KEY].tolist(), dtype=np.float32)  # (N, 16)
print(f"  shape            : {arr.shape}")
print(f"  base detected    : {arr[:, 6].mean()*100:.1f}% of frames")
print(f"  column detected  : {arr[:, 13].mean()*100:.1f}% of frames")
print(f"  delta_x mean±std : {arr[:,14].mean():+.3f} ± {arr[:,14].std():.3f}")
print(f"  delta_y mean±std : {arr[:,15].mean():+.3f} ± {arr[:,15].std():.3f}")
print(f"  any NaN          : {np.isnan(arr).any()}")
print(f"  any None rows    : {df_all[OUTPUT_KEY].isna().sum()}")
