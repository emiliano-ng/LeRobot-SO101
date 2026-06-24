# lerobot_dataset_config.py
observation_keys = [
    "observation.state",              # (6,)   joint angles
    "observation.images",         # (H,W,3) raw
    "observation.images.column_crop", # (96,96,3) zoomed
    "observation.yolo_features",      # (16,)  <-- new
]
