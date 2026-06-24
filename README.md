# SO101 — Imitation Learning for Robotic Manipulation with YOLO-Augmented Observations

A robotic manipulation system that learns to autonomously disassemble and store a LEGO column using an **SO101 leader-follower arm**, combining **imitation learning (ACT policy)** with a **custom YOLO-based perception pipeline**.

Built on the [LeRobot framework](https://github.com/huggingface/lerobot) (Hugging Face). Developed as a final project for TE3002B at Tecnológico de Monterrey.

---

## Overview

The goal was to train the SO101 robot to autonomously perform a complete pick-and-place sequence — detecting a LEGO column, grasping it, transporting it, and releasing it in a storage area — without human intervention, even when the object appears in positions not seen during training.

The key challenge: standard imitation learning takes raw camera frames as input, which makes it sensitive to position changes. Our solution was to augment the observation space with a compact YOLO feature vector that explicitly encodes object location and size, making the policy more robust to spatial variability.

**Results:** 40% pick success rate, 80% place success rate, 63% overall across 10 evaluation runs.

---

## My Contribution

This was a five-person team project. My specific contributions were:

- **YOLO integration into the LeRobot dataset pipeline** — designed and implemented the `add_yolo_to_dataset.py` script that post-processes recorded demonstrations, runs YOLO inference on every camera frame, and writes a 16-dimensional feature vector (`observation.yolo_features`) back into the Parquet dataset.
- **Feature vector design** — defined the 16-element representation encoding centroid position, bounding box dimensions, confidence, area, and relative delta between the base and column objects.
- **Training pipeline** — ran the ACT policy training using the LeRobot framework, monitored convergence with Weights & Biases, and iterated on dataset quality (250 collected → 50 selected for final training, 100k steps).
- **Dataset and model publication** on Hugging Face.

---

## System Architecture

```
RGB Camera
    │
    ▼
YOLO Detector (YOLOv4-tiny, Darknet)
    │  detects: "base", "column"
    ▼
16-dim Feature Vector
[base_cx, base_cy, base_bw, base_bh, base_conf, base_area, base_valid,
 col_cx,  col_cy,  col_bw,  col_bh,  col_conf,  col_area,  col_valid,
 delta_x, delta_y]
    │
    ├── + Robot joint states (6 DOF + gripper)
    ▼
ACT Policy (Action Chunking Transformer)
    │
    ▼
SO101 Follower Arm — executes action chunk
```

---

## Dataset

- **250 demonstrations** collected via teleoperation (leader-follower setup).
- **50 high-quality trajectories** selected for training after manual review.
- Each demonstration includes: RGB frames, joint states (6 joints + gripper), gripper state, action trajectories, and YOLO feature vectors.
- Variations in lighting, time of day, and object position were captured intentionally.

Dataset on Hugging Face: [emiliano-ng/SO101](https://huggingface.co/datasets/emiliano-ng/SO101)  
Trained model: [emiliano-ng/SO101_Model](https://huggingface.co/emiliano-ng/SO101_Model)

---

## Results

| Phase | Success Rate |
|-------|-------------|
| Pick  | 4/10 (40%)  |
| Place | 8/10 (80%)  |
| Overall | 63%       |

The place phase was significantly more reliable than the pick phase. The main failure mode in picking was grasp initiation — the policy struggled when the column appeared far from the centroid positions seen during training. This suggests the next step is a larger, more diverse dataset.

---

## Key Technical Decisions

**Why YOLO features instead of raw pixels?**  
ACT policies can learn from raw images, but they tend to overfit to spatial positions in the training data. By providing an explicit object-centric representation (centroids, bounding boxes, relative positions), the policy gets a more stable signal that generalizes better to unseen object positions.

**Why YOLOv4-tiny?**  
The SO101 setup runs on a standard workstation without a dedicated inference GPU. YOLOv4-tiny provides a good speed/accuracy tradeoff for a two-class problem (base, column) at this scale.

**Why 50 demonstrations out of 250?**  
Quality over quantity. Many demonstrations had abrupt joint movements, corrections, or inconsistent gripper timing. Filtering to the smoothest 50 produced more stable training dynamics and cleaner loss curves.

---

## Repository Structure

```
├── scripts/
│   ├── add_yolo_to_dataset.py   # Post-processing: adds YOLO features to LeRobot dataset
│   ├── recorder.py              # Records demonstrations with YOLO observations
│   ├── darknet_detect.py        # Darknet backend wrapper (ctypes)
│   ├── yolo_extract.py          # YOLO inference + feature vector extraction
│   └── lerobot_dataset_config.py # Observation key definitions
├── custom_cfg/
│   ├── yolov4-tiny-custom.cfg   # YOLO architecture config
│   ├── custom.names             # Class names: base, column
│   └── custom.data              # Darknet data config
├── backup/                      # YOLO training checkpoints (every 100 epochs)
├── Sample-Results/              # Parquet dataset sample + trajectory plots
│   └── chunk-000/
│       ├── file-000.parquet
│       └── states_episode_*.png
└── readme_images/               # Setup photos and trajectory visualizations
```

---

## Installation

```bash
# Install LeRobot (pinned commit)
pip install git+https://github.com/huggingface/lerobot@1396b9fab7aecddd10006c33c47a487ffdcb54b4

# Build Darknet with OpenCV support and place libdarknet.so in libs/
# See: https://github.com/AlexeyAB/darknet
```

To add YOLO features to an existing LeRobot dataset:

```bash
python scripts/add_yolo_to_dataset.py
```

---

## Limitations and Future Work

- The dataset is small (50 demonstrations) and was collected manually — performance degrades with larger position offsets.
- Camera angle changes have a disproportionate impact on detection quality.
- Next steps: larger dataset with more positional diversity, slower demonstration collection for smoother trajectories, exploration of depth camera integration.

---

## Team

Developed by Arturo Balboa, Oscar de la Rosa, Angel Hernandez, **Emiliano Niño**, and Rigoberto Soto as part of TE3002B at Tecnológico de Monterrey.