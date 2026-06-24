#!/usr/bin/env python3
"""
Image feature extraction utilities using pjreddie/darknet through ctypes.
"""

import cv2
import numpy as np
from pathlib import Path
import logging
from ctypes import (
    Structure,
    POINTER,
    pointer,
    c_int,
    c_float,
    c_void_p,
)

from darknet_detect import load_darknet_network
from darknet_detect import BOX,DETECTION,IMAGE
import darknet_detect as darknet


CLASS_NAMES = ["claw", "column", "base"]  
CONF_THRESHOLD = 0.45
NMS_THRESHOLD = 0.4


YOLO_FEATURE_NAMES = [
    "base_cx", "base_cy", "base_bw", "base_bh",
    "base_conf", "base_area", "base_valid",
    "col_cx", "col_cy", "col_bw", "col_bh",
    "col_conf", "col_area", "col_valid",
    "delta_x", "delta_y",
]


def _resolve_path(path: str | Path, base_dir: Path | None = None) -> Path:
    path = Path(path)

    if path.is_absolute():
        return path.resolve()

    if base_dir is None:
        base_dir = Path(".").resolve()

    return (base_dir / path).resolve()


def _frame_to_darknet_image(frame: np.ndarray) -> tuple[IMAGE, np.ndarray]:
    """
    Convierte un frame BGR de OpenCV a IMAGE de Darknet.

    Importante:
    - Darknet espera float32 en formato CHW.
    - OpenCV usa BGR, por eso convertimos a RGB.
    - Regresamos también el arreglo numpy para mantener viva la memoria.
    """

    if frame is None:
        raise ValueError("El frame es None")

    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f"Frame inválido. Se esperaba HxWx3, llegó: {frame.shape}")

    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w, c = rgb.shape

    chw = rgb.transpose(2, 0, 1)
    data_np = np.ascontiguousarray(chw, dtype=np.float32) / 255.0
    data_ptr = data_np.ctypes.data_as(POINTER(c_float))

    darknet_image = IMAGE(w=w, h=h, c=c, data=data_ptr)

    return darknet_image, data_np


class YOLODetector:
    """
    Wrapper para Tiny YOLO usando libdarknet.so de pjreddie/darknet.

    Returns:
        {
            "base": {
                "bbox": [x, y, w, h],
                "centroid": [cx, cy],
                "confidence": float,
                "area": int,
            },
            "column": {...},
        }
    """

    def __init__(
        self,
        weights: str,
        config: str,
        class_names: list[str] | None = None,
        conf_threshold: float = CONF_THRESHOLD,
        nms_threshold: float = NMS_THRESHOLD,
        gpu_id: int = 0,
    ):
        work_dir = Path(".").resolve()

        cfg_path = _resolve_path(config, work_dir)
        weights_path = _resolve_path(weights, work_dir)

        self.class_names = class_names if class_names is not None else CLASS_NAMES
        self.conf_threshold = float(conf_threshold)
        self.nms_threshold = float(nms_threshold)

        self.net = load_darknet_network(
            cfg_path=cfg_path,
            weights_path=weights_path,
            gpu_id=gpu_id,
        )


        logging.info(f"[YOLO] Loaded Darknet model")
        logging.info(f"[YOLO] cfg: {cfg_path}")
        logging.info(f"[YOLO] weights: {weights_path}")
        logging.info(f"[YOLO] classes: {self.class_names}")



    def detect(self, frame: np.ndarray) -> dict[str, dict]:
        h, w = frame.shape[:2]
    
        darknet_image, data_np = _frame_to_darknet_image(frame)
    
        # Mantener viva la memoria del frame convertido
        _keep_alive = data_np
    
        num = c_int(0)
        pnum = pointer(num)
        null_map = POINTER(c_int)()

        darknet.predict_image(self.net, darknet_image)

        dets,num,pnum = darknet.get_network_boxes_safe(
            self.net,
            w,
            h,
            (self.conf_threshold),
            (0.5),
        )

    
        num_dets = pnum[0]
    
        if not dets or num_dets == 0:
            return {}
    
        try:
            darknet.do_nms_sort(
                dets,
                num_dets,
                len(self.class_names),
                c_float(self.nms_threshold),
            )
    
            best: dict[str, dict] = {}
    
            for i in range(num_dets):
                det = dets[i]
    
                for cid, name in enumerate(self.class_names):
                    conf = float(det.prob[cid])
    
                    if conf < self.conf_threshold:
                        continue
    
                    bbox = det.bbox
    
                    bw = int(round(bbox.w))
                    bh = int(round(bbox.h))
                    x = int(round(bbox.x - bbox.w / 2))
                    y = int(round(bbox.y - bbox.h / 2))
    
                    x = max(0, min(x, w - 1))
                    y = max(0, min(y, h - 1))
                    bw = max(1, min(bw, w - x))
                    bh = max(1, min(bh, h - y))
    
                    if name not in best or conf > best[name]["confidence"]:
                        best[name] = {
                            "bbox": [x, y, bw, bh],
                            "centroid": [x + bw / 2, y + bh / 2],
                            "confidence": conf,
                            "area": bw * bh,
                        }

            return best
    
        finally:

            darknet.free_detections(dets, num_dets)

    def draw_overlay(self, frame: np.ndarray, detections: dict) -> np.ndarray:
        """
        Dibuja cajas y etiquetas.
        """

        out = frame.copy()
        colours = {
            "base": (0, 255, 120),
            "column": (0, 160, 255),
        }

        for name, det in detections.items():
            x, y, bw, bh = det["bbox"]
            colour = colours.get(name, (200, 200, 200))

            cv2.rectangle(out, (x, y), (x + bw, y + bh), colour, 2)

            label = f"{name} {det['confidence']:.2f}"
            cv2.putText(
                out,
                label,
                (x, max(y - 6, 12)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55, colour,
                2,
            )

        return out

class YOLOObservationProcessor:
    """
    Wraps the standard LeRobot robot_observation_processor and appends a YOLO
    feature vector to every observation dict before it is handed to record_loop.
 
    Usage:
        base_processor = make_default_processors()[2]   # robot_observation_processor
        yolo_processor = YOLOObservationProcessor(base_processor, detector)
        # Pass yolo_processor as robot_observation_processor to record_loop
    """
 
    def __init__(
        self,
        base_processor: any,
        detector: YOLODetector,
        debug_overlay: bool = False,
    ):
        self.base_processor  = base_processor
        self.detector        = detector
        # LeRobot stores images under this key pattern in the observation dict
        self.image_obs_key   = f"front"
        self.debug_overlay   = debug_overlay
        self._last_detections: dict = {}   # expose for external logging
 
    def __call__(self, observation: dict) -> dict:
        """
        1. Run the standard processor (handles image resizing, normalisation, etc.)
        2. Grab the processed camera frame and run YOLO inference.
        3. Inject `observation.yolo_features` into the output dict.
        """
        # Step 1 — standard processing
        processed = self.base_processor(observation)
 
        # Step 2 — YOLO inference on the designated camera frame
        frame = processed.get(self.image_obs_key)
        if frame is not None:
            # frame may be a torch.Tensor (C, H, W) or np.ndarray (H, W, C)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            if hasattr(frame, "numpy"):
                # Convert CHW float tensor → HWC uint8 for OpenCV
                np_frame = (frame.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            else:
                np_frame = frame
 
            detections = self.detector.detect(np_frame)
            self._last_detections = detections
 
            if self.debug_overlay:
                overlay = self.detector.draw_overlay(np_frame, detections)
                # Display in a separate window; press 'q' in the main rerun view to close
                cv2.imshow("YOLO detections", overlay)
                cv2.waitKey(1)
        else:
            logging.warning(
                f"[YOLO] Key '{self.image_obs_key}' not found in observation. "
                "Check that YOLO_CAMERA_KEY matches your camera config."
            )

            detections = {}
            self._last_detections = {}
 
        # Step 3 — append feature vector
        feat_vec = extract_feature_vector(detections, np_frame.shape if frame is not None else (480, 640))
        feat_vec = np.asarray(
            extract_feature_vector(detections, np_frame.shape if np_frame is not None else (480, 640)),
            dtype=np.float32,
        ).reshape(-1)
        
        if feat_vec.shape[0] != len(YOLO_FEATURE_NAMES):
            raise ValueError(
                f"YOLO feature length mismatch: got {feat_vec.shape[0]}, "
                f"expected {len(YOLO_FEATURE_NAMES)}"
            )

        for name, value in zip(YOLO_FEATURE_NAMES, feat_vec, strict=True):
            processed[name] = float(value)
 
        return processed

def extract_feature_vector(
    detections: dict,
    frame_shape: tuple,
    depth_map: np.ndarray = None,
) -> np.ndarray:
    """
    Returns a 16-dim vector regardless of which objects are detected.
    Missing detections are zero-filled.

    -> For each detection
    Absolute position X [0,4,8]
    Absolute position Y[1,5,9]
    Confidece[2,6,10]
    Validity[3,7,11]

    Distance claw to column [12, 13]
    Distance claw to base [14, 15]
    
    """
    H, W = frame_shape[:2]
    feat = np.zeros(16, dtype=np.float32)

    for idx, cls in enumerate(CLASS_NAMES): # For each YOLO class (claw, base, col)
        base_i = idx * 4
        det = detections.get(cls)

        if det is None:
            feat[base_i + 3] = 0.0
            continue

        x, y, bw, bh = det["bbox"]
        cx, cy = det["centroid"]

        feat[base_i + 0] = cx / W
        feat[base_i + 1] = cy / H
        feat[base_i + 2] = det["confidence"]
        feat[base_i + 3] = 1.0

    if "base" in detections and "column" in detections:
        bcx, bcy = detections["base"]["centroid"]
        ccx, ccy = detections["column"]["centroid"]
        gcx, gcy = detections["claw"]["centroid"]
        feat[12] = (ccx - gcx) / W # X dist to Column
        feat[13] = (ccy - gcy) / H # Y dist to Column

        feat[14] = (bcx - gcx) / W # X dist to Base
        feat[15] = (bcy - gcy) / H # X dist to Base

    return feat

