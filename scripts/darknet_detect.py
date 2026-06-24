#!/usr/bin/env python3
from ctypes import CDLL, RTLD_GLOBAL, c_char_p, c_int, c_void_p
from pathlib import Path
import os
from ctypes import (
    Structure,
    POINTER,
    pointer,
    c_int,
    c_float,
    c_void_p,
	byref,	
)

# Importante: cargar la librería con RTLD_GLOBAL
darknet_dir = Path(".").resolve()

lib_path = darknet_dir / "libs/libdarknet.so"

print(str(lib_path))
lib = CDLL(str(lib_path), mode=RTLD_GLOBAL)

class METADATA(Structure):
    _fields_ = [("classes", c_int),
                ("names", POINTER(c_char_p))]

class BOX(Structure):
    _fields_ = [
        ("x", c_float),
        ("y", c_float),
        ("w", c_float),
        ("h", c_float),
    ]


class DETECTION(Structure):
    _fields_ = [
        ("bbox", BOX),
        ("classes", c_int),
        ("prob", POINTER(c_float)),
        ("mask", POINTER(c_float)),
        ("objectness", c_float),
        ("sort_class", c_int),
    ]


class IMAGE(Structure):
    _fields_ = [
        ("w", c_int),
        ("h", c_int),
        ("c", c_int),
        ("data", POINTER(c_float)),
    ]

lib.network_width.argtypes = [c_void_p]
lib.network_width.restype = c_int
network_width = lib.network_width
lib.network_height.argtypes = [c_void_p]
lib.network_height.restype = c_int
network_height = lib.network_height

predict = lib.network_predict
predict.argtypes = [c_void_p, POINTER(c_float)]
predict.restype = POINTER(c_float)

set_gpu = lib.cuda_set_device
set_gpu.argtypes = [c_int]

make_image = lib.make_image
make_image.argtypes = [c_int, c_int, c_int]
make_image.restype = IMAGE

lib.get_network_boxes.argtypes = [c_void_p, c_int, c_int, c_float, c_float, POINTER(c_int), c_int, POINTER(c_int)]
lib.get_network_boxes.restype = POINTER(DETECTION)

make_network_boxes = lib.make_network_boxes
make_network_boxes.argtypes = [c_void_p]
make_network_boxes.restype = POINTER(DETECTION)

free_detections = lib.free_detections
free_detections.argtypes = [POINTER(DETECTION), c_int]

free_ptrs = lib.free_ptrs
free_ptrs.argtypes = [POINTER(c_void_p), c_int]

network_predict = lib.network_predict
network_predict.argtypes = [c_void_p, POINTER(c_float)]

reset_rnn = lib.reset_rnn
reset_rnn.argtypes = [c_void_p]

load_net = lib.load_network
load_net.argtypes = [c_char_p, c_char_p, c_int]
load_net.restype = c_void_p

do_nms_obj = lib.do_nms_obj
do_nms_obj.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

do_nms_sort = lib.do_nms_sort
do_nms_sort.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

free_image = lib.free_image
free_image.argtypes = [IMAGE]

letterbox_image = lib.letterbox_image
letterbox_image.argtypes = [IMAGE, c_int, c_int]
letterbox_image.restype = IMAGE

load_meta = lib.get_metadata
lib.get_metadata.argtypes = [c_char_p]
lib.get_metadata.restype = METADATA

load_image = lib.load_image_color
load_image.argtypes = [c_char_p, c_int, c_int]
load_image.restype = IMAGE

rgbgr_image = lib.rgbgr_image
rgbgr_image.argtypes = [IMAGE]

predict_image = lib.network_predict_image
predict_image.argtypes = [c_void_p, IMAGE]
predict_image.restype = POINTER(c_float)

def load_darknet_network(
    darknet_dir=".",
    cfg_path=None,
    weights_path=None,
    gpu_id=0,
):
    darknet_dir = Path(".").resolve()

    if cfg_path is None:
        cfg_path = darknet_dir / "custom_cfg/yolov4-tiny-custom.cfg"

    else:
        cfg_path = Path(cfg_path).resolve()

    if weights_path is None:
        weights_path = darknet_dir / "backup/yolov4-tiny-custom_final.weights"
    else:
        weights_path = Path(weights_path).resolve()

    for p in [lib_path, cfg_path, weights_path]:
        if not p.exists():
            raise FileNotFoundError(f"No existe: {p}")


    # Si Darknet fue compilado con GPU=1
    try:
        lib.cuda_set_device.argtypes = [c_int]
        lib.cuda_set_device.restype = None
        lib.cuda_set_device(gpu_id)
    except AttributeError:
        print("Sin cuda_set_device, probablemente compilado sin GPU.")

    # Importante: declarar bien la firma de load_network
    lib.load_network.argtypes = [c_char_p, c_char_p, c_int]
    lib.load_network.restype = c_void_p

    if hasattr(lib, "do_nms_sort"):
        lib.do_nms_sort.argtypes = [
            POINTER(DETECTION),
            c_int,
            c_int,
            c_float,
        ]
        lib.do_nms_sort.restype = None
    elif hasattr(lib, "do_nms_obj"):
        lib.do_nms_obj.argtypes = [
            POINTER(DETECTION),
            c_int,
            c_int,
            c_float,
        ]
        lib.do_nms_obj.restype = None
        _nms_func = self.lib.do_nms_obj
    else:
        raise RuntimeError("No encontré do_nms_sort ni do_nms_obj en libdarknet.so")

    lib.free_detections.argtypes = [POINTER(DETECTION), c_int]
    lib.free_detections.restype = None

    cfg = os.fsencode(str(cfg_path))
    weights = os.fsencode(str(weights_path))

    print("Cargando red Darknet...")
    net = lib.load_network(cfg, weights, 0)

    if not net:
        raise RuntimeError("load_network regresó NULL")

    print("Red cargada OK:", net)

    # net = el puntero a la red
    return  net

def get_network_boxes_safe(
    net,
    frame_w,
    frame_h,
    conf_threshold=0.45,
    hier_threshold=0.5,
    relative=0,
):
    num = c_int(0)
    null_map = POINTER(c_int)()
    pnum = pointer(num)

    dets = lib.get_network_boxes(
        net,
        c_int(frame_w),
        c_int(frame_h),
        c_float(conf_threshold),
        c_float(hier_threshold),
        null_map,
        c_int(relative),
        pnum
    )

    return dets, num.value, pnum
