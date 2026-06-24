from yolo_extract import YOLODetector
import cv2
import os
import time
from pathlib import Path
import av

WORK_DIR = Path(".").resolve()
CFG_PATH = WORK_DIR / "custom_cfg/yolov4-tiny-custom.cfg"
WEIGHTS_PATH = WORK_DIR / "backup/yolov4-tiny-custom_final.weights"
 

def iter_video_frames_av(video_path: Path):
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        stream.codec_context.thread_type = "AUTO"
        for packet in container.demux(stream):
            for frame in packet.decode():
                yield cv2.cvtColor(
                    frame.to_ndarray(format="rgb24"),
                    cv2.COLOR_RGB2BGR
                )

def main():
	# detector = YOLODetector('backup/yolov4-tiny-custom_final.weights', 'custom_cfg/yolov4-tiny-custom.cfg', use_gpu=False, conf_threshold=0.1)

	cfg = CFG_PATH
	weights = WEIGHTS_PATH
	detector = YOLODetector(WEIGHTS_PATH, CFG_PATH, conf_threshold=0.02)

	cap = cv2.VideoCapture(2)
#	cap = cv2.VideoCapture("./dataset/test_single_episode/videos/observation.images.front/chunk-000/file-000.mp4")

	while cap.isOpened():
		ret,frame = cap.read()

#	for frame in iter_video_frames_av("./dataset/test_single_episode/videos/observation.images.front/chunk-000/file-000.mp4"):
		if frame is None:
			print("noframe")
			continue	
		
		cv2.imshow("papu", frame)

		if hasattr(frame, "numpy"): 
             # Convert CHW float tensor → HWC uint8 for OpenCV
			 frame = (frame.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
		
		detected = detector.detect(frame)
		overlay = detector.draw_overlay(frame,detected)
		cv2.imshow("papu", overlay)
		cv2.waitKey(1)
		time.sleep(1/30) 


if __name__ == '__main__':
	main()
