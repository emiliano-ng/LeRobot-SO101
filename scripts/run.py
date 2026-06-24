# In your data collection script (e.g. record_episodes.py)
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.feature_utils import hw_to_dataset_features
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
from lerobot.teleoperators.so_leader.config_so_leader import SO101LeaderConfig
from lerobot.teleoperators.so_leader.so_leader import SO101Leader
from lerobot.utils.control_utils import init_keyboard_listener
from lerobot.utils.utils import log_say
from lerobot.utils.visualization_utils import init_rerun
from lerobot.scripts.lerobot_record import record_loop
from lerobot.processor import make_default_processors
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.factory import make_pre_post_processors

from yolo_extract import YOLODetector, extract_feature_vector, YOLO_FEATURE_NAMES, YOLOObservationProcessor

import logging
from pathlib import Path
import cv2
import numpy as np


NUM_EPISODES = 5
FPS = 30
EPISODE_TIME_SEC = 27
RESET_TIME_SEC = 0
TASK_DESCRIPTION = "LEGO deconstructor"

WORK_DIR = Path(".").resolve()
CFG_PATH = WORK_DIR / "custom_cfg/yolov4-tiny-custom.cfg"
WEIGHTS_PATH = WORK_DIR / "backup/yolov4-tiny-custom_final.weights"
PRETRAINED_DIR = WORK_DIR / "outputs/angel01/checkpoints/100000/pretrained_model/"

PORT_LEADER = "/dev/ttyACM0"
PORT_FOLLOW = "/dev/ttyACM1"
CAM_INDEX = 2

YOLO_CONF_THRESHOLD = 0.2
YOLO_NMS_THRESHOLD  = 0.4
YOLO_CLASS_NAMES = ["base", "column"]               # must match your training order
YOLO_CAMERA_KEY  = "front"                          # which camera to run YOLO on
YOLO_FEATURE_DIM = 16                               # fixed vector length — do not change
 
# Set True to draw bounding boxes on the camera feed (useful during recording)
YOLO_DEBUG_OVERLAY = True
 
def make_yolo_dataset_feature() -> dict:
    """
    Returns a LeRobot-compatible feature descriptor for the YOLO vector.
 
    LeRobot uses the same format as HuggingFace datasets under the hood.
    A 1-D float32 sequence is described with dtype, shape, and optional names.
    """
    return {
        "observation.yolo_features": {
            "dtype": "float32",
            "shape": (YOLO_FEATURE_DIM,),
            "names": YOLO_FEATURE_NAMES,
        }
    }


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
 
    detector = YOLODetector(WEIGHTS_PATH, CFG_PATH)

    # Create robot configuration
    robot_config = SO101FollowerConfig(
        id="follower_dlr",
        cameras={
            "front": OpenCVCameraConfig(index_or_path=CAM_INDEX, width=640, height=480, fps=FPS) 
        },
        port=PORT_FOLLOW,
    )

    # Initialize the robot and teleoperator
    robot = SO101Follower(robot_config)

    # Configure the dataset features
    action_features = hw_to_dataset_features(robot.action_features, "action")
    obs_features = hw_to_dataset_features(robot.observation_features, "observation")
    yolo_features = make_yolo_dataset_feature()
    dataset_features = {**action_features, **obs_features, **yolo_features}

    from lerobot.policies.act.modeling_act import ACTPolicy

    pol_config = ACTPolicy.from_pretrained(str(PRETRAINED_DIR))

    # Create the dataset
    dataset = LeRobotDataset.create(
        repo_id="emiliano-ng/eval_act_your_dataset",
        fps=FPS,
        features=dataset_features,
        robot_type=robot.name,
        use_videos=True,
        image_writer_threads=4,
    )

    # Initialize the keyboard listener and rerun visualization
    _, events = init_keyboard_listener()
    init_rerun(session_name="recording")

    # Connect the robot and teleoperator
    robot.connect()

    # Create the required processors
    teleop_action_processor , robot_action_processor, robot_observation_processor = make_default_processors()

    # Wrap the standard observation processor with the YOLO layer
    yolo_observation_processor = YOLOObservationProcessor(
        base_processor=robot_observation_processor,
        detector=detector,
        debug_overlay=YOLO_DEBUG_OVERLAY,
    )

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=pol_config.config,
        pretrained_path=str(PRETRAINED_DIR),
    )
    pol_config.eval()

    episode_idx = 0
    while episode_idx < NUM_EPISODES and not events["stop_recording"]:
        log_say(f"Recording episode {episode_idx + 1} of {NUM_EPISODES}")

        record_loop(
            robot=robot,
            events=events,
            fps=FPS,
            teleop_action_processor=teleop_action_processor,
            robot_action_processor=robot_action_processor,
            robot_observation_processor=yolo_observation_processor,
            dataset=dataset,
            control_time_s=EPISODE_TIME_SEC,
            single_task=TASK_DESCRIPTION,
            display_data=True,
            policy=pol_config,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
        )

        # Reset the environment if not stopping or re-recording
        if not events["stop_recording"] and (episode_idx < NUM_EPISODES - 1 or events["rerecord_episode"]):
            log_say("Reset the environment")
            record_loop(
                robot=robot,
                events=events,
                fps=FPS,
                teleop_action_processor=teleop_action_processor,
                robot_action_processor=robot_action_processor,
                robot_observation_processor=robot_observation_processor,
                control_time_s=RESET_TIME_SEC,
                single_task=TASK_DESCRIPTION,
                display_data=True,
                policy=pol_config,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
            )

        if events["rerecord_episode"]:
            log_say("Re-recording episode")
            events["rerecord_episode"] = False
            events["exit_early"] = False
            dataset.clear_episode_buffer()
            continue

        dataset.save_episode()
        episode_idx += 1

    # Clean up
    log_say("Stop recording")
    robot.disconnect()

if __name__ == '__main__':
    main()
