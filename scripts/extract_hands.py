from pathlib import Path

import cv2
import mediapipe as mp
import pandas as pd
import numpy as np


# Project directories
PROJECT_DIR = Path(__file__).resolve().parent.parent
print(f"proj_dir:{PROJECT_DIR}")
VIDEO_DIR = PROJECT_DIR / "data" / "videos"
OUTPUT_DIR = PROJECT_DIR / "data" / "processed"
MODEL_PATH = PROJECT_DIR / "models" / "hand_landmarker.task"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode


def process_video(video_path):

    print(f"Processing: {video_path.name}")

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        cap.release()
        raise ValueError("Invalid video FPS")

    # Allow detection of two hands
    options = HandLandmarkerOptions(
        base_options=BaseOptions(
            model_asset_path=str(MODEL_PATH)
        ),
        running_mode=RunningMode.VIDEO,
        num_hands=2,
    )

    rows = []
    frame_id = 0

    try:
        with HandLandmarker.create_from_options(options) as landmarker:

            while True:

                success, frame = cap.read()

                if not success:
                    break

                rgb = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB
                )

                image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb
                )

                timestamp_ms = int(
                    round(frame_id * 1000 / fps)
                )

                result = landmarker.detect_for_video(
                    image,
                    timestamp_ms
                )

                # Initialize one row per frame
                row = {
                    "video": video_path.name,
                    "frame": frame_id,
                    "time": frame_id / fps,
                    "left_detected": 0,
                    "right_detected": 0,
                    "left_confidence": np.nan,
                    "right_confidence": np.nan,
                }

                # Initialize all 42 landmarks as missing
                for hand in ["left", "right"]:
                    for i in range(21):

                        row[f"{hand}_x{i}"] = np.nan
                        row[f"{hand}_y{i}"] = np.nan
                        row[f"{hand}_z{i}"] = np.nan

                # Process each detected hand
                for hand_idx, landmarks in enumerate(
                    result.hand_landmarks
                ):

                    if not result.handedness[hand_idx]:
                        continue

                    category = result.handedness[hand_idx][0]

                    hand_label = category.category_name.lower()
                    confidence = float(category.score)

                    if hand_label not in ["left", "right"]:
                        continue

                    row[f"{hand_label}_detected"] = 1
                    row[f"{hand_label}_confidence"] = confidence

                    for i, landmark in enumerate(landmarks):

                        row[f"{hand_label}_x{i}"] = landmark.x
                        row[f"{hand_label}_y{i}"] = landmark.y
                        row[f"{hand_label}_z{i}"] = landmark.z

                rows.append(row)
                frame_id += 1

    finally:
        cap.release()

    output_path = OUTPUT_DIR / f"{video_path.stem}.csv"

    pd.DataFrame(rows).to_csv(
        output_path,
        index=False
    )

    print(f"Saved: {output_path}")


def main():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"MediaPipe model not found: {MODEL_PATH}"
        )

    videos = sorted(VIDEO_DIR.glob("*.mp4"))

    if not videos:
        print("No MP4 videos found.")
        return

    for video in videos:
        process_video(video)


if __name__ == "__main__":
    main()