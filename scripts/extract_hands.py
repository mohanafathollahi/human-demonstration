from pathlib import Path

import cv2
import mediapipe as mp
import pandas as pd


VIDEO_DIR = Path("data/downloads")
OUTPUT_DIR = Path("data/processed")
MODEL_PATH = "models/hand_landmarker.task"

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

    options = HandLandmarkerOptions(
        base_options=BaseOptions(
            model_asset_path=MODEL_PATH
        ),
        running_mode=RunningMode.VIDEO,
        num_hands=1,
    )

    rows = []
    frame_id = 0

    with HandLandmarker.create_from_options(options) as landmarker:

        while True:

            success, frame = cap.read()

            if not success:
                break

            # Convert BGR to RGB
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

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

            row = {
                "video": video_path.name,
                "frame": frame_id,
                "time": frame_id / fps,
                "hand_detected": 0,
            }

            # Initialize landmarks as missing
            for i in range(21):
                row[f"x{i}"] = float("nan")
                row[f"y{i}"] = float("nan")
                row[f"z{i}"] = float("nan")

            if result.hand_landmarks:

                row["hand_detected"] = 1

                landmarks = result.hand_landmarks[0]

                for i, landmark in enumerate(landmarks):

                    row[f"x{i}"] = landmark.x
                    row[f"y{i}"] = landmark.y
                    row[f"z{i}"] = landmark.z

            rows.append(row)

            frame_id += 1

    cap.release()

    output_path = OUTPUT_DIR / f"{video_path.stem}.csv"

    pd.DataFrame(rows).to_csv(
        output_path,
        index=False
    )

    print(f"Saved: {output_path}")


def main():
    videos = sorted(VIDEO_DIR.glob("*.mp4"))
    if not videos:
        print("No MP4 videos found.")
        return

    for video in videos:
        process_video(video)


if __name__ == "__main__":
    main()