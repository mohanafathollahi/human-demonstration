from pathlib import Path

import cv2
import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent.parent

VIDEO_DIR = PROJECT_DIR / "data" / "videos"
CSV_DIR = PROJECT_DIR / "data" / "processed"
OUTPUT_DIR = PROJECT_DIR / "data" / "visualizations"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# MediaPipe hand landmark connections
HAND_CONNECTIONS = [
    # Thumb
    (0, 1), (1, 2), (2, 3), (3, 4),

    # Index finger
    (0, 5), (5, 6), (6, 7), (7, 8),

    # Middle finger
    (5, 9), (9, 10), (10, 11), (11, 12),

    # Ring finger
    (9, 13), (13, 14), (14, 15), (15, 16),

    # Pinky finger
    (13, 17), (17, 18), (18, 19), (19, 20),

    # Palm
    (0, 17),
]


def draw_hand(frame, row, hand, color):
    """Draw one hand's landmarks and skeleton."""

    if row[f"{hand}_detected"] != 1:
        return frame

    height, width = frame.shape[:2]

    points = []

    # Convert normalized coordinates to pixel coordinates
    for i in range(21):

        x = row[f"{hand}_x{i}"]
        y = row[f"{hand}_y{i}"]

        if pd.isna(x) or pd.isna(y):
            points.append(None)
            continue

        px = int(x * width)
        py = int(y * height)

        points.append((px, py))

    # Draw connections
    for start, end in HAND_CONNECTIONS:

        if points[start] is None or points[end] is None:
            continue

        cv2.line(
            frame,
            points[start],
            points[end],
            color,
            2,
        )

    # Draw landmarks
    for point in points:

        if point is None:
            continue

        cv2.circle(
            frame,
            point,
            4,
            color,
            -1,
        )

    # Label hand near wrist
    if points[0] is not None:

        x, y = points[0]

        cv2.putText(
            frame,
            hand.upper(),
            (x + 10, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )

    return frame


def visualize_video(video_path):

    csv_path = CSV_DIR / f"{video_path.stem}.csv"

    if not csv_path.exists():
        print(f"Missing CSV: {csv_path}")
        return

    print(f"Processing {video_path.name}")

    df = pd.read_csv(csv_path)

    if df.empty:
        print(f"Empty CSV: {csv_path}")
        return

    # Match CSV rows to video frames
    df = df.set_index("frame")

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)

    if not np.isfinite(fps) or fps <= 0:
        cap.release()
        raise ValueError("Invalid video FPS")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_path = OUTPUT_DIR / f"{video_path.stem}_annotated.mp4"

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot create {output_path}")

    frame_id = 0

    try:
        while True:

            success, frame = cap.read()

            if not success:
                break

            if frame_id in df.index:

                row = df.loc[frame_id]

                # Left hand: Green (BGR)
                frame = draw_hand(
                    frame,
                    row,
                    "left",
                    (0, 255, 0),
                )

                # Right hand: Blue (BGR)
                frame = draw_hand(
                    frame,
                    row,
                    "right",
                    (255, 0, 0),
                )

            # Display frame number
            cv2.putText(
                frame,
                f"Frame: {frame_id}",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )

            writer.write(frame)
            frame_id += 1

    finally:
        cap.release()
        writer.release()

    print(f"Saved: {output_path}")


def main():

    videos = sorted(VIDEO_DIR.glob("*.mp4"))

    if not videos:
        print("No videos found.")
        return

    for video_path in videos:
        visualize_video(video_path)


if __name__ == "__main__":
    main()