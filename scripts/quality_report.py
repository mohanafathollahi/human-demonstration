from pathlib import Path

import pandas as pd


DATA_DIR = Path("data/processed")

results = []

for file in sorted(DATA_DIR.glob("*.csv")):

    df = pd.read_csv(file)

    total_frames = len(df)

    if total_frames == 0:
        continue

    detected_frames = df["hand_detected"].sum()

    detection_rate = detected_frames / total_frames

    results.append({
        "video": file.stem,
        "total_frames": total_frames,
        "detected_frames": int(detected_frames),
        "detection_rate": round(detection_rate, 3),
    })


report = pd.DataFrame(results)

if report.empty:
    print("No processed video data found.")
else:
    print(report.to_string(index=False))

    report.to_csv(
        DATA_DIR / "quality_report.csv",
        index=False
    )