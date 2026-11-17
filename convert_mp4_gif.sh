#!/bin/bash

INPUT_DIR="data/visualizations"

for video in "$INPUT_DIR"/*.mp4; do
    # Skip if no MP4 files exist
    [ -e "$video" ] || continue

    # Replace .mp4 extension with .gif
    output="${video%.mp4}.gif"

    echo "Converting: $video"

    ffmpeg -y -i "$video" \
        -vf "fps=12,scale=480:-1:flags=lanczos" \
        -loop 0 \
        "$output"

    echo "Saved: $output"
done

echo "All videos converted!"