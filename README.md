# 
Annotated gif hand movements
<img src="/home/mohana/Documents/Roboto_RL/Human-Demonstration/data/visualizations/demo_01_annotated.gif" width="300">


## step 1:
Find shortest videos from [Assembly101](https://huggingface.co/datasets/cvml-nus/assembly101?utm_source=chatgpt.com) dataset,
which contains real human demonstrations of assembling, disassembling objects and including multiview 
videos and hand-pose annotations,
by runnng scripts/smallest_videos.py

## step 2:
Download the video:

```bash
hf download cvml-nus/assembly101 \
>     "recordings/<name of video.mp4>" \
>     --repo-type dataset \
>     --local-dir data/downloads
```
## step 3:
Extract a short segment by FFmpeg, as an example: convert assembly_original.mp4

to 5 seconds clip of demo_02.mp4.


```bash
ffmpeg -ss 10 \
>     -i data/videos/<downloaded mp4> \
>     -t 5 \
>     -vf "scale=640:-2,fps=15" \
>     -c:v libx264 \
>     -crf 28 \
>     -an \
>   data/videos/<demo_id>.mp4
```
## step 4:

Extract hand movements from real videos and measure 
the quality of the recorded demonstrations.

  <img src="figures/Hand%20Tracking%20Pipeline%20Infographic.png" width="58%">

For every video, the scripts would need to reads the frames,
detects one hand, extracts 21 landmarks, each 
represented by x, y, and z coordinates.

* Pretrained MediaPipe Hand Landmarker model should be downloaded.

It is a pretrained machine learning model provided by 
Google MediaPipe to detect and track human hands in 
images and videos.


```bash
mkdir -p models
wget -O models/hand_landmarker.task \
https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
```
## step 5: 

To visualize the accuracy of extracted CSV compare to human manipulation videos, OpenCV has been used.
It draws and connects 21 keypoints per hand, distinguishing left and right hands with different colors, 
and generates annotated MP4 videos for visually inspecting hand movements, tracking accuracy, and detection errors.