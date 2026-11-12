from huggingface_hub import HfApi
from huggingface_hub import hf_hub_download
from pathlib import Path
import shutil

api = HfApi()

repo = "cvml-nus/assembly101"

info = api.repo_info(
    repo_id=repo,
    repo_type="dataset",
    files_metadata=True
)

videos = []

for file in info.siblings:
    if file.rfilename.endswith("_rgb.mp4"):
        if file.size is not None:
            videos.append((file.rfilename, file.size))

videos.sort(key=lambda x: x[1])

print("10 smallest RGB videos:\n")

for path, size in videos[:10]:
    print(f"{size / 1024**2:.1f} MB | {path}")


