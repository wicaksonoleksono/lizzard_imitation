# 1) Train the model (assumes you have some videos to label)
python interface/main.py --mode train --config path/to/config.yaml --video path/to/video1.mp4 path/to/video2.mp4

# 2) Inference on new videos
python interface/main.py --mode infer --config path/to/config.yaml --video path/to/new_video.mp4
