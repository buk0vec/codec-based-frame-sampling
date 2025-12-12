from fractions import Fraction
from pathlib import Path
import pandas as pd
from keyframes import get_ffmpeg_keyframe_indices_for_target_frame_rate
from utils import get_frames, get_duration
import numpy as np
import math

# Get the keyframes from the initial scenecut experiment for clips which do not need to be trimmed.
# Avoids re-running ffmpeg when evaluating. Only valid when using 1 FPS, compression ratio of 2,
# no bframes, as those were settings used in that expirement.
def get_cached_sampled_frames(path: Path, scenecut: int):
    json_path = Path(f"kf_info/{path.parts[3]}/kf_fps=1_ratio=2_sc={scenecut}_bframes=0.json")
    json = pd.read_json(json_path, orient="records")
    frame_idx = json[json.path == str(path)].iloc[0].keyframes
    return get_frames(path, frame_idx), frame_idx
  
# Get uniformly sampled frames from a video, given a target FPS and the video frame rate
def get_uniform_frames_trim(path: Path, total_frames: int, frames_per_sample: int, video_fr: Fraction, start = None, end = None) -> tuple[list[str], list[int]]:
    if start is not None or end is not None:
        video_duration = get_duration(path)
        if end is not None and end > video_duration:
            print(f"WARN: end given after video end", path)
            end = None
        if start is not None and start > video_duration:
            print("WARN: start given after video start", path)
            start = None
    
    start_frame = int(round(float(Fraction(start) * video_fr))) if start is not None else 0
    end_frame = int(math.floor(float(Fraction(end) * video_fr))) if end is not None else total_frames

    frame_idx = [i for i in range(start_frame, end_frame, frames_per_sample)]

    # For tvqa, ensure we don't exceed available frames
    if path.is_dir():
        frame_idx = [i for i in frame_idx if i < total_frames]

    return get_frames(path, frame_idx), frame_idx
    

# Get h264 keyframes from a video, given a target fps, video fps, compression ratio, and x264 params
def get_h264_frames_trim(path: Path, video_fr: Fraction, target_rate: Fraction, compression_ratio: Fraction, scenecut: int, bframes: int =0, start=None, end=None ):
  if start is not None or end is not None:
    video_duration = get_duration(path)
    if end is not None and end > video_duration:
        print("WARN: end given after video end", path)
        end = None
    # Some tvqa start times are incorrect, not sure what to do about that other than to not clip them
    if start is not None and start > video_duration:
        print("WARN: start given after video start", path)
        start = None
  frame_idx = get_ffmpeg_keyframe_indices_for_target_frame_rate(path, video_fr, target_rate, compression_ratio, sc_threshold=scenecut, bframes=bframes, start=start, end=end)
  return get_frames(path, frame_idx), frame_idx

