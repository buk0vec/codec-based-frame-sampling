"""
kf_count.py

Inital experimentation to see how varying scenecut impacts the number of encoded IDR frames. 
"""

import subprocess
import os
from tqdm.auto import tqdm
import glob
from pathlib import Path
import numpy as np
from fractions import Fraction
import csv
import math

def get_total_frames(path: Path) -> int:
    if path.is_dir():
        return len([f for f in path.iterdir() if f.is_file()])
    else:
        ffprobe_cmd = [
            "ffprobe",
            "-v",
            "error",
            "-count_frames",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=nb_read_frames",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ]
        try:
            output = subprocess.check_output(ffprobe_cmd).decode("utf-8")
            return int(output.strip())
        except Exception:
            print("ffprobe failed for file", path)
            return 0


def get_frame_rate(path: Path) -> Fraction:
    if path.is_dir():
        return Fraction(3)
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=r_frame_rate",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]

    try:
        output = subprocess.check_output(cmd).decode("utf-8").strip()
        return Fraction(output)
    except Exception as e:
        print(f"Error getting frame rate: {e}")
        return Fraction(0)


def get_keyframes_ffmpeg(
    path: str,
    keyint_min: int = 8,
    keyint_max: int = 24,
    sc_threshold: int = 40,
    bframes: int = 0,  # -1 is default, libx264 defaults to 3
) -> tuple[list[int], int, int]:
    tmp_file = "ffmpeg/out.mp4"
    cmd = None
    p = Path(path)
    if p.is_dir():
        cmd = [
            "ffmpeg",
            "-framerate",
            "3",  # tvqa videos all are at 3 fps
            "-i",
            p / "%05d.jpg",
            "-v", "error",
            "-c:v",
             "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", # Make sure h/w is even
            "libx264",
            "-g",
            str(keyint_max),
            "-keyint_min",
            str(keyint_min),
            "-sc_threshold",
            str(sc_threshold),
            "-flags",
            "+cgop",  # Use closed GOP to force frames to reference current
            "-bf",
            str(bframes),
            tmp_file,
        ]
    else:
        cmd = [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(path),
             "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", # Make sure h/w is even
            "-c:v",
            "libx264",
            "-g",
            str(keyint_max),
            "-keyint_min",
            str(keyint_min),
            "-sc_threshold",
            str(sc_threshold),
            "-flags",
            "+cgop",  # Use closed GOP to force frames to reference current
            "-bf",
            str(bframes),
            tmp_file,
        ]
    subprocess.check_call(cmd)
    ffprobe_cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "frame=pict_type,key_frame",
        "-of",
        "csv=p=0",
        tmp_file,
    ]

    output = subprocess.check_output(ffprobe_cmd).decode("utf-8")

    lines = output.strip().split("\n")
    idr_indices = []
    frame_count = 0
    idr_count = 0
    for i, line in enumerate(lines):
        line = line.strip(",")  # ffmpeg does trailing commas sometimes
        if not line:
            continue
        key_flag, ptype = line.split(",")
        frame_count += 1
        if ptype == "I" and key_flag == "1":
            idr_indices.append(i)
            idr_count += 1
    os.remove(tmp_file)
    return idr_indices, idr_count, frame_count


# Returns all the full paths to the MVBench videos. This includes paths to the tvqa directories which contain single frames
def get_all_video_locations() -> list[Path]:
    video_extensions = ["mp4", "webm"]
    globs = [
        os.path.join("data/MVBench/video", "**", f"*.{ext}") for ext in video_extensions
    ]

    video_files = [Path(f) for g in globs for f in glob.glob(g, recursive=True)]

    # get tvqa dirs
    tvqa_dir = Path("data/MVBench/video/tvqa/frames_fps3_hq")
    frame_dirs = [f for f in tvqa_dir.iterdir() if f.is_dir()]

    return video_files + frame_dirs


def get_ffmpeg_keyframe_indices_for_target_frame_rate(
    path: str,
    video_fr: Fraction,
    target_rate: Fraction,
    compression_ratio: Fraction,
    sc_threshold: int = 40,
    bframes: int = 0,
):
    # Use ceiling so we don't oversample
    min_scene_len = math.ceil(float(video_fr / target_rate))
    max_scene_len = math.ceil(float(video_fr / target_rate * compression_ratio))

    keyframes = get_keyframes_ffmpeg(
        path,
        keyint_min=min_scene_len,
        keyint_max=max_scene_len,
        sc_threshold=sc_threshold,
        bframes=bframes,
    )

    return keyframes


def eval_keyframe_count_by_dir(
    files: list[Path],
    target_rate: Fraction,
    compression_ratio: Fraction,
    sc_threshold: int,
    bframes: int,
    write_stats: bool = True,
    output_file: str = "kf_stats.csv",
):
    """
    Given a config and a list of different scenecut thresholds, go through each folder and get the number of keyframes
    encoded by libx264.
    """
    folders_dict = {}
    for file in files:
        if file.parts[3] not in folders_dict:
            folders_dict[file.parts[3]] = []
        folders_dict[file.parts[3]].append(file)

    with open(output_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["path", "frame_count", "uniform_frames", "key_frames"])
        for folder in folders_dict:
            print(f"===Processing folder {folder}===")
            i = 0.0
            total_frames_saved = 0
            total_keyframes = 0
            total_uniform_frames = 0
            total_ratio = 0.0
            avg_fps = 0
            pbar = tqdm(folders_dict[folder])
            for f in pbar:
                video_fr = get_frame_rate(f)
                keyframes, kf_count, frame_count = (
                    get_ffmpeg_keyframe_indices_for_target_frame_rate(
                        f,
                        video_fr,
                        target_rate,
                        compression_ratio,
                        sc_threshold,
                        bframes,
                    )
                )
                total_keyframes += kf_count
                # Calculate the number of frames that would be used if we used uniform sampling
                uniform_sampling = math.floor(frame_count / video_fr * target_rate)
                total_uniform_frames += uniform_sampling
                frames_saved = uniform_sampling - kf_count
                total_frames_saved += frames_saved
                total_ratio += kf_count / uniform_sampling
                writer.writerow([str(f), frame_count, uniform_sampling, kf_count])
                i += 1.0
                avg_reduction = total_frames_saved / i
                avg_ratio = total_ratio / i
                avg_fps += (float(video_fr) - avg_fps)/ i
                pbar.set_description(
                    f"Tot kf: {total_keyframes}, tot unif: {total_uniform_frames}, avg frm reduction: {avg_reduction:.2f}, avg rate: {avg_ratio:.2f}, avg input fps: {avg_fps}"
                )
            print("Finished processing folder")
            print(
                f"Total keyframes: {total_keyframes}, total uniformly sampled frames: {total_uniform_frames}, avg. frame reduction: {total_frames_saved / i}, avg. frame compression ratio: {total_ratio / i}, avg input fps: {avg_fps}"
            )

if __name__ == "__main__":
    files = get_all_video_locations()
    
    print("Target rate 1 FPS, compression ratio 2, sc_threshold 100, no bframes")
    
    eval_keyframe_count_by_dir(
        files,
        target_rate=Fraction(1),
        compression_ratio=Fraction(2),
        sc_threshold=100,
        bframes=0,
        write_stats=True,
        output_file="kf_stats_threshold=100_nobframes.csv",
    )