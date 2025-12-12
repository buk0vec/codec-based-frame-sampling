"""
generate_kfs_parallel.py

Precomputes FRS keyframes in parallel for MVBench videos. Please don't use this, it does not take trimmed clips into account.
So every trimmed clip will need to have its keyframes recalculated at eval time!
"""

import subprocess
import os
from tqdm.auto import tqdm
import glob
from pathlib import Path
import numpy as np
import math
from fractions import Fraction
import csv
import argparse
import json
import multiprocessing
import time

N_PARALLEL = 7
N_FFMPEG_THREADS = 1

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
    path: Path | str,
    keyint_min: int = 8,
    keyint_max: int = 24,
    sc_threshold: int = 40,
    bframes: int =  0,  # -1 uses default, libx264 defaults to 3
) -> tuple[list[int], int, int]:
    cmd = None
    p = Path(path)
    if p.is_dir():
        cmd = [
        "ffmpeg", 
        "-threads", str(N_FFMPEG_THREADS), # For parallel use
        "-framerate", "3", # tvqa videos all are at 3 fps
        "-i", str(p / "%05d.jpg"), 
        "-c:v", "libx264", 
        "-preset", "superfast",
        "-v", "error",
        "-vf", "scale=-2:360", # Make sure h/w is even
        "-x264-params", f"rc-lookahead={min(keyint_max, 60)}", # Full lookahead, capped for now just to get a full run in
        "-g", str(keyint_max), 
        "-keyint_min", str(keyint_min), 
        "-sc_threshold", str(sc_threshold), 
        "-flags", "+cgop", # Use closed GOP to force frames to reference current 
        "-bf", str(bframes),
        "-f", "h264",
        "pipe:1",
    ]
    else:
        cmd = [
        "ffmpeg", 
        "-v", "error",
        "-i", str(path), 
        "-c:v", "libx264", 
        "-preset", "superfast",
        "-threads", str(N_FFMPEG_THREADS), # For parallel use
        "-vf", "scale=-2:360", # Make sure h/w is even
        "-g", str(keyint_max), 
        "-keyint_min", str(keyint_min), 
        "-x264-params", f"rc-lookahead={min(keyint_max, 60)}", # Full lookahead, capped for now just to get a full run in
        "-sc_threshold", str(sc_threshold), 
        "-flags", "+cgop", # Use closed GOP to force frames to reference current 
        "-bf", str(bframes),
        "-f", "h264",
        "pipe:1",
    ]

    ffmpeg_process = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    ffprobe_cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "frame=pict_type,key_frame",
        "-of", "csv=p=0", 
        "-"
    ]
    
    output = subprocess.check_output(ffprobe_cmd, stdin=ffmpeg_process.stdout)
    ffmpeg_process.wait()

    lines = output.decode("utf-8").strip().split("\n")
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

    return frame_dirs + video_files


def get_ffmpeg_keyframe_indices_for_target_frame_rate(
    path: Path | str,
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
    target_rate: Fraction = Fraction(1),
    compression_ratio: Fraction = Fraction(2),
    sc_threshold: int = 40,
    bframes: int = 0,
    file_suffix: str = "",
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


    stats_file = Path(f"kf_stats/kf_stats_{file_suffix}.csv")
    batch_start = time.time()

    with open(stats_file, mode="w", newline="") as stats:
        stats_writer = csv.writer(stats)
        stats_writer.writerow(["path", "frame_count", "uniform_frames", "key_frames"])
        for folder in folders_dict:
            start = time.time()
            keyframes_file = Path(f"kf_info/{folder}/kf_{file_suffix}.json")
            with open(keyframes_file, mode="w", newline="") as kf_json:
                kf_json.write('[\n')
                print(f"===Folder: {folder}, scenecut: {sc_threshold}===")
                i = 0.0
                total_frames_saved = 0
                total_keyframes = 0
                total_uniform_frames = 0
                total_ratio = 0.0
                avg_fps = 0
                configs = [(f,
                        target_rate,
                        compression_ratio,
                        sc_threshold,
                        bframes) for f in folders_dict[folder]]
                
                with multiprocessing.Pool(processes=N_PARALLEL) as pool:
                    results = list(tqdm(pool.imap(eval_mp, configs), total=len(configs)))
                
                    for i, (f, keyframes, kf_count, frame_count, uniform_sampling, frames_saved, ratio, video_fr) in enumerate(results):
                        if i > 0:
                            kf_json.write(',\n')
                        json.dump({
                            "path": str(f),
                            "keyframes": keyframes    
                        }, kf_json, indent=4)
                        total_keyframes += kf_count
                        total_uniform_frames += uniform_sampling
                        total_frames_saved += frames_saved
                        total_ratio += ratio
                        avg_fps += video_fr
                        stats_writer.writerow([str(f), frame_count, uniform_sampling, kf_count])
                        
                    end = time.time()
                    print(f"Finished processing folder in {end - start} secs, {(end-start) / i} secs/video")
                    print(
                        f"SC: {sc_threshold}, total videos: {i}, total keyframes: {total_keyframes}, total uniformly sampled frames: {total_uniform_frames}, avg. frame reduction: {total_frames_saved / i}, avg. frame compression ratio: {total_ratio / i}, avg input fps: {avg_fps}"
                    )
                    kf_json.write('\n]')
    batch_end = time.time()
    print(f"Batch complete in {batch_end-batch_start} secs, {(batch_end - batch_start) / len(files)} secs/video")

def eval_keyframe_count(
    file: Path,
    target_rate: Fraction = Fraction(1),
    compression_ratio: Fraction = Fraction(2),
    sc_threshold: int = 40,
    bframes: int = 0,
):
    video_fr = get_frame_rate(file)
    keyframes, kf_count, frame_count = get_ffmpeg_keyframe_indices_for_target_frame_rate(
        file,
        video_fr,
        target_rate,
        compression_ratio,
        sc_threshold,
        bframes,
    )
    uniform_sampling = math.floor(frame_count / video_fr * target_rate)
    frames_saved = uniform_sampling - kf_count
    ratio = kf_count / uniform_sampling
    return file, keyframes, kf_count, frame_count, uniform_sampling, frames_saved, ratio, video_fr
    

def eval_mp(args):
    return eval_keyframe_count(*args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("generate_kfs")
    parser.add_argument("--target_fps", type=int, default=1)
    parser.add_argument("--compression_ratio", type=int, default=2)
    parser.add_argument("--sc_min", type=int, default=0)
    parser.add_argument("--sc_max", type=int, default=200)
    parser.add_argument("--sc_step", type=int, default=20)
    parser.add_argument("--bframes", type=int, default=0)

    args = parser.parse_args()

    files = get_all_video_locations()

    print(
        f"Target FPS {args.target_fps}, \
          compression ratio {args.compression_ratio}, \
          sc range {args.sc_min} {args.sc_max} {args.sc_step}, \
        bframes {args.bframes}"
    )

    scs = [sc for sc in range(args.sc_min, args.sc_max + args.sc_step, args.sc_step)]
    
    for sc in tqdm(scs):
        argstr = f"fps={args.target_fps}_ratio={args.compression_ratio}_sc={sc}_bframes={args.bframes}"
        eval_keyframe_count_by_dir(
            files,
            target_rate=Fraction(args.target_fps),
            compression_ratio=Fraction(args.compression_ratio),
            sc_threshold=sc,
            bframes=args.bframes,
            file_suffix=argstr
        )
