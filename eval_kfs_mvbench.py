import numpy as np
import pandas as pd
import glob
from pathlib import Path
from fractions import Fraction
import subprocess
from tqdm.auto import tqdm 
import uuid
from PIL import Image
from fractions import Fraction
from google import genai
from google.genai import types
import os
import argparse
import multiprocessing
import httpx
import backoff

N_FFMPEG_THREADS=1
N_PARALLEL=6

# Repurposing some eval code from https://github.com/OpenGVLab/Ask-Anything/blob/main/video_chat2/mvbench.ipynb
data_list = {
    "Action Sequence": ("action_sequence.json", "data/MVBench/video/star/Charades_v1_480/", "video", True), # has start & end
    "Action Prediction": ("action_prediction.json", "data/MVBench/video/star/Charades_v1_480/", "video", True), # has start & end
    "Action Antonym": ("action_antonym.json", "data/MVBench/video/ssv2_video/", "video", False),
    "Fine-grained Action": ("fine_grained_action.json", "data/MVBench/video/Moments_in_Time_Raw/videos/", "video", False),
    "Unexpected Action": ("unexpected_action.json", "data/MVBench/video/FunQA_test/test/", "video", False),
    "Object Existence": ("object_existence.json", "data/MVBench/video/clevrer/video_validation/", "video", False),
    "Object Interaction": ("object_interaction.json", "data/MVBench/video/star/Charades_v1_480/", "video", True), # has start & end
    "Object Shuffle": ("object_shuffle.json", "data/MVBench/video/perception/videos/", "video", False),
    "Moving Direction": ("moving_direction.json", "data/MVBench/video/clevrer/video_validation/", "video", False),
    "Action Localization": ("action_localization.json", "data/MVBench/video/sta/sta_video/", "video", True),  # has start & end
    "Scene Transition": ("scene_transition.json", "data/MVBench/video/scene_qa/video/", "video", False),
    "Action Count": ("action_count.json", "data/MVBench/video/perception/videos/", "video", False),
    "Moving Count": ("moving_count.json", "data/MVBench/video/clevrer/video_validation/", "video", False),
    "Moving Attribute": ("moving_attribute.json", "data/MVBench/video/clevrer/video_validation/", "video", False),
    "State Change": ("state_change.json", "data/MVBench/video/perception/videos/", "video", False),
    # "Fine-grained Pose": ("fine_grained_pose.json", "data/MVBench/video/nturgbd/", "video", False), # TODO: obtain later?
    "Character Order": ("character_order.json", "data/MVBench/video/perception/videos/", "video", False),
    "Egocentric Navigation": ("egocentric_navigation.json", "data/MVBench/video/vlnqa/", "video", False),
    "Episodic Reasoning": ("episodic_reasoning.json", "data/MVBench/video/tvqa/frames_fps3_hq/", "frame", False),  # this was originally true (uses timestamps),
    # but some of the timestamps are so unbelievably wrong that it's probably not worth using them and just ingesting the whole video
    "Counterfactual Inference": ("counterfactual_inference.json", "data/MVBench/video/clevrer/video_validation/", "video", False),
}

data_dir = "data/MVBench/json"

def get_frames(path: Path, frames: list[int]):
    prefix = str(uuid.uuid4())
    if path.is_dir():
        for f in frames:
            img = Image.open(path / f"{f+1:05d}.jpg")
            img.thumbnail((768, 768), Image.Resampling.LANCZOS) # Resize for optimal token usage w/ gemini
            img.save(f"ffmpeg/{prefix}_frame{f:04d}.jpg")
    else:
        frame_select_str = "+".join([f"eq(n\\,{x})" for x in frames])
        cmd = [
            "ffmpeg",
            "-nostdin",
            "-i", str(path),
            "-v", "error",
            "-vf", f"select={frame_select_str},scale=768:768:force_original_aspect_ratio=decrease", # Resize for optimal token usage w/ gemini
            "-vsync", "0",
            "-q:v", "2",
            f"ffmpeg/{prefix}_frame%04d.jpg"
        ]
        subprocess.check_call(cmd)
    paths = glob.glob(f"ffmpeg/{prefix}*.jpg")
    return sorted(paths)

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
    video_fr: Fraction,
    keyint_min: int = 8,
    keyint_max: int = 24,
    sc_threshold: int = 40,
    bframes: int = -1,  # -1 uses libx264 default of 3
    trim_start=None,
    trim_end=None
) -> tuple[list[int], int, int]:
    cmd = None
    p = Path(path)

    start_offset_frames = 0
    trim_args = []

    if trim_start is not None:
        start_offset_frames = int(round(trim_start * video_fr))
        trim_args.extend(["-ss", str(trim_start)])

    if trim_end is not None:
        trim_args.extend(["-to", str(trim_end)])
    
    
    if p.is_dir():
        cmd = [
        "ffmpeg", 
        "-nostdin",
        "-threads", str(N_FFMPEG_THREADS), # For parallel use
        *trim_args, 
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
        "-nostdin",
        "-v", "error",
        *trim_args, 
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
        items = line.split(",")
        frame_count += 1
        if "I" in items and "1" in items:
            idr_indices.append(i + start_offset_frames)
            idr_count += 1
    return idr_indices, idr_count, frame_count



def get_uniform_frames_trim(path: Path, total_frames: int, frames_per_sample: int, video_fr: Fraction, start = None, end = None) -> list[Path]:
    start_frame = int(round(float(Fraction(start) * video_fr))) if start is not None else 0
    end_frame = int(np.floor(float(Fraction(end) * video_fr))) if end is not None else total_frames
    frame_idx = [i for i in range(start_frame, end_frame + 1, frames_per_sample)]
    return get_frames(path, frame_idx), frame_idx

def get_ffmpeg_keyframe_indices_for_target_frame_rate(
    path: str,
    video_fr: Fraction,
    target_rate: Fraction,
    compression_ratio: Fraction,
    sc_threshold: int = 40,
    bframes: int = -1,
    start = None,
    end = None,
):
    # Use ceiling so we don't oversample
    min_scene_len = np.ceil(float(video_fr / target_rate))
    max_scene_len = np.ceil(float(video_fr / target_rate * compression_ratio))

    keyframes, _, _ = get_keyframes_ffmpeg(
        path,
        video_fr,
        keyint_min=min_scene_len,
        keyint_max=max_scene_len,
        sc_threshold=sc_threshold,
        bframes=bframes,
        trim_start=start,
        trim_end=end
    )
    return keyframes
  
def get_h264_frames_trim(path: str, video_fr: Fraction, target_rate: Fraction, compression_ratio: Fraction, scenecut: int, bframes: int =0, start=None, end=None ):
  frame_idx = get_ffmpeg_keyframe_indices_for_target_frame_rate(path, video_fr, target_rate, compression_ratio, sc_threshold=scenecut, bframes=bframes, start=start, end=end)
  return get_frames(path, frame_idx), frame_idx
  
def qa_template(question, candidates, answer):
    question = f"Question: {question}\n"
    question += "Options:\n"
    answer_idx = -1
    for idx, c in enumerate(candidates):
        question += f"({chr(ord('A') + idx)}) {c}\n"
        if c == answer:
            answer_idx = idx
    question = question.rstrip()
    answer = chr(ord('A') + answer_idx)
    return question, answer

@backoff.on_exception(backoff.expo, httpx.HTTPError)
def prompt_gemini_mvbench(keyframes, question, candidates, answer):
    question, answer = qa_template(question, candidates, answer)
    system_prompt = "Carefully watch the video and pay attention to the cause and sequence of events, the detail and movement of objects, \
    and the action and pose of persons. Based on your observations, select the best option that accurately addresses the question.\n"
    question_prompt="\nOnly give the best option."
    full_prompt = question + question_prompt

    client = genai.Client()

    file_sizes = [os.path.getsize(kf) for kf in keyframes]

    inline = False
    # Inline files if sub 15 MB
    if sum(file_sizes) < 15_000_000:
        inline = True
        remote_kfs = [Image.open(kf) for kf in keyframes]
    else:
        remote_kfs = [client.files.upload(file=kf) for kf in keyframes]
    
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=[full_prompt, *remote_kfs],
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
        )
    )

    if not inline:
        for kf in remote_kfs:
            client.files.delete(name=kf.name)

    token_count = list(filter(lambda x: x.modality == types.MediaModality.IMAGE, response.usage_metadata.prompt_tokens_details))[0].token_count

    text_list = response.text.split("(")
    correct = False if len(text_list) < 2 or text_list[1][0] != answer else True
    return response.text, token_count, correct

# 1 FPS, compression ratio = 2, 
def run_mvbench_category(mv_df: pd.DataFrame, category: str, video_dir: Path, trimmed: bool, scenecut: int):
    results = []
    total_correct = 0
    # See if we can use pre-cached keyframes
    use_cache = False
    cache_df = None
    if not trimmed:
      json_path = Path(f"kf_info/{video_dir.parts[3]}/kf_fps=1_ratio=2_sc={scenecut}_bframes=0.json")
      if json_path.is_file():
        use_cache = True
        cache_df = pd.read_json(json_path, orient="records")
        # Make sure we don't miss out on data in the data0613 folder
        if video_dir.parts[3] in ["clevrer", "star"]:
          json_path = Path(f"kf_info/data0613/kf_fps=1_ratio=2_sc={scenecut}_bframes=0.json")
          cache_df = pd.concat([cache_df, pd.read_json(json_path)], ignore_index=True)
    all_args = [[i, video_dir, row, use_cache, cache_df, scenecut, trimmed, category] for i, row in mv_df.iterrows()]
    with multiprocessing.Pool(N_PARALLEL) as p:
      results = list(tqdm(p.imap(run_mvbench_one, all_args), total=mv_df.shape[0]))
      total_correct = sum(map(lambda x: x["correct"] if x is not None else 0, results))
      return results, total_correct / len(mv_df)

def run_mvbench_one(args):
  i, video_dir, row, use_cache, cache_df, scenecut, trimmed, category = args
  video_path = Path(video_dir) / Path(row.video)
  if not video_path.exists():
      supp_glob = str(Path("data/MVBench/video/data0613/*/**/") / Path(row.video))
      print("No match found for file", video_path,  "checking supplementary folder...:")
      extra_data = glob.glob(supp_glob)
      if len(extra_data) == 0:
          print("File not found at all! That's not great")
          return None
      else:
          print("Substitute file found:", extra_data[0])
          video_path = Path(extra_data[0])
  video_fr = get_frame_rate(video_path)
  if use_cache:
    hits =  cache_df[cache_df.path == str(video_path)]
    if len(hits) == 0:
      print("ERROR: CACHE MISS ON FILE", str(video_path))
      frame_idxs = get_ffmpeg_keyframe_indices_for_target_frame_rate(
      video_path,
      video_fr,
        Fraction(1),
        Fraction(2),
        scenecut,
        0, # No bframes allowed!
        start = row.start if trimmed else None,
        end = row.end if trimmed else None,
      )
    else:
      frame_idxs = hits.iloc[0].keyframes
  else:
    frame_idxs = get_ffmpeg_keyframe_indices_for_target_frame_rate(
    video_path,
    video_fr,
      Fraction(1),
      Fraction(2),
      scenecut,
      0, # No bframes allowed!
      start = row.start if trimmed else None,
      end = row.end if trimmed else None,
    )
  
  video_frames = get_frames(video_path, frame_idxs)
    
  response, token_count, correct = prompt_gemini_mvbench(video_frames, row.question, row.candidates, row.answer)
  result = {
      "video": video_path,
      "category": category,
      "qid": i,
      "question": row.question,
      "candidates": row.candidates,
      "answer_gt": row.answer,
      "answer_pred": response,
      "correct": correct,
      "token_count": token_count,
      "frame_count": len(frame_idxs)
  }
  for f in video_frames:
      os.remove(f)
  return result

def run_mvbench_h264(scenecut: int):
    all_results = []
    all_avgs = []
    for i, category in enumerate(data_list):
        json, video_dir, _, trimmed = data_list[category]
        print(f"({i + 1:02d}/19) Starting", category, )
        
        df = pd.read_json(Path(data_dir) / Path(json), orient="records")
        results, correct_avg = run_mvbench_category(df, category, Path(video_dir), trimmed, scenecut)
        all_results.append(results)
        all_avgs.append(correct_avg)
        print("Finished", category)
        print("Average correct:", correct_avg)
    print("Done")
    return all_results, all_avgs

if __name__ == "__main__":
  parser = argparse.ArgumentParser("eval_kfs_mvbench")
  parser.add_argument("--scenecut", type=int, default=80)

  args = parser.parse_args()

  results, avgs = run_mvbench_h264(args.scenecut)
  pd.DataFrame(results).to_json(f"mvbench_sc={args.scenecut}.json", orient="records", default_handler=str)