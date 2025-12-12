"""
eval_frs_mvbench_with_cache.py

Evaluates MVBench using FRS. Tries to use precomputed keyframes from generate_kfs_parallel.
"""

from keyframes import get_ffmpeg_keyframe_indices_for_target_frame_rate
from mvbench import find_video, run_gemini_mvbench
import pandas as pd
from pathlib import Path
from fractions import Fraction
from tqdm.auto import tqdm 
from fractions import Fraction
import os
import argparse
import multiprocessing
from utils import get_frame_rate, get_frames

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
    # Some of the timestamps for episodic reasoning are invalid, have extra checks for this
    "Episodic Reasoning": ("episodic_reasoning.json", "data/MVBench/video/tvqa/frames_fps3_hq/", "frame", True),  
    "Counterfactual Inference": ("counterfactual_inference.json", "data/MVBench/video/clevrer/video_validation/", "video", False),
}

data_dir = "data/MVBench/json"

# 1 FPS, compression ratio = 2, 
def run_mvbench_category(mv_df: pd.DataFrame, category: str, video_dir: Path, trimmed: bool, scenecut: int):
    results = []
    total_correct = 0
    # See if we can use pre-cached keyframes from initial experiment
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
  video_path = find_video(Path(video_dir), Path(row.video))
  if video_path == None:
    raise Exception("Missing video path")
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
  answer = str(row.answer)
  candidates = [str(c) for c in row.candidates]
  response, token_count, correct = run_gemini_mvbench(video_frames, row.question, candidates, answer)
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
        all_results += results
        all_avgs.append(correct_avg)
        print("Finished", category)
        print("Average correct:", correct_avg)
    print("Done")
    return all_results, all_avgs

if __name__ == "__main__":
  parser = argparse.ArgumentParser("eval_frs_mvbench_with_cache")
  parser.add_argument("--scenecut", type=int, default=80)

  args = parser.parse_args()

  results, avgs = run_mvbench_h264(args.scenecut)
  pd.DataFrame(results).to_json(f"evals/mvbench_sc={args.scenecut}.json", orient="records", default_handler=str)