"""
eval_mvbench_frs.py

Evaluates MVBench using frame reduction sampling. Can also specify the option to use uniform sampling instead.
"""

import argparse
from pathlib import Path
import json
import pandas as pd
import multiprocessing
from keyframes import FrameReductionSampler
from tqdm.auto import tqdm
import os
from mvbench import find_video, run_gemini_mvbench
from utils import get_frames

N_PARALLEL = 7

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

def run_mvbench_one(args):
  i, video_dir, row, trimmed, category, fps, uniform, compression_factor, scenecut = args
  video_path = find_video(Path(video_dir), Path(row.video))
  if video_path is None:
    raise Exception("Missing video")
  sampler = FrameReductionSampler(fps, scenecut, compression_factor)
  start = row.start if trimmed else None
  end = row.end if trimmed else None
  if uniform:
    keyframes = sampler.uniform(video_path, start=start, end=end)
  else:
    keyframes = sampler.solve(video_path, start=start, end=end)
    
  video_frames = get_frames(video_path, keyframes)
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
      "frame_count": len(keyframes)
  }
  for f in video_frames:
      os.remove(f)
  return result

def run_mvbench_category(
  mv_df: pd.DataFrame,
  category: str, 
  video_dir: Path, 
  trimmed: bool, 
  fps: int, 
  uniform: bool,
  compression_factor: float,
  scenecut: int
  ):
    results = []
    total_correct = 0
    all_args = [[
      i, 
      video_dir, 
      row, 
      trimmed, 
      category, 
      fps, 
      uniform,
      compression_factor,
      scenecut
      ] for i, row in mv_df.iterrows()]
    with multiprocessing.Pool(N_PARALLEL) as p:
      results = list(tqdm(p.imap(run_mvbench_one, all_args), total=mv_df.shape[0]))
      total_correct = sum(map(lambda x: x["correct"] if x is not None else 0, results))
      return results, total_correct / len(mv_df)

def run_mvbench_target(fps, uniform, compression_factor, scenecut):
    all_results = []
    all_avgs = []
    for i, category in enumerate(data_list):
        json, video_dir, _, trimmed = data_list[category]
        print(f"({i + 1:02d}/19) Starting", category)
        df = pd.read_json(Path(data_dir) / Path(json), orient="records")
        results, correct_avg = run_mvbench_category(df, category, Path(video_dir), trimmed, fps, uniform, compression_factor, scenecut)
        all_results += results
        all_avgs.append(correct_avg)
        print("Finished", category)
        print("Average correct:", correct_avg)
    print("Done")
    return all_results, all_avgs

if __name__ == "__main__":
  parser = argparse.ArgumentParser("eval_mvbench_frs")
  parser.add_argument("-f", "--fps", type=int)
  parser.add_argument("-c", "--compression_factor", type=float)
  parser.add_argument("-u", "--uniform", action="store_true")
  parser.add_argument("-s", "--scenecut", type=int, default=80)
  parser.add_argument("-o", "--output")
  
  args = parser.parse_args()
  
  if args.fps is None:
    print("ERROR: must specify sampling rate")
    exit(1)
  if args.uniform and args.compression_factor is not None:
      print("ERROR: compression factor passed with uniform sampling")
      exit(1)
  if not args.uniform and args.compression_factor is None:
    print("ERROR: must provide compression factor")
    exit(1)
  output_file = f"mvbench_sr={args.fps}{'_uniform' if args.uniform else f'_scenecut={args.scenecut}_alpha={args.compression_factor}'}.json"
  if args.output is not None:
    output_file = args.output
  
  results, avgs = run_mvbench_target(args.fps, args.uniform, args.compression_factor, args.scenecut)
  pd.DataFrame(results).to_json(output_file, orient="records", default_handler=str)
  
  
  