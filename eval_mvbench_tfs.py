"""
eval_mvbench_tfs.py

Evaluates MVBench using target frame sampling. Can also specify the option to use uniform sampling instead.
"""

import argparse
from pathlib import Path
import json
import pandas as pd
import multiprocessing
from keyframes import TargetFrameSampler
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
  i, video_dir, row, trimmed, category, keyframes, target, uniform = args
  video_path = find_video(Path(video_dir), Path(row.video))
  if video_path is None:
    raise Exception("Missing video")
  if keyframes is None:
    sampler = TargetFrameSampler(target_frames=target)
    start = row.start if trimmed else None
    end = row.end if trimmed else None
    if uniform:
      keyframes = sampler.uniform(video_path, start=start, end=end)
    else:
      keyframes = sampler.solve(video_path, start=start, end=end)["keyframes"]
    
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

def run_mvbench_category(mv_df: pd.DataFrame, category: str, video_dir: Path, trimmed: bool, category_frames: dict | None, target: int | None, uniform: bool | None):
    results = []
    total_correct = 0
    all_args = [[
      i, 
      video_dir, 
      row, 
      trimmed, 
      category, 
      category_frames[str(i)]["keyframes"] if category_frames is not None else None, 
      target, 
      uniform
      ] for i, row in mv_df.iterrows()]
    with multiprocessing.Pool(N_PARALLEL) as p:
      results = list(tqdm(p.imap(run_mvbench_one, all_args), total=mv_df.shape[0]))
      total_correct = sum(map(lambda x: x["correct"] if x is not None else 0, results))
      return results, total_correct / len(mv_df)

def run_mvbench_target(computed_frames, target, uniform):
    all_results = []
    all_avgs = []
    for i, category in enumerate(data_list):
        json, video_dir, _, trimmed = data_list[category]
        category_frames = computed_frames[category] if computed_frames is not None else None
        print(f"({i + 1:02d}/19) Starting", category)
        df = pd.read_json(Path(data_dir) / Path(json), orient="records")
        results, correct_avg = run_mvbench_category(df, category, Path(video_dir), trimmed, category_frames, target, uniform)
        all_results += results
        all_avgs.append(correct_avg)
        print("Finished", category)
        print("Average correct:", correct_avg)
    print("Done")
    return all_results, all_avgs

if __name__ == "__main__":
  parser = argparse.ArgumentParser("eval_mvbench_tfs")
  parser.add_argument("-n", "--n_frames", type=int)
  parser.add_argument("-u", "--uniform", action="store_true")
  parser.add_argument("-f", "--keyframe_file")
  parser.add_argument("-o", "--output")
  
  args = parser.parse_args()
  
  computed_frames = None
  
  output_file = None
  if args.keyframe_file is not None:
    if args.uniform or args.n_frames is not None:
      print("ERROR: extra sampling args passed while using keyframe file")
      exit(1)
    kf_file = Path(args.keyframe_file)
    if not kf_file.is_file():
      print("ERROR: keyframe file does not exist")
      exit(1)
    with open(kf_file, "r") as f:
      computed_frames = json.load(f)
    output_file = f"mvbench_target_file={kf_file.stem}.json"
  elif args.n_frames is None:
    print("ERROR: must provide target frame count if not passing keyframe file")
    exit(1)
  else:
    output_file = f"mvbench_target_frames={args.n_frames}{'_uniform' if args.uniform else ''}.json"
  if args.output is not None:
    output_file = args.output
  
  results, avgs = run_mvbench_target(computed_frames, args.n_frames, args.uniform)
  pd.DataFrame(results).to_json(output_file, orient="records", default_handler=str)
  
  
  