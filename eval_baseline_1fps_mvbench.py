"""
eval_baseline_1fps_mvbench.py

Runs a 1 FPS uniform sampling baseline on MVBench and saves the results.
"""

import pandas as pd
from pathlib import Path
from fractions import Fraction
from sampling import get_uniform_frames_trim
from tqdm.auto import tqdm 
from fractions import Fraction
import os
import multiprocessing
from mvbench import find_video, run_gemini_mvbench
from utils import get_frame_rate, get_total_frames
import math

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

data_dir = "../data/MVBench/json"

# 1 FPS baseline, uses some pre-calc'ed frame counts from earlier
def process_single_example_baseline(args):
    video_dir, row, i, trimmed, category = args
    video_path = find_video(Path(video_dir), Path(row.video))
    if video_path is None:
        raise Exception("Could not find video")
    TARGET_RATE = Fraction(1)
    video_fr = get_frame_rate(video_path)
    video_fc = get_total_frames(video_path)
    min_scene_len = int(math.ceil(float(video_fr / TARGET_RATE)))
    if trimmed:
        video_frames, frame_idxs = get_uniform_frames_trim(video_path, video_fc, min_scene_len, Fraction(video_fr), start=row.start, end=row.end)
    else:
        video_frames, frame_idxs = get_uniform_frames_trim(video_path, video_fc, min_scene_len, Fraction(video_fr))
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

def run_mvbench_category_baseline(mv_df: pd.DataFrame, category: str, video_dir: Path, trimmed: bool):
    results = []
    total_correct = 0
    all_args = [(video_dir, row, i, trimmed, category) for i, row in mv_df.iterrows()]
    with multiprocessing.Pool(N_PARALLEL) as p:
        results = list(tqdm(p.imap(process_single_example_baseline, all_args), total=mv_df.shape[0]))
        total_correct = sum(map(lambda x: x["correct"] if x is not None else 0, results))
        return results, total_correct / len(mv_df)

def run_mvbench_baseline():
    all_results = []
    all_avgs = []
    for i, category in enumerate(data_list):
        json, video_dir, _, trimmed = data_list[category]
        print(f"({i + 1:02d}/19) Starting", category, )
        df = pd.read_json(Path(data_dir) / Path(json), orient="records", dtype=False)
        results, correct_avg = run_mvbench_category_baseline(df, category, Path(video_dir), trimmed)
        all_results += results
        all_avgs.append(correct_avg)
        print("Finished", category)
        print("Average correct:", correct_avg)
    print("Done")
    return all_results, all_avgs
  
if __name__ == "__main__":
  all_results, all_avgs = run_mvbench_baseline()
  pd.DataFrame(all_results).to_json("../evals/mvbench_baseline.json", orient="records", default_handler=str)
  