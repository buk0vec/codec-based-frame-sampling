"""
precompute_target_keyframes.py

Computes and saves TFS-sampled frames for later evaluation. Can be used with eval_mvbench_tfs.py for quicker benchmarks.
"""


import multiprocessing
from pathlib import Path
import json
from mvbench import find_video
from tqdm.auto import tqdm
from keyframes import TargetFrameSampler
import pandas as pd
import argparse

N_PARALLEL = 7

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
    i, video_dir, row, trimmed, target = args
    video_path = find_video(Path(video_dir), Path(row.video))
    start = None
    end = None
    if trimmed:
        start = row.start
        end = row.end
    sampler = TargetFrameSampler(target_frames=target, log_tqdm=True)
    res = sampler.solve(video_path)
    if trimmed:
        res["start"] = start
        res["end"] = end
    return i, res

def run_mvbench_category(mv_df: pd.DataFrame, video_dir: Path, trimmed: bool, target: int):
    results = []
    all_args = [[i, video_dir, row, trimmed, target] for i, row in mv_df.iterrows()]
    with multiprocessing.Pool(N_PARALLEL) as p:
        results = list(tqdm(p.imap(run_mvbench_one, all_args), total=mv_df.shape[0]))
        results_map = {i: kfs for (i, kfs) in results}
        return results_map

def run_mvbench_h264(target: int):
    results_dict = {}
    for i, category in enumerate(data_list):
        json, video_dir, _, trimmed = data_list[category]
        print(f"({i + 1:02d}/19) Starting", category)
        
        df = pd.read_json(Path(data_dir) / Path(json), orient="records")
        results = run_mvbench_category(df, Path(video_dir), trimmed, target)
        results_dict[category] = results
        print("Finished", category)
    print("Done")
    return results_dict  

if __name__ == "__main__":
    parser = argparse.ArgumentParser("precompute_target_keyframes")
    parser.add_argument("--frames", type=int, default=16)
    args = parser.parse_args()
    target = args.frames
    print(f"Computing {target} keyframes for each MVBench video")
    results = run_mvbench_h264(target)
    with open(f"computed_keyframes/mvbench_target={target}.json", "w") as f:
        json.dump(results, f)
