# Codec-based Frame Sampling

Have you ever wondered if the file format of your video can automatically determine the best frames to pass to a multimodal large language model? Have you ever wondered if this approach can give the same accuracy as uniform sampling with a lower token cost?

Well great, because **that what I've been wondering about too.** This project aims to leverage H.264 encoding to select I-frames from the video stream and use them as thumbnails for video understanding tasks. By intelligently constructing our encoding parameters, we should be able to see comperable performance to uniform sampling with fewer image input tokens.

## Getting Started

To get started, install the dependencies from `requirements.txt` and ensure that `ffmpeg` is installed on your system with `libx264`. If you want to run benchmarks on MVBench, pull the LFS from HuggingFace into `data/MVBench` and unzip the necessary files. For MVBench evals, also make sure that the `GEMINI_API_KEY` env variable is set on your system.

## Usage

For MVBench evaluation, please use `eval_mvbench_frs.py` and `eval_mvbench_tfs.py`. Uniform sampling baselines can also be ran with the `--uniform` argument. Some other scripts are included, but mostly as artifacts from the project as I precomputed some of the keyframes before running benchmarks. These two scripts do not require precomputation and should work correctly.

I've also made convenience classes for using target frame sampling and frame reduction sampling outside of the benchmarks. Example usage:

```python
from keyframes import TargetFrameSampler, FrameReductionSampler
from pathlib import Path

video_path = Path("path/to/video")

frs = FrameReductionSampler(sample_rate=Fraction(1), scenecut=80, compression_factor=2)
frs_frames = frs.solve(video_path)
frs_frames_uniform = frs.uniform(video_path)

tfs = TargetFrameSampler(target_frames=16)
tfs_frames = tfs.solve(video_path)
tfs_frames_uniform = tfs.uniform(video_path)
```

Other helpful convience methods:

- `get_frames` for exporting frames from video into files with ffmpeg
- Many MVBench-specific functions in `mvbench.py` for evaluation w/ Gemini.
- `Sampler.get_video_meta` for getting video FPS and total frame count.
- 