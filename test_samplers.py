from fractions import Fraction
from keyframes import Sampler, TargetFrameSampler, FrameReductionSampler
from pathlib import Path
import math

video_path = Path("test_data/0KZYF.mp4")

def test_video_info():
  sampler = Sampler()
  total_frames, fps = sampler.get_video_meta(video_path)
  assert total_frames == 921
  assert math.isclose(fps, 29.97, rel_tol=0.001)
  
def test_tfs():
  sampler = TargetFrameSampler(target_frames=16)
  keyframes = sampler.solve(video_path)["keyframes"]
  keyframes_trim = sampler.solve(video_path, start=1, end=4)["keyframes"]
  uniform = sampler.uniform(video_path)
  uniform_trim = sampler.uniform(video_path, start=1, end=4)
  assert keyframes != keyframes_trim
  assert uniform != uniform_trim
  assert len(keyframes) == 16
  assert len(keyframes_trim) == 16
  assert len(uniform) == 16
  assert len(uniform_trim) == 16
  
  assert uniform[0] == 0
  assert keyframes[0] == 0
  assert uniform_trim[0] == 29
  assert keyframes_trim[0] == 29

def test_frs():
  sampler = FrameReductionSampler(Fraction(1), 80, 2.0)
  keyframes = sampler.solve(video_path)
  keyframes_trim = sampler.solve(video_path, start=1, end=4)
  uniform = sampler.uniform(video_path)
  uniform_trim = sampler.uniform(video_path, start=1, end=4)
  assert keyframes != keyframes_trim
  assert uniform != uniform_trim
  assert len(keyframes) > len(keyframes_trim)
  assert len(uniform) > len(uniform_trim)
  assert len(uniform) > len(keyframes)
  assert len(uniform_trim) > len(keyframes_trim)

  
  assert uniform[0] == 0
  assert keyframes[0] == 0
  assert uniform_trim[0] == 29
  assert keyframes_trim[0] == 29

