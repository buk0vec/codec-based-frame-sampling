from fractions import Fraction
from pathlib import Path
import subprocess
import json
from tqdm.auto import tqdm
import numpy as np
import math

# Extract I-frame indices from a video using ffmpeg
def get_keyframes_ffmpeg(
    path: Path | str,
    video_fr: Fraction,
    keyint_min: int = 8,
    keyint_max: int = 24,
    sc_threshold: int = 40,
    bframes: int = 0,  # -1 uses libx264 default of 3
    trim_start: float | None = None,
    trim_end: float | None = None,
    n_threads: int = 1,
    scale: str = "360",
    lookahead_cap: int | None=60,
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
    
    # Allow lookahead to be capped for encoding efficiency. 
    rc_lookahead = keyint_max
    if lookahead_cap is not None:
        rc_lookahead = min(keyint_max, lookahead_cap)
    
    if p.is_dir():
        cmd = [
        "ffmpeg", 
        "-nostdin",
        "-threads", str(n_threads), # For parallel use
        *trim_args, 
        "-framerate", "3", # tvqa videos all are at 3 fps
        "-i", str(p / "%05d.jpg"), 
        "-c:v", "libx264", 
        "-preset", "superfast",
        "-v", "error",
        "-vf", f"scale=-2:{scale}", # Make sure h/w is even
        "-x264-params", f"rc-lookahead={rc_lookahead}", 
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
        "-threads", str(n_threads), # For parallel use
        "-vf", f"scale=-2:{scale}", # Make sure h/w is even
        "-g", str(keyint_max), 
        "-keyint_min", str(keyint_min), 
        "-x264-params", f"rc-lookahead={rc_lookahead}", 
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
  
  
def get_ffmpeg_keyframe_indices_for_target_frame_rate(
    path: Path | str,
    video_fr: Fraction,
    target_rate: Fraction,
    compression_ratio: Fraction,
    sc_threshold: int = 40,
    bframes: int = 0,
    start: float | None = None,
    end : float | None = None,
):
    # Use ceiling so we don't oversample
    min_scene_len = math.ceil(float(video_fr / target_rate))
    max_scene_len = math.ceil(float(video_fr / target_rate * compression_ratio))

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

class Sampler:
    def __init__(self, log=False, log_tqdm=False):
        self.log = log
        self.log_tqdm = log_tqdm
    
    
    def _print(self, s):
        if self.log:
            if self.log_tqdm:
                tqdm.write(s)
            else:
                print(s)
    
    def get_video_meta(self, path: Path) -> tuple[int, Fraction]:
        # Handle directories containing video frames. All of these are 3 FPS in MVBench,
        # different handling needed for usage outside of MVBench.
        if path.is_dir():
            return len([f for f in path.iterdir() if f.is_file()]), Fraction(3)
        cmd = [
            "ffprobe", 
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=avg_frame_rate,nb_frames", 
            "-of", "json", 
            str(path)
        ]
        out = subprocess.check_output(cmd)
        data = json.loads(out)["streams"][0]
        fps_str = data.get("avg_frame_rate", "30/1")
        if fps_str == "0/0": 
            fps_str = "30/1"
        num, den = map(int, fps_str.split('/'))
        fps = Fraction(num, den) if den != 0 else Fraction(30, 1)
        
        # Estimate duration/frames
        nb_frames = data.get("nb_frames")
        if nb_frames:
            total = int(nb_frames)
        else:
            # Fallback
            cmd_dur = ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)]
            dur = float(subprocess.check_output(cmd_dur).strip())
            total = int(dur * float(fps))
            
        return total, fps
    
    
class FrameReductionSampler(Sampler):
    def __init__(self, sample_rate: Fraction, scenecut: int, compression_factor: float, log=False, log_tqdm=False):
        self.sample_rate = sample_rate
        self.compression_factor = compression_factor
        self.scenecut = scenecut
        super().__init__(log=log, log_tqdm=log_tqdm)
        
    def uniform(self, path: Path, start: float | None = None, end: float | None = None):
        total_frames, fps = self.get_video_meta(path)
        start_frame = 0
        
        if end is not None:
            end_frame = int(fps * end)
            if end_frame >= total_frames:
                end = None
            else:
                total_frames = end_frame
        if start is not None:
            start_frame = int(fps * start)
            # Account for incorrect start timings
            if start_frame >= total_frames:
               start = None
            else:
                total_frames -= start_frame
        
        frames_per_sample = int(fps / self.sample_rate)
        frame_idx = [i for i in range(start_frame, start_frame + total_frames, frames_per_sample)]
        return frame_idx
    
    def solve(self, path: Path, start: float | None = None, end: float | None = None):
        total_frames, fps = self.get_video_meta(path)
        
        start_frame = 0
        
        if end is not None:
            end_frame = int(fps * end)
            if end_frame >= total_frames:
                end = None
            else:
                total_frames = end_frame
        if start is not None:
            start_frame = int(fps * start)
            # Account for incorrect start timings
            if start_frame >= total_frames:
               start = None
            else:
                total_frames -= start_frame
        
        keyframes = get_ffmpeg_keyframe_indices_for_target_frame_rate(
            path,
            fps,
            self.sample_rate,
            Fraction(self.compression_factor),
            self.scenecut,
            start = start,
            end = end
        )
        
        return keyframes
        
        

class TargetFrameSampler(Sampler):
    def __init__(self, target_frames=16, max_feasible_iter=100, max_binary_iter=9, log=False, log_tqdm=False, sc_max=300):
        self.target = target_frames
        self.feasible_iter = max_feasible_iter
        self.binary_iter = max_binary_iter
        self.sc_max = sc_max
        super().__init__(log=log, log_tqdm=log_tqdm)

    def _run_probe(self, path: Path, fps: Fraction, sc: int, k_min: int, k_max: int, scale="240", lc: int | None=None, start=None, end=None):
        frames, count, _ = get_keyframes_ffmpeg(
            path, 
            fps, 
            keyint_min=k_min, 
            keyint_max=k_max, 
            sc_threshold=sc,
            bframes=0,
            trim_start=start,
            trim_end=end,
            n_threads=1,
            scale=scale,
            lookahead_cap=lc
        )
        return frames, count
    
    def solve(self, path, start=None, end=None) -> dict:
        total_frames, fps = self.get_video_meta(path)
        start_frame = 0
        
        if end is not None:
            end_frame = int(fps * end)
            if end_frame >= total_frames:
                end = None
            else:
                total_frames = end_frame
        if start is not None:
            start_frame = int(fps * start)
            # Account for incorrect start timings
            if start_frame >= total_frames:
               start = None
            else:
                total_frames -= start_frame
        
        # If we have <= target frames, return all the frames we can
        if total_frames <= self.target:
            self._print(f"{path} has {total_frames} frames <= {self.target}, uniformly sampling.")
            return {"keyframes": self._interp_frame_indices(list(range(start_frame, start_frame+total_frames)))}

        # Fix keyint_min (nyquist sampling freq) total_frames/(target * 2), clipped between [1, 4] so that we don't just end up uniformly sampling
        keyint_min = min(4, max(int(total_frames/(self.target * 2)), 1))
        # print("Finding feasible")
        # Find feasible keyint_max
        feasible, keyint_max, keyframes, keyframe_count = self._find_feasible_max(path, total_frames, fps, keyint_min, start=start, end=end)

        # If we didn't find a feasible configuration, default to uniform sampling
        if not feasible:
            return {"keyframes": self._interp_frame_indices(list(range(start_frame, start_frame+total_frames)))}

        # Return early if we have the correct amount of frames from feasibility check
        if keyframe_count == self.target:
            return  {
                "keyframes": keyframes,
                "keyint_min": keyint_min,
                "keyint_max": keyint_max,
                "scenecut": self.sc_max
            }
        # print("BS", path, fps, keyint_min, keyint_max, start, end, start_frame, total_frames, keyframe_count)
        # Run binary search to find optimal scenecut
        sc, keyframes, keyframe_count = self._find_scenecut_bs(path, fps, keyint_min, keyint_max, start=start, end=end)

        # Interpolate down if needed and return
        return  {
            "keyframes": self._interp_frame_indices(keyframes),
            "keyint_min": keyint_min,
            "keyint_max": keyint_max,
            "scenecut": sc
        }

    def uniform(self, path, start=None, end=None):
        total_frames, fps = self.get_video_meta(path)
        start_frame = 0
        
        if end is not None:
            end_frame = int(fps * end)
            if end_frame >= total_frames:
                end = None
            else:
                total_frames = end_frame
        if start is not None:
            start_frame = int(fps * start)
            # Account for incorrect start timings
            if start_frame >= total_frames:
               start = None
            else:
                total_frames -= start_frame
        
        return self._interp_frame_indices(list(range(start_frame, start_frame+total_frames)))
    
    def _find_scenecut_bs(self, path, fps, keyint_min, keyint_max, start=None, end=None) -> tuple[int, list[int], int]:
        # Binary search for optimal scenecut
        sc = self.sc_max
        sc_min = 0
        last_feasible_sc = sc
        last_feasible_keyframes = []
        for i in range(self.binary_iter):
            keyframes, count = self._run_probe(path, fps, sc, keyint_min, keyint_max, start=start, end=end)
            if count == self.target:
                return sc, keyframes, count
            if count > self.target:
                last_feasible_sc = sc
                last_feasible_keyframes = keyframes
                sc_max = sc
            else:
                sc_min = sc
            sc = sc_min + (sc_max - sc_min) // 2
        self._print(f"WARN: binary scenecut search reahed max iterations.\npath={path} keyint_min={keyint_min} keyint_max={keyint_max} sc={sc} feasible_sc={last_feasible_sc} n_kfs={len(last_feasible_keyframes)}")
        return last_feasible_sc, last_feasible_keyframes, len(last_feasible_keyframes)
        
    def _find_feasible_max(self, path, total_frames, fps, keyint_min, alpha=0.9, start=None, end=None) -> tuple[bool, int, list[int], int]:
        # First find feasible start, use keyint_max = total_frames - target*keyint_min and scenecut = 300. While we are getting less keyframes than target, decrease 
        # keyint_max by a factor. Do this until n_frames >= target.
        infeasible = (False, -1, [], 0)
        if total_frames < self.target:
            self._print(f"WARN: feasibility check failed for path={path}, less than {self.target} frames")
            return infeasible
        keyint_max = total_frames - self.target*keyint_min
        for i in range(self.feasible_iter):
            frames, count = self._run_probe(path, fps, self.sc_max, keyint_min, keyint_max, start=start, end=end)
            if count >= self.target:
                return True, keyint_max, frames, count
            keyint_max = int(alpha * keyint_max)
            if keyint_max <= keyint_min:
                self._print(f"WARN: feasiblility check hit keyint_min\npath={path} total_frames={total_frames} keyint_min={keyint_min} keyint_max={keyint_max}")
                return infeasible
        self._print(f"WARN: feasiblility check reached max iterations\npath={path} total_frames={total_frames} keyint_min={keyint_min} keyint_max={keyint_max}")
        return infeasible
            
        

    def _interp_frame_indices(self, indices) -> list[int]:
        if len(indices) == self.target:
            return indices
        target_idx = np.linspace(0, len(indices)-1, self.target).astype(int) # pyright: ignore
        
        return [indices[i] for i in target_idx]

