from fractions import Fraction
from pathlib import Path
import uuid
from PIL import Image
import subprocess
import glob

def get_frames(path: Path, frames: list[int]) -> list[str]:
    prefix = str(uuid.uuid4())
    if path.is_dir():
        for f in frames:
            img: Image.Image = Image.open(path / f"{f+1:05d}.jpg")
            # Resize for optimal token usage w/ gemini
            img.thumbnail((768, 768), Image.Resampling.LANCZOS) # pyright: ignore
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
            "-q:v", "2", # High quality jpeg
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
          
def get_duration(path: Path) -> float:
  if path.is_dir():
    return len([f for f in path.iterdir() if f.is_file()]) / 3.0
  result = subprocess.run(
    ["ffprobe", 
     "-v", "error", 
     "-show_entries", "format=duration", 
     "-of", "default=noprint_wrappers=1:nokey=1", 
     str(path)
    ],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT
  )
  return float(result.stdout)