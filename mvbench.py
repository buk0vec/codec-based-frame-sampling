import glob
from pathlib import Path
import backoff
from google import genai
from google.genai import types
import httpx
import os
from PIL import Image
from google.genai.errors import APIError

class NoResponseError(Exception):
  pass

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

# Either loads keyframes to memory or uploads them with File API
@backoff.on_exception(backoff.expo, (httpx.HTTPError, APIError))
def upload_thumbnails(client, keyframes: list[Path]):
  file_sizes = [os.path.getsize(kf) for kf in keyframes]
  inline = False
  # Inline files if sub 15 MB
  if sum(file_sizes) < 15_000_000:
      inline = True
      remote_kfs = [Image.open(kf) for kf in keyframes]
  else:
      remote_kfs = [client.files.upload(file=kf) for kf in keyframes]
  return remote_kfs, inline
    

# Takes in question, candidates, answer, and tuple w/ Gemini-friendly frames and whether they are inlined
@backoff.on_exception(backoff.expo, (httpx.HTTPError, APIError, NoResponseError))
def prompt_gemini_mvbench(client, question, candidates, answer, remote_kfs):
    question, answer = qa_template(question, candidates, answer)
    system_prompt = "Carefully watch the video and pay attention to the cause and sequence of events, the detail and movement of objects, \
    and the action and pose of persons. Based on your observations, select the best option that accurately addresses the question.\n"
    question_prompt="\nOnly give the best option."
    full_prompt = question + question_prompt
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=[full_prompt, *remote_kfs],
        config=types.GenerateContentConfig(
            system_instruction=system_prompt, 
            # Temp = 0 for reproducability (and so I don't need to average multiple runs)
            temperature=0
        )
    )
    token_count = list(filter(lambda x: x.modality == types.MediaModality.IMAGE, response.usage_metadata.prompt_tokens_details))[0].token_count
    # Silent error catch
    if response is None:
      print("WARN: response missing from Gemini")
      raise NoResponseError()
    
    # Random denial catch
    if response.candidates is None:
      print("ERROR: empty candidates from Gemini, safety filter probably triggered")
      # Return a wrong answer immediately so the benchmark continues
      return "NO_RESPONSE", token_count, False
    candidate = response.candidates[0]
    
    # Unnatural generation catch
    if candidate.finish_reason != "STOP":
      print(f"ERROR: Blocked by Gemini. Reason: {candidate.finish_reason}")
      # Return a wrong answer immediately so the benchmark continues
      return "BLOCKED", token_count, False

    if response.text is None:
      print("WARN: text missing from Gemini")
      raise NoResponseError()
    
    text_list = response.text.split("(")
    correct = False if len(text_list) < 2 or text_list[1][0] != answer else True
    return response.text, token_count, correct

# Use separate functions for file upload and prompts so we get separate retry/backoff,
# this way we don't reload keyframes if Gemini fails.
def run_gemini_mvbench(keyframes, question, candidates, answer):
  client = genai.Client()
  kfs, inline = upload_thumbnails(client, keyframes)
  res = prompt_gemini_mvbench(client, question, candidates, answer, kfs)
  if not inline:
      for kf in kfs:
          client.files.delete(name=kf.name)
  return res
  
def find_video(video_dir: Path, stub: Path):
  video_path = video_dir / stub
  if not video_path.exists():
    supp_glob = str(Path("data/MVBench/video/data0613/*/**/") / stub)
    # print("No match found for file", video_path,  "checking supplementary folder...:")
    extra_data = glob.glob(supp_glob)
    if len(extra_data) == 0:
        print("No replacement found for {video_path}! That's not great")
        return None
    else:
        # print("Substitute file found:", extra_data[0])
        video_path = Path(extra_data[0])
  return video_path