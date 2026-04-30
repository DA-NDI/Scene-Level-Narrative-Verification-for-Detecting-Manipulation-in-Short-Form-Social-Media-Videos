"""
Narrative Manipulation Detection Pipeline — Evaluation on 100 Videos
Pipeline: PySceneDetect → OCR keyframes → Gemini LLM scoring
Comparison: our score vs commercial manipulation_score from d-tiktok dataset
"""

import os
import json
import csv
import time
import random
import cv2
import numpy as np
import easyocr
import google.generativeai as genai
from scenedetect import SceneManager, open_video
from scenedetect.detectors import AdaptiveDetector

# ── Config ────────────────────────────────────────────────────────────
VIDEO_FOLDER   = "./videos"
METADATA_FILE  = "d-tiktok_20251129_20251206.json"
RESULTS_FILE   = "pipeline_results.json"
SAMPLE_SIZE    = 100
RANDOM_SEED    = 42

genai.configure(api_key=os.environ["GEMINI_API_KEY"])
model = genai.GenerativeModel("gemini-2.5-flash")
ocr_reader = easyocr.Reader(['en', 'uk', 'ru'], gpu=False, verbose=False)

# ── Load metadata ─────────────────────────────────────────────────────
print("Loading metadata...")
with open(METADATA_FILE) as f:
    all_data = json.load(f)

# Filter to records that have local video + required fields
video_files = set(os.listdir(VIDEO_FOLDER))
eligible = [
    e for e in all_data
    if f"{e.get('id')}.mp4" in video_files
    and e.get('manipulation_score') is not None
    and e.get('visual_description')
]
print(f"Eligible (local video + metadata): {len(eligible)}")

# ── Stratified sample ─────────────────────────────────────────────────
random.seed(RANDOM_SEED)

low    = [e for e in eligible if e['manipulation_score'] <= 20]
medium = [e for e in eligible if 21 <= e['manipulation_score'] <= 59]
high   = [e for e in eligible if e['manipulation_score'] >= 60]

print(f"Pool — Low: {len(low)}, Medium: {len(medium)}, High: {len(high)}")

# 20 low / 60 medium / 20 high
sample = (
    random.sample(low,    min(20, len(low)))   +
    random.sample(medium, min(60, len(medium))) +
    random.sample(high,   min(20, len(high)))
)
random.shuffle(sample)
print(f"Sample size: {len(sample)}")

# ── Load already-processed results (for resumability) ─────────────────
if os.path.exists(RESULTS_FILE):
    with open(RESULTS_FILE) as f:
        results = json.load(f)
    done_ids = {r['id'] for r in results}
    print(f"Resuming — already done: {len(done_ids)}")
else:
    results = []
    done_ids = set()

# ── Pipeline functions ────────────────────────────────────────────────

def run_scene_detection(video_path):
    """Returns list of (start_sec, end_sec) for each scene."""
    try:
        video = open_video(video_path)
        sm = SceneManager()
        sm.add_detector(AdaptiveDetector(adaptive_threshold=27.0))
        sm.detect_scenes(video)
        scenes = sm.get_scene_list()
        # Filter very short flashes
        scenes = [s for s in scenes if (s[1] - s[0]).get_seconds() >= 0.5]
        return [(s[0].get_seconds(), s[1].get_seconds()) for s in scenes]
    except Exception as e:
        return []


def extract_ocr_from_frame(video_path, timestamp_sec):
    """Grabs one frame at timestamp and runs OCR."""
    try:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(timestamp_sec * fps))
        ret, frame = cap.read()
        cap.release()
        if not ret:
            return ""
        texts = ocr_reader.readtext(frame, detail=0)
        return " | ".join(texts) if texts else ""
    except Exception:
        return ""


def build_gemini_prompt(entry, scenes, ocr_texts):
    """Constructs the prompt for narrative manipulation detection."""
    visual_desc   = entry.get('visual_description', 'N/A')
    audio_context = entry.get('audio_context', 'N/A')
    description   = entry.get('translated_description', entry.get('description', 'N/A'))
    n_scenes      = len(scenes)
    total_dur     = scenes[-1][1] if scenes else entry.get('video_length', 0)

    scene_summary = ""
    for i, ((s, e), ocr) in enumerate(zip(scenes[:6], ocr_texts[:6])):
        scene_summary += f"  Scene {i+1} ({s:.1f}s–{e:.1f}s): OCR='{ocr}'\n"
    if not scene_summary:
        scene_summary = "  (no scenes detected)\n"

    prompt = f"""You are a fact-checking analyst specializing in video narrative manipulation.

Analyze this TikTok video for narrative manipulation — specifically "cheapfakes": authentic footage recontextualized through misleading captions, audio, or selective editing.

VIDEO METADATA:
- Duration: {total_dur:.1f}s
- Scene count (PySceneDetect): {n_scenes}
- Post description: {str(description)[:200]}

VISUAL DESCRIPTION (automated VLM analysis):
{str(visual_desc)[:400]}

AUDIO CONTEXT (automated transcription):
{str(audio_context)[:300]}

ON-SCREEN TEXT (OCR per scene):
{scene_summary}

TASK: Assess whether this video shows signs of narrative manipulation.
Score from 0 to 100 where:
  0  = clearly authentic, consistent narrative
  50 = ambiguous or slightly suspicious
  100 = clear narrative manipulation (misleading captions, out-of-context footage, etc.)

Respond in this exact JSON format:
{{
  "score": <integer 0-100>,
  "reasoning": "<2-3 sentences explaining key signals>"
}}"""
    return prompt


def call_gemini(prompt, retries=3):
    """Call Gemini with retry on rate limit."""
    for attempt in range(retries):
        try:
            response = model.generate_content(prompt)
            text = response.text.strip()
            # Extract JSON block
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0].strip()
            elif "```" in text:
                text = text.split("```")[1].split("```")[0].strip()
            parsed = json.loads(text)
            return int(parsed['score']), str(parsed['reasoning'])
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                return None, f"Error: {e}"


# ── Main loop ─────────────────────────────────────────────────────────
print(f"\nStarting pipeline on {len(sample)} videos...\n")

for idx, entry in enumerate(sample):
    vid_id = entry['id']
    if vid_id in done_ids:
        continue

    video_path = os.path.join(VIDEO_FOLDER, f"{vid_id}.mp4")
    commercial_score = entry['manipulation_score']

    print(f"[{idx+1:>3}/{len(sample)}] id={vid_id[:8]}… commercial={commercial_score}", end=" | ")

    # 1. Scene detection
    scenes = run_scene_detection(video_path)
    print(f"scenes={len(scenes)}", end=" | ")

    # 2. OCR on middle frame of each scene (max 6 scenes)
    ocr_texts = []
    for (start, end) in scenes[:6]:
        mid = (start + end) / 2
        ocr_texts.append(extract_ocr_from_frame(video_path, mid))

    # 3. Gemini LLM scoring
    prompt = build_gemini_prompt(entry, scenes, ocr_texts)
    our_score, reasoning = call_gemini(prompt)
    print(f"our={our_score}")

    result = {
        "id":               vid_id,
        "commercial_score": commercial_score,
        "our_score":        our_score,
        "scene_count":      len(scenes),
        "reasoning":        reasoning,
        "ocr_texts":        ocr_texts,
        "video_length":     entry.get('video_length'),
        "actor_country":    entry.get('actor_country'),
    }
    results.append(result)
    done_ids.add(vid_id)

    # Save after each video
    with open(RESULTS_FILE, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Rate limit: ~15 req/min on free tier
    time.sleep(1.5)

print(f"\nDone. Results saved to {RESULTS_FILE}")
print(f"Total processed: {len(results)}")
