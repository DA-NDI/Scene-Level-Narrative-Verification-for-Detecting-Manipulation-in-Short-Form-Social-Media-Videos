import os
import json
import random
import time
import argparse
from pathlib import Path

import cv2
import torch
import easyocr
import whisper
import numpy as np
from PIL import Image
from scenedetect import SceneManager, open_video
from scenedetect.detectors import ContentDetector, AdaptiveDetector

from facenet_pytorch import MTCNN
from transformers import pipeline
from google import genai

# ── Config ────────────────────────────────────────────────────────────────────
CONFIG_NAME  = "Fusion_3.1_Lite"  # Change this to rename your output JSON
LLM_MODEL    = "gemini-3.1-flash-lite-preview"

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
MTCNN_DEVICE = "cpu" if DEVICE == "mps" else DEVICE 

try:
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
except KeyError:
    print("FATAL: GEMINI_API_KEY environment variable not set.")
    exit(1)

DATA_FILE    = Path("../fakett/FakeTT_DATA_OPENSOURCE/data.json")
VIDEO_DIR    = Path("../fakett/FakeTT_DATA_OPENSOURCE/video")
RESULTS_DIR  = Path("../evaluation_results")

N_REAL       = 50
N_FAKE       = 50
RANDOM_SEED  = 42
MAX_SCENES   = 4

# ── Lazy Model Loading ────────────────────────────────────────────────────────
_models = {}

def get_models():
    if not _models:
        print(f"Loading Pipeline Models into Memory on {DEVICE}...")
        _models["mtcnn"] = MTCNN(keep_all=False, device=MTCNN_DEVICE)
        _models["deepfake"] = pipeline("image-classification", model="prithivMLmods/Deepfake-Detect-Siglip2", device=DEVICE)
        _models["ocr"] = easyocr.Reader(["en", "uk", "ru"], gpu=(DEVICE in ("cuda", "mps")), verbose=False)
        _models["whisper"] = whisper.load_model("base", device=DEVICE)
    return _models

# ── Scene Detection ───────────────────────────────────────────────────────────
def detect_scenes(video_path):
    for threshold in [27, 15]:
        try:
            video = open_video(str(video_path))
            sm = SceneManager()
            sm.add_detector(ContentDetector(threshold=threshold, min_scene_len=int(1.5*25)))
            sm.detect_scenes(video, show_progress=False)
            scenes = [s for s in sm.get_scene_list() if (s[1]-s[0]).get_seconds() >= 1.5]
            if len(scenes) > 1:
                return [(s[0].get_seconds(), s[1].get_seconds(), s[0].get_frames(), s[1].get_frames()) for s in scenes]
        except Exception:
            pass
    return []

# ── Feature Extraction Streams ────────────────────────────────────────────────
def process_video_streams(video_path):
    scenes = detect_scenes(video_path)
    models = get_models()
    
    scene_data = []
    visual_scores = []
    
    cap = cv2.VideoCapture(str(video_path))
    
    for (start_sec, end_sec, start_frame, end_frame) in scenes[:MAX_SCENES]:
        mid_frame = start_frame + ((end_frame - start_frame) // 2)
        cap.set(cv2.CAP_PROP_POS_FRAMES, mid_frame)
        ret, frame = cap.read()
        
        if not ret: continue
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # --- Stream A: Pixel Level Detection ---
        boxes, _ = models["mtcnn"].detect(frame_rgb)
        scene_df_score = None
        if boxes is not None:
            box = [int(b) for b in boxes[0]]
            face_array = frame_rgb[max(0, box[1]):box[3], max(0, box[0]):box[2]]
            if face_array.size > 0:
                face_pil = Image.fromarray(face_array)
                result = models["deepfake"](face_pil)
                is_fake = result[0]['label'].lower() in ['deepfake', 'fake']
                scene_df_score = result[0]['score'] if is_fake else 1.0 - result[0]['score']
                visual_scores.append(scene_df_score)

        # --- Stream B (Part 1): OCR Text Extraction ---
        ocr_texts = models["ocr"].readtext(frame, detail=0)
        scene_ocr = " | ".join(ocr_texts) if ocr_texts else "None"
        
        scene_data.append({
            "start": start_sec, 
            "end": end_sec, 
            "ocr": scene_ocr,
            "df_score": round(scene_df_score, 4) if scene_df_score else "No Face Detected"
        })

    cap.release()
    
    # --- Stream B (Part 2): Audio Extraction ---
    try:
        transcript_res = models["whisper"].transcribe(str(video_path))
        transcript = transcript_res.get("text", "").strip()[:600]
    except Exception as e:
        transcript = f"[ASR Error: {e}]"
        
    avg_df_score = float(np.mean(visual_scores)) if visual_scores else 0.0
    
    return scene_data, transcript, avg_df_score

# ── LLM Fusion & Reasoning ────────────────────────────────────────────────────
def evaluate_narrative(entry, scene_data, transcript, avg_df_score):
    scene_lines = ""
    for i, sd in enumerate(scene_data):
        scene_lines += f"  Scene {i+1} ({sd['start']:.1f}s-{sd['end']:.1f}s): Pixel Artifact Score: {sd['df_score']} | OCR: '{sd['ocr']}'\n"
    
    if not scene_lines:
        scene_lines = "  (no distinct scenes extracted)\n"

    prompt = f"""You are a forensic video analyst evaluating a social media video for manipulation.

STEP 1 — Review the visual artifact scores (0.0 = Real, 1.0 = Fake Face Detected).
STEP 2 — Read the textual and audio context. 
STEP 3 — Determine if the video is manipulated either physically (deepfake) OR narratively (authentic footage with false context/cheapfake).

METADATA:
- Claimed event/Post description: {entry.get('description', '')[:300]}
- Overall Pixel Deepfake Score: {avg_df_score:.4f}

Audio transcript (Whisper):
{transcript or 'N/A'}

Scene Data (OCR & Artifacts):
{scene_lines}

Return ONLY valid JSON (no markdown, no explanation):
{{
  "score": <integer 0-100>,
  "label": "fake" or "real",
  "primary_signal": "pixel_artifact" | "visual_text_mismatch" | "audio_text_mismatch" | "authentic",
  "reasoning": "1-2 sentences explaining the verdict based on the provided cross-modal data."
}}

Scoring: 0-30=authentic | 31-60=ambiguous | 61-100=manipulated"""

    try:
        response = client.models.generate_content(
            model=LLM_MODEL,
            contents=prompt
        )
        
        text = response.text.strip()
        if "```json" in text:
            text = text.split("```json", 1)[1].split("```", 1)[0].strip()
        elif "```" in text:
            text = text.split("```", 1)[1].split("```", 1)[0].strip()
            
        parsed = json.loads(text)
        score = max(0, min(100, int(parsed.get("score", 0))))
        label = str(parsed.get("label", "fake")).strip().lower()
        
        return {
            "score": score,
            "label": label,
            "primary_signal": parsed.get("primary_signal", "unknown"),
            "reasoning": parsed.get("reasoning", ""),
            "pixel_score": avg_df_score
        }
    except Exception as e:
        return {"score": 50, "label": "fake", "primary_signal": "error", "reasoning": str(e), "pixel_score": avg_df_score}

# ── Metrics ───────────────────────────────────────────────────────────────────
def calculate_metrics(records):
    valid = [r for r in records if r.get("pred_label") in ["fake", "real"]]
    if not valid: return {}
    tp = sum(1 for r in valid if r["gold_label"]=="fake" and r["pred_label"]=="fake")
    tn = sum(1 for r in valid if r["gold_label"]=="real" and r["pred_label"]=="real")
    fp = sum(1 for r in valid if r["gold_label"]=="real" and r["pred_label"]=="fake")
    fn = sum(1 for r in valid if r["gold_label"]=="fake" and r["pred_label"]=="real")
    pr = tp/(tp+fp) if (tp+fp)>0 else 0
    re = tp/(tp+fn) if (tp+fn)>0 else 0
    f1 = 2*pr*re/(pr+re) if (pr+re)>0 else 0
    return {"n":len(valid),"tp":tp,"tn":tn,"fp":fp,"fn":fn, "accuracy":(tp+tn)/len(valid),"precision":pr,"recall":re,"f1":f1}

# ── Main Execution ────────────────────────────────────────────────────────────
def run_pipeline():
    RESULTS_DIR.mkdir(exist_ok=True)
    out_file = RESULTS_DIR / f"{CONFIG_NAME}_results.json"

    # Load & Sample Data
    rows = []
    with DATA_FILE.open() as f:
        for line in f:
            if line.strip(): rows.append(json.loads(line.strip()))

    eligible = [r for r in rows if r.get("annotation") in {"fake","real"} and (VIDEO_DIR/f"{r['video_id']}.mp4").exists()]
    random.seed(RANDOM_SEED)
    sample = random.sample([r for r in eligible if r["annotation"]=="real"], N_REAL) + \
             random.sample([r for r in eligible if r["annotation"]=="fake"], N_FAKE)
    random.shuffle(sample)

    # Resume State
    results, done_ids = [], set()
    if out_file.exists():
        saved = json.load(out_file.open())
        results = saved.get("results", [])
        done_ids = {r["video_id"] for r in results}
        print(f"Resuming: {len(done_ids)} videos already processed for {CONFIG_NAME}.")

    GREEN, RED, RESET = "\033[32m", "\033[31m", "\033[0m"

    for idx, row in enumerate(sample, 1):
        vid_id = row["video_id"]
        if vid_id in done_ids: continue

        vid_path = VIDEO_DIR / f"{vid_id}.mp4"
        scene_data, transcript, avg_df_score = process_video_streams(vid_path)
        eval_res = evaluate_narrative(row, scene_data, transcript, avg_df_score)

        result_entry = {
            "video_id": vid_id,
            "gold_label": row["annotation"],
            "pred_label": eval_res["label"],
            "pred_score": eval_res["score"],
            "pixel_artifact_score": avg_df_score,
            "correct": eval_res["label"] == row["annotation"],
            "primary_signal": eval_res["primary_signal"],
            "reasoning": eval_res["reasoning"]
        }
        
        m = calculate_metrics(results + [result_entry])
        color = GREEN if result_entry["correct"] else RED
        mark = "✓" if result_entry["correct"] else "✗"

        print(f"{color}[{idx:>3}/100] {mark} gold={row['annotation']:<4} pred={result_entry['pred_label']:<4} "
              f"LLM_score={result_entry['pred_score']:>3} | Pixel_ViT={avg_df_score:.2f}{RESET}")
        print(f"         → {result_entry['reasoning'][:120]}...")
        
        if m:
            print(f"         [{m['n']:>3}] Acc={m['accuracy']:.2f} P={m['precision']:.2f} R={m['recall']:.2f} F1={m['f1']:.2f}")

        results.append(result_entry)
        done_ids.add(vid_id)

        # Save with config identifier
        with out_file.open("w", encoding="utf-8") as f:
            json.dump({
                "config": CONFIG_NAME, 
                "model": LLM_MODEL, 
                "metrics": calculate_metrics(results), 
                "results": results
            }, f, indent=2)
            
        time.sleep(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="End-to-End Multimodal Deepfake & Narrative Pipeline")
    args = parser.parse_args()
    run_pipeline()