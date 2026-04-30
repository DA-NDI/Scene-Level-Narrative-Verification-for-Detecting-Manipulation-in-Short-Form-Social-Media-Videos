"""
fmnv_gemma_eval.py — FMNV evaluation with Gemma-4 local inference

Config H: Gemma-4 local (on PC with RTX 3090)
- Same modalities as Gemini configs (OCR + Whisper + keyframes)
- Uses Ollama with OpenAI-compatible API

Run on PC after pulling Gemma-4 model:
  ollama pull gemma4:27b
  export OLLAMA_URL="http://localhost:11434/v1/chat/completions"
  python dyplom_v2/fmnv_gemma_eval.py --config H
"""

import argparse
import json
import base64
from io import BytesIO
from pathlib import Path
import numpy as np
import cv2
import whisper
import easyocr
from PIL import Image
import torch
import requests
from scenedetect import SceneManager, open_video
from scenedetect.detectors import ContentDetector, AdaptiveDetector

# ── Configuration ──────────────────────────────────────────────────────────────
FMNV_DIR      = Path("FMNV")
DATA_FILE     = FMNV_DIR / "data.json"
VIDEO_DIR     = FMNV_DIR / "videos"
RESULTS_DIR   = Path("ablation_results_fmnv")
OLLAMA_URL    = "http://localhost:11434/v1/chat/completions"
MODEL_NAME    = "gemma4:26b"
RANDOM_SEED   = 42
MAX_SCENES    = 4

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── Lazy model loading ─────────────────────────────────────────────────────────
_models = {}

def get_ocr():
    if "ocr" not in _models:
        print("Loading EasyOCR...")
        _models["ocr"] = easyocr.Reader(["en", "uk", "ru"], gpu=(DEVICE == "cuda"), verbose=False)
    return _models["ocr"]

def get_whisper():
    if "whisper" not in _models:
        print("Loading Whisper...")
        _models["whisper"] = whisper.load_model("base", device="cpu")
    return _models["whisper"]

# ── Scene detection ────────────────────────────────────────────────────────────
def detect_scenes(video_path):
    """Cascading scene detector — same as all other configs."""
    for threshold in [27, 15]:
        try:
            video = open_video(str(video_path))
            sm = SceneManager()
            sm.add_detector(ContentDetector(threshold=threshold, min_scene_len=int(1.5*25)))
            sm.detect_scenes(video, show_progress=False)
            scenes = [s for s in sm.get_scene_list() if (s[1]-s[0]).get_seconds() >= 1.5]
            if len(scenes) > 1:
                return [(s[0].get_seconds(), s[1].get_seconds()) for s in scenes]
        except Exception:
            pass

    try:
        video = open_video(str(video_path))
        sm = SceneManager()
        sm.add_detector(AdaptiveDetector(adaptive_threshold=3.0))
        sm.detect_scenes(video, show_progress=False)
        scenes = [s for s in sm.get_scene_list() if (s[1]-s[0]).get_seconds() >= 1.5]
        if len(scenes) > 1:
            return [(s[0].get_seconds(), s[1].get_seconds()) for s in scenes]
    except Exception:
        pass

    # Fallback
    try:
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        cap.release()
        if duration > 0:
            return [(i*10, min((i+1)*10, duration)) for i in range(int(duration // 10) + 1)]
    except Exception:
        pass

    return []

# ── Modality extraction ────────────────────────────────────────────────────────
def extract_ocr(video_path, scenes):
    """Extract text overlays from scenes."""
    ocr_reader = get_ocr()
    all_text = []
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25

    for scene_start, scene_end in scenes[:MAX_SCENES]:
        frame_idx = int(scene_start * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue
        try:
            results = ocr_reader.readtext(frame)
            text = " ".join([r[1] for r in results])
            all_text.append(text)
        except Exception:
            pass

    cap.release()
    return " ".join(all_text)

def extract_whisper(video_path):
    """Extract audio transcript."""
    try:
        whisper_model = get_whisper()
        result = whisper_model.transcribe(str(video_path), language="en", verbose=False)
        return result["text"]
    except Exception:
        return ""

def extract_keyframes(video_path, scenes):
    """Extract keyframe images as base64."""
    keyframes_b64 = []
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25

    for scene_start, scene_end in scenes[:MAX_SCENES]:
        frame_idx = int((scene_start + (scene_end - scene_start) / 2) * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            # Compress
            buffered = BytesIO()
            pil_img.save(buffered, format="JPEG", quality=85)
            img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
            keyframes_b64.append(img_b64)

    cap.release()
    return keyframes_b64

# ── Ollama inference ───────────────────────────────────────────────────────────
def gemma_evaluate(ocr_text, audio_text, keyframes_b64):
    """Send to Gemma-4 via Ollama for narrative verification."""

    # Build message content
    content = [
        {"type": "text", "text": "You are a video forensics analyst."}
    ]

    # Add keyframes
    for i, kf_b64 in enumerate(keyframes_b64):
        content.append({
            "type": "image",
            "image": {
                "url": f"data:image/jpeg;base64,{kf_b64}"
            }
        })
        content.append({"type": "text", "text": f"[Keyframe {i+1}]"})

    # Add prompt
    prompt = f"""Analyze this video's narrative consistency.

OCR Text:
{ocr_text if ocr_text else "(no text found)"}

Audio Transcript:
{audio_text if audio_text else "(no audio found)"}

Assess on scale 0-100:
- 0-30: Consistent (authentic)
- 30-70: Ambiguous
- 70-100: Inconsistent (manipulated)

Respond with JSON: {{"score": <0-100>, "reasoning": "<brief>"}}"""

    content.append({"type": "text", "text": prompt})

    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL_NAME,
                "messages": [{"role": "user", "content": content}],
                "temperature": 0.7,
            },
            timeout=60
        )
        response.raise_for_status()
        result = response.json()
        text = result["choices"][0]["message"]["content"].strip()

        # Extract JSON
        if "{" in text and "}" in text:
            start = text.index("{")
            end = text.rindex("}") + 1
            json_str = text[start:end]
            parsed = json.loads(json_str)
            return parsed.get("score", 50), parsed.get("reasoning", "")
        else:
            return 50, "Could not parse response"
    except Exception as e:
        return 50, f"Error: {str(e)}"

# ── Main evaluation ────────────────────────────────────────────────────────────
def evaluate_config_h():
    """Run Config H (Gemma-4 local)."""
    print("\n🎬 RUNNING CONFIG H (Gemma-4:26b local)")

    # Load FMNV data
    with open(DATA_FILE) as f:
        all_data = json.load(f)

    # Filter to available videos
    video_files = {v.stem: v for v in VIDEO_DIR.glob("*.mp4")}
    available = [d for d in all_data if d['video_id'] in video_files]

    print(f"   Found {len(available)} videos")

    results = []
    for i, entry in enumerate(available):
        vid_id = entry['video_id']
        video_path = video_files[vid_id]
        gold_label = "fake" if entry['label'].lower() == 'false' else "real"

        if i > 0 and i % 10 == 0:
            print(f"   Progress: {i}/{len(available)}")

        try:
            # Extract modalities
            scenes = detect_scenes(video_path)
            ocr_text = extract_ocr(video_path, scenes)
            audio_text = extract_whisper(video_path)
            keyframes_b64 = extract_keyframes(video_path, scenes)

            # Get prediction
            pred_score, reasoning = gemma_evaluate(ocr_text, audio_text, keyframes_b64)
            pred_label = "fake" if pred_score >= 50 else "real"
            correct = pred_label == gold_label

            results.append({
                "video_id": vid_id,
                "gold_label": gold_label,
                "pred_label": pred_label,
                "pred_score": pred_score,
                "correct": correct,
                "scene_count": len(scenes),
                "ocr_length": len(ocr_text),
                "audio_length": len(audio_text),
                "reasoning": reasoning[:100]
            })
        except Exception as e:
            results.append({
                "video_id": vid_id,
                "gold_label": gold_label,
                "pred_label": None,
                "pred_score": None,
                "correct": False,
                "error": str(e)
            })

    # Compute metrics
    valid = [r for r in results if r.get('pred_score') is not None]
    if valid:
        tp = sum(1 for r in valid if r['gold_label'] == 'fake' and r['pred_label'] == 'fake')
        tn = sum(1 for r in valid if r['gold_label'] == 'real' and r['pred_label'] == 'real')
        fp = sum(1 for r in valid if r['gold_label'] == 'real' and r['pred_label'] == 'fake')
        fn = sum(1 for r in valid if r['gold_label'] == 'fake' and r['pred_label'] == 'real')

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        acc = (tp + tn) / len(valid)

        metrics = {
            "n": len(valid),
            "accuracy": round(acc, 3),
            "precision": round(prec, 3),
            "recall": round(rec, 3),
            "f1": round(f1, 3),
            "tp": tp, "tn": tn, "fp": fp, "fn": fn
        }
    else:
        metrics = {}

    # Save results
    RESULTS_DIR.mkdir(exist_ok=True)
    output_file = RESULTS_DIR / f"fmnv_H.json"
    with open(output_file, 'w') as f:
        json.dump({"config": "H", "model": MODEL_NAME, "metrics": metrics, "results": results}, f, indent=2)

    print(f"   Results saved to {output_file}")
    if metrics:
        print(f"   F1: {metrics['f1']:.3f} | Acc: {metrics['accuracy']:.3f} | Prec: {metrics['precision']:.3f} | Rec: {metrics['recall']:.3f}")

if __name__ == "__main__":
    evaluate_config_h()
