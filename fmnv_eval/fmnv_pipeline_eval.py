"""
fmnv_pipeline_eval.py — Fact-verification pipeline on FMNV dataset

FMNV: Fake Multimodal News Video dataset
- 98 videos locally available
- 60 false + 38 true (binary labels)
- Classes: fa (fake audio), fv (fake video), fc (fake context), ft (fake text)

Configs A-G: Run on Mac with Gemini API
Config H: Prepare for PC with Gemma-4 local

Usage:
  # Run Config F on Mac (recommended)
  export GEMINI_API_KEY="your-key"
  python dyplom_v2/fmnv_pipeline_eval.py --config F

  # Run all available configs
  python dyplom_v2/fmnv_pipeline_eval.py --all

  # Compare results
  python dyplom_v2/fmnv_pipeline_eval.py --compare
"""

import argparse
import json
import os
import random
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", message="FP16 is not supported on CPU")
warnings.filterwarnings("ignore", message="'pin_memory' argument is set as true")

import cv2
import numpy as np
import torch
import clip
import whisper
import easyocr
from google import genai
from PIL import Image
from scenedetect import SceneManager, open_video
from scenedetect.detectors import ContentDetector, AdaptiveDetector

# ── Configuration ──────────────────────────────────────────────────────────────
FMNV_DIR    = Path("FMNV")
DATA_FILE   = FMNV_DIR / "data.json"
VIDEO_DIR   = FMNV_DIR / "videos"
RESULTS_DIR = Path("ablation_results_fmnv")
RANDOM_SEED = 42
MODEL_NAME  = "gemini-2.5-flash"
MAX_SCENES  = 4

CONFIGS = {
    "A": "text_only",       # OCR + Whisper, no images
    "B": "vision",          # OCR + Whisper + keyframes
    "C": "vision_clip",     # OCR + Whisper + keyframes + CLIP
    "D": "vision_lite",     # same as B — gemini-2.5-flash-lite
    "E": "vision_g3flash",  # same as B — gemini-3-flash-preview
    "F": "vision_g31lite",  # same as B — gemini-3.1-flash-lite (RECOMMENDED)
    "G": "vision_25pro",    # same as B — gemini-2.5-pro
}

CONFIG_MODELS = {
    "A": "gemini-2.5-flash",
    "B": "gemini-2.5-flash",
    "C": "gemini-2.5-flash",
    "D": "gemini-2.5-flash-lite",
    "E": "gemini-3-flash-preview",
    "F": "gemini-3.1-flash-lite-preview",
    "G": "gemini-2.5-pro",
}

# Device detection
if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

print(f"Device: {DEVICE}")

# ── Lazy model loading ─────────────────────────────────────────────────────────
_models = {}

def get_gemini():
    if "gemini" not in _models:
        _models["gemini"] = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    return _models["gemini"]

def get_ocr():
    if "ocr" not in _models:
        print("Loading EasyOCR...")
        _models["ocr"] = easyocr.Reader(["en", "uk", "ru"], gpu=(DEVICE in ("cuda", "mps")), verbose=False)
    return _models["ocr"]

def get_whisper():
    if "whisper" not in _models:
        print("Loading Whisper...")
        _models["whisper"] = whisper.load_model("base", device="cpu")
    return _models["whisper"]

def get_clip():
    if "clip" not in _models:
        print("Loading CLIP...")
        _models["clip_model"], _models["clip_processor"] = clip.load("ViT-B/32", device=DEVICE)
    return _models["clip_model"], _models["clip_processor"]

# ── Scene detection ────────────────────────────────────────────────────────────
def detect_scenes(video_path):
    """Cascading scene detector — same as FakeTT pipeline."""
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

    # Fallback: uniform 10-second segmentation
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
    """Extract keyframe images."""
    keyframes = []
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25

    for scene_start, scene_end in scenes[:MAX_SCENES]:
        frame_idx = int((scene_start + (scene_end - scene_start) / 2) * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            keyframes.append(Image.fromarray(rgb))

    cap.release()
    return keyframes

def extract_clip_scores(video_path, scenes, ocr_text):
    """Extract CLIP visual-text alignment scores."""
    if not ocr_text.strip():
        return []

    try:
        model, processor = get_clip()
        keyframes = extract_keyframes(video_path, scenes)

        scores = []
        with torch.no_grad():
            text_features = model.encode_text(clip.tokenize(ocr_text).to(DEVICE))
            for kf in keyframes:
                kf_tensor = processor(kf).unsqueeze(0).to(DEVICE)
                image_features = model.encode_image(kf_tensor)
                similarity = (image_features @ text_features.T).cpu().numpy()[0][0]
                scores.append(float(similarity))

        return scores
    except Exception:
        return []

# ── Gemini inference ───────────────────────────────────────────────────────────
def gemini_evaluate(video_path, ocr_text, audio_text, clip_scores, config, keyframes=None):
    """Send to Gemini for narrative verification."""
    client = get_gemini()
    model = CONFIG_MODELS[config]

    # Build prompt
    if config == "A":
        # Text-only
        prompt = f"""You are a video forensics analyst. Analyze the following video metadata and determine if the narrative is consistent or manipulated.

OCR Text Overlays:
{ocr_text if ocr_text else "(no text found)"}

Audio Transcript:
{audio_text if audio_text else "(no audio found)"}

Based on ONLY the text and audio (no visual evidence), assess narrative consistency on a scale 0-100:
- 0-30: Highly consistent (likely authentic)
- 30-70: Mixed/ambiguous
- 70-100: Highly inconsistent (likely manipulated)

Respond with ONLY a JSON object:
{{"score": <number 0-100>, "reasoning": "<brief explanation>"}}"""
    else:
        # Vision-enabled
        parts = []

        # Add keyframes first (to exploit position bias)
        if keyframes:
            for i, kf in enumerate(keyframes):
                parts.append(kf)
                parts.append(f"[Keyframe {i+1}]")

        # Add text prompt
        clip_info = ""
        if clip_scores:
            avg_clip = np.mean(clip_scores)
            clip_info = f"CLIP visual-text alignment score: {avg_clip:.3f} (0=misaligned, 1=perfectly aligned)\n"

        prompt = f"""You are a video forensics analyst. Analyze this video's scenes and determine if the narrative is consistent or manipulated.

{clip_info}
OCR Text Overlays:
{ocr_text if ocr_text else "(no text found)"}

Audio Transcript:
{audio_text if audio_text else "(no audio found)"}

Assess narrative consistency on a scale 0-100:
- 0-30: Highly consistent (likely authentic)
- 30-70: Mixed/ambiguous
- 70-100: Highly inconsistent (likely manipulated)

If no visual evidence is available, assign score ≤ 25.

Respond with ONLY a JSON object:
{{"score": <number 0-100>, "reasoning": "<brief explanation>"}}"""

        parts.append(prompt)

    try:
        if config == "A":
            response = client.models.generate_content(model=model, contents=prompt)
        else:
            response = client.models.generate_content(model=model, contents=parts)

        text = response.text.strip()
        # Extract JSON
        if "{" in text and "}" in text:
            start = text.index("{")
            end = text.rindex("}") + 1
            json_str = text[start:end]
            result = json.loads(json_str)
            return result.get("score", 50), result.get("reasoning", "")
        else:
            return 50, "Could not parse response"
    except Exception as e:
        return 50, f"Error: {str(e)}"

# ── Main evaluation ────────────────────────────────────────────────────────────
def evaluate_config(config):
    """Run evaluation for a single config."""
    print(f"\n🎬 RUNNING CONFIG {config} ({CONFIG_MODELS[config]})")

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

        # Skip if already processed
        if i > 0 and i % 10 == 0:
            print(f"   Progress: {i}/{len(available)}")

        try:
            # Extract modalities
            scenes = detect_scenes(video_path)
            ocr_text = extract_ocr(video_path, scenes)
            audio_text = extract_whisper(video_path)
            clip_scores = extract_clip_scores(video_path, scenes, ocr_text) if config != "A" else []
            keyframes = extract_keyframes(video_path, scenes) if config != "A" else None

            # Get prediction
            pred_score, reasoning = gemini_evaluate(video_path, ocr_text, audio_text, clip_scores, config, keyframes)
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
                "clip_score": np.mean(clip_scores) if clip_scores else None,
                "reasoning": reasoning[:100]  # Truncate for storage
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
    output_file = RESULTS_DIR / f"fmnv_{config}.json"
    with open(output_file, 'w') as f:
        json.dump({"config": config, "model": CONFIG_MODELS[config], "metrics": metrics, "results": results}, f, indent=2)

    print(f"   Results saved to {output_file}")
    if metrics:
        print(f"   F1: {metrics['f1']:.3f} | Acc: {metrics['accuracy']:.3f} | Prec: {metrics['precision']:.3f} | Rec: {metrics['recall']:.3f}")

    return metrics, results

# ── CLI ────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", choices=list(CONFIGS.keys()), help="Single config to run")
    parser.add_argument("--all", action="store_true", help="Run all available configs")
    parser.add_argument("--compare", action="store_true", help="Print comparison table")
    args = parser.parse_args()

    if args.compare:
        # Load and print results
        print("\n" + "=" * 100)
        print("FMNV RESULTS COMPARISON (Gemini A-G)")
        print("=" * 100)
        print(f"{'Config':<8} {'Model':<30} {'Accuracy':<10} {'Precision':<12} {'Recall':<10} {'F1':<10} {'TP':<3} {'TN':<3} {'FP':<3} {'FN':<3}")
        print("-" * 100)

        for config in ["A", "B", "C", "D", "E", "F", "G"]:
            result_file = RESULTS_DIR / f"fmnv_{config}.json"
            if result_file.exists():
                with open(result_file) as f:
                    data = json.load(f)
                m = data.get("metrics", {})
                if m:
                    print(f"{config:<8} {CONFIG_MODELS[config]:<30} {m.get('accuracy', 0):<10.3f} {m.get('precision', 0):<12.3f} {m.get('recall', 0):<10.3f} {m.get('f1', 0):<10.3f} {m.get('tp', 0):<3} {m.get('tn', 0):<3} {m.get('fp', 0):<3} {m.get('fn', 0):<3}")
    elif args.all:
        for config in ["A", "B", "C", "D", "E", "F", "G"]:
            evaluate_config(config)
    elif args.config:
        evaluate_config(args.config)
    else:
        print("Usage:")
        print("  python fmnv_pipeline_eval.py --config F     # Run single config")
        print("  python fmnv_pipeline_eval.py --all          # Run all A-G")
        print("  python fmnv_pipeline_eval.py --compare      # Compare results")

if __name__ == "__main__":
    main()
