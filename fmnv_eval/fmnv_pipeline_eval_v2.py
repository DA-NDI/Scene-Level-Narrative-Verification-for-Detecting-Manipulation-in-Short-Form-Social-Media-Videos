"""
fmnv_pipeline_eval_v2.py — FMNV evaluation with rich real/fake comparison output
Refactored to match 3090/gemma4_ablation_2_images.py style: better output, error recovery, config.

FMNV: Fake Multimodal News Video dataset
- 98 videos locally available
- 60 fake + 38 real
- Classes: fa (fake audio), fv (fake video), fc (fake context), ft (fake text)

Configs A-G: Gemini API models

Usage:
  export GEMINI_API_KEY="your-key"
  python fmnv_eval/fmnv_pipeline_eval_v2.py --config F
  python fmnv_eval/fmnv_pipeline_eval_v2.py --compare
"""

import argparse
import base64
import json
import os
import random
import re
import string
import time
import warnings
from io import BytesIO
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

# ── EXPERIMENT CONFIGURATION ──────────────────────────────────────────────────
# Change these to test different setups — results auto-save with new filenames
CONFIG_LABEL = "F"  # Which config to run (A-G)
MODEL_NAME   = "gemini-3.1-flash-lite-preview"
MAX_SCENES   = 4
THRESHOLD    = 50  # Classification threshold

# Auto-generated description for comparison table
CONFIG_DESC  = f"{MODEL_NAME} (max_scenes={MAX_SCENES}, τ={THRESHOLD})"
# ──────────────────────────────────────────────────────────────────────────────

# ── Paths ──
SCRIPT_DIR    = Path(__file__).resolve().parent
ROOT_DIR      = SCRIPT_DIR.parent
FMNV_DIR      = ROOT_DIR / "FMNV"
DATA_FILE     = FMNV_DIR / "data.json"
VIDEO_DIR     = FMNV_DIR / "videos"
RESULTS_DIR   = ROOT_DIR / "ablation_results_fmnv"

# Device detection
if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

print(f"Device: {DEVICE}")

# Config mapping
CONFIG_MODELS = {
    "A": "gemini-2.5-flash",
    "B": "gemini-2.5-flash",
    "C": "gemini-2.5-flash",
    "D": "gemini-2.5-flash-lite",
    "E": "gemini-3-flash-preview",
    "F": "gemini-3.1-flash-lite-preview",
    "G": "gemini-2.5-pro",
}

CONFIG_MODES = {
    "A": "text_only",
    "B": "vision",
    "C": "vision_clip",
    "D": "vision",
    "E": "vision",
    "F": "vision",
    "G": "vision",
}

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

# ── Scene detection (same cascading strategy as FakeTT) ───────────────────────
def detect_scenes(video_path):
    """Cascading scene detector."""
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

# ── Feature extraction ─────────────────────────────────────────────────────────
def extract_frame(video_path, ts):
    try:
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(ts * fps))
        ret, frame = cap.read()
        cap.release()
        if not ret:
            return None, None
        pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        return frame, pil
    except Exception:
        return None, None

def pil_to_base64(pil_img):
    buf = BytesIO()
    pil_img.thumbnail((224, 224))
    pil_img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def run_ocr(frame):
    if frame is None:
        return ""
    try:
        texts = get_ocr().readtext(frame, detail=0)
        return " | ".join(texts) if texts else ""
    except Exception:
        return ""

def run_whisper(video_path):
    try:
        result = get_whisper().transcribe(str(video_path), fp16=False, language=None, task="transcribe")
        return result.get("text", "").strip()[:600]
    except Exception as e:
        return f"[ASR error]"

def run_clip(frame_pil):
    """Return CLIP visual-text alignment score (0-1)."""
    if frame_pil is None:
        return None
    try:
        model, processor = get_clip()
        image = processor(frame_pil).unsqueeze(0).to(DEVICE)

        # Text prompts for real/fake classification
        texts = ["This is authentic news footage", "This is manipulated or fake footage"]
        text_tokens = clip.tokenize(texts).to(DEVICE)

        with torch.no_grad():
            image_features = model.encode_image(image)
            text_features = model.encode_text(text_tokens)
            logits = (image_features @ text_features.T).softmax(dim=-1)

        # Return confidence in "authentic" class (higher = more authentic-looking)
        return float(logits[0, 0].item())
    except Exception:
        return None

# ── Prompt builder ─────────────────────────────────────────────────────────────
def build_prompt(entry, scenes, scene_data, transcript, n_images, use_clip=False):
    """Build structured prompt for Gemini."""
    scene_lines = ""
    for i, sd in enumerate(scene_data[:MAX_SCENES]):
        clip_info = f" [CLIP alignment: {sd['clip_score']:.2f}]" if sd.get('clip_score') is not None and use_clip else ""
        scene_lines += f"  Scene {i+1} ({sd['start']:.1f}s-{sd['end']:.1f}s): OCR='{sd['ocr']}'{clip_info}\n"

    if not scene_lines:
        scene_lines = "  (no scenes detected)\n"

    no_img_rule = ""
    if n_images == 0:
        no_img_rule = "CONSTRAINT: No images available. Score MUST be <= 25.\n"

    return f"""You are a video forensics analyst detecting narrative manipulation in multimodal news videos.
{n_images} keyframe image(s) are attached above this text.

STEP 1 — Examine the attached images carefully. Describe what you see in each frame.
STEP 2 — Read the metadata and audio transcript below.
STEP 3 — Look for cross-modal contradictions: visual content that conflicts with audio claims or text overlays.
{no_img_rule}
NOTE: Genuine news footage, conflict imagery, or real events alone DO NOT indicate manipulation.
Only specific contradictions between visual evidence and claimed narrative count as signals.

METADATA:
- Title: {entry.get('title', '')[:200]}
- Description: {entry.get('description', '')[:300]}
- Detected scenes: {len(scenes)} | Keyframes sent: {n_images}

Audio transcript (Whisper ASR):
  {transcript or 'N/A'}

OCR text per scene:
{scene_lines}
Return ONLY valid JSON (no markdown, no code blocks, no explanation):
{{
  "visual_descriptions": ["MAXIMUM 5 WORDS describing what you see in each image"],
  "reasoning": "MAXIMUM 2 SENTENCES explaining any cross-modal signals detected.",
  "contradiction_found": true or false,
  "primary_signal": "visual" | "audio" | "text" | "none",
  "score": <integer 0-100>,
  "label": "fake" or "real"
}}

Scoring guide: 0-30=authentic | 31-60=ambiguous | 61-100=manipulated
"""

# ── Gemini API call with robust error recovery ─────────────────────────────────
def call_gemini(prompt, pil_images, config="F", retries=3):
    """Call Gemini with images, robust JSON parsing and fallback."""
    client = get_gemini()

    for attempt in range(retries):
        try:
            # Build content with images first (position bias)
            content = []
            for pil in pil_images:
                if pil is not None:
                    b64 = pil_to_base64(pil)
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
                    })

            # Add text prompt
            content.append({"type": "text", "text": prompt})

            # Call Gemini
            response = client.models.generate_content(
                model=CONFIG_MODELS.get(config, "gemini-3.1-flash-lite-preview"),
                contents=content,
                generation_config={
                    "temperature": 0.1,
                    "top_p": 0.95,
                    "max_output_tokens": 512,
                }
            )

            text = response.text.strip() if response.text else ""

            # Try JSON parsing
            parsed = {}
            score, label, reasoning = None, None, ""

            try:
                # Strip markdown if present
                if "```json" in text:
                    text = text.split("```json", 1)[1].split("```", 1)[0].strip()
                elif "```" in text:
                    text = text.split("```", 1)[1].split("```", 1)[0].strip()

                # Parse JSON
                parsed = json.loads(text)
                score = int(parsed.get("score", 50))
                label = str(parsed.get("label", "fake")).strip().lower()
                reasoning = str(parsed.get("reasoning", ""))

            except Exception:
                # Fallback: aggressive regex parsing
                if text.strip():
                    s_match = re.search(r'"score"\s*:\s*(\d+)', text, re.IGNORECASE)
                    l_match = re.search(r'"label"\s*:\s*"?(real|fake)"?', text, re.IGNORECASE)
                    if s_match and l_match:
                        score = int(s_match.group(1))
                        label = l_match.group(1).lower()
                        reasoning = "[Salvaged from truncated output]"
                    else:
                        raise ValueError("Could not parse score/label from response")
                else:
                    raise ValueError("Empty response from Gemini")

            # Validate and constrain
            if score is not None:
                score = max(0, min(100, score))
            if label not in {"fake", "real"}:
                label = "fake" if score >= 50 else "real"

            return {
                "score": score, "label": label,
                "visual_descriptions": parsed.get("visual_descriptions", []),
                "contradiction_found": parsed.get("contradiction_found", False),
                "primary_signal": parsed.get("primary_signal", "unknown"),
                "reasoning": reasoning,
                "error": None,
            }

        except Exception as exc:
            print(f"    [Attempt {attempt+1}/{retries}] Gemini error: {str(exc)[:80]}")
            if attempt < retries - 1:
                time.sleep(2)
            else:
                return {
                    "score": None, "label": None,
                    "visual_descriptions": [],
                    "contradiction_found": None,
                    "primary_signal": None,
                    "reasoning": f"Error: {str(exc)[:100]}",
                    "error": str(exc),
                }

# ── Full pipeline for one video ────────────────────────────────────────────────
def run_video(video_path, metadata, config="F"):
    """Process single video: scenes → features → Gemini → result."""
    scenes = detect_scenes(video_path)
    scene_data = []
    pil_images = []

    for (start, end) in scenes[:MAX_SCENES]:
        mid = (start + end) / 2
        frame_bgr, frame_pil = extract_frame(video_path, mid)
        ocr = run_ocr(frame_bgr)
        clip_score = run_clip(frame_pil) if CONFIG_MODES.get(config) == "vision_clip" else None

        scene_data.append({
            "start": start, "end": end, "ocr": ocr, "pil": frame_pil,
            "clip_score": clip_score
        })
        pil_images.append(frame_pil)

    transcript = run_whisper(str(video_path))
    prompt = build_prompt(
        metadata, scenes, scene_data, transcript, len(pil_images),
        use_clip=(CONFIG_MODES.get(config) == "vision_clip")
    )
    result = call_gemini(prompt, pil_images, config=config)

    return {
        "video_id": metadata.get("video_id", metadata.get("id", "")),
        "gold_label": metadata.get("label", metadata.get("annotation", "unknown")),
        "pred_label": result["label"],
        "pred_score": result["score"],
        "correct": result["label"] == metadata.get("label", metadata.get("annotation")),
        "scene_count": len(scenes),
        "image_count": len(pil_images),
        "transcript_len": len(transcript),
        "contradiction_found": result["contradiction_found"],
        "primary_signal": result["primary_signal"],
        "reasoning": result["reasoning"],
        "error": result["error"],
        "model": CONFIG_MODELS.get(config, MODEL_NAME),
        "config": CONFIG_LABEL,
        "desc": CONFIG_DESC
    }

# ── Metrics calculation ────────────────────────────────────────────────────────
def metrics(records):
    """Compute TP/TN/FP/FN and derived metrics."""
    valid = [r for r in records if r.get("pred_score") is not None]
    if not valid:
        return {}

    tp = sum(1 for r in valid if r["gold_label"] == "fake" and r["pred_label"] == "fake")
    tn = sum(1 for r in valid if r["gold_label"] == "real" and r["pred_label"] == "real")
    fp = sum(1 for r in valid if r["gold_label"] == "real" and r["pred_label"] == "fake")
    fn = sum(1 for r in valid if r["gold_label"] == "fake" and r["pred_label"] == "real")

    pr = tp / (tp + fp) if (tp + fp) > 0 else 0
    re = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * pr * re / (pr + re) if (pr + re) > 0 else 0

    return {
        "n": len(valid), "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "accuracy": (tp + tn) / len(valid),
        "precision": pr, "recall": re, "f1": f1
    }

# ── Main FMNV run ─────────────────────────────────────────────────────────────
def run_fmnv(config="F"):
    """Run evaluation on FMNV dataset."""
    RESULTS_DIR.mkdir(exist_ok=True)
    out_file = RESULTS_DIR / f"fmnv_{config}.json"

    # Load FMNV data
    rows = []
    try:
        with DATA_FILE.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    except Exception as e:
        print(f"ERROR: Cannot load FMNV data: {e}")
        return

    # Filter to available videos with labels
    eligible = [
        r for r in rows
        if r.get("label") in {"fake", "real"}
        and (VIDEO_DIR / f"{r.get('video_id', '')}.mp4").exists()
    ]

    print(f"Found {len(eligible)} labeled FMNV videos with files")

    # Resume from existing results
    results, done_ids = [], set()
    if out_file.exists():
        try:
            saved = json.load(out_file.open())
            results = saved.get("results", [])
            done_ids = {r["video_id"] for r in results}
            print(f"Resuming: {len(done_ids)} already done\n")
        except Exception:
            print("Could not resume; starting fresh\n")

    # Color codes
    GREEN, RED, RESET = "\033[32m", "\033[31m", "\033[0m"

    # Process each video
    for idx, row in enumerate(eligible, 1):
        vid_id = row["video_id"]
        if vid_id in done_ids:
            continue

        t0 = time.time()
        result = run_video(VIDEO_DIR / f"{vid_id}.mp4", row, config=config)
        t_elapsed = time.time() - t0

        # Update metrics
        m = metrics(results + [result])
        color = GREEN if result.get("correct") else RED
        mark = "✓" if result.get("correct") else "✗"

        # Rich output with real/fake comparison
        print(
            f"{color}[{idx:>3}/{len(eligible)}] {mark} "
            f"gold={row['label']:<4} pred={result['pred_label'] or '?':<4} "
            f"score={str(result['pred_score'] or '?'):>3} | "
            f"sc={result['scene_count']:>2} imgs={result['image_count']} "
            f"({t_elapsed:.1f}s){RESET}"
        )

        # Show reasoning if available
        if result["reasoning"] and "Error" not in str(result["reasoning"]):
            print(f"         → {result['reasoning'][:120]}")

        # Show running metrics
        if m:
            print(
                f"         [{m['n']:>3}] "
                f"Acc={m['accuracy']:.3f} P={m['precision']:.3f} "
                f"R={m['recall']:.3f} F1={m['f1']:.3f} | "
                f"TP={m['tp']} TN={m['tn']} FP={m['fp']} FN={m['fn']}"
            )

        results.append(result)
        done_ids.add(vid_id)

        # Save incrementally
        with out_file.open("w", encoding="utf-8") as f:
            json.dump({
                "config": config,
                "model": CONFIG_MODELS.get(config, MODEL_NAME),
                "desc": CONFIG_DESC,
                "metrics": metrics(results),
                "results": results
            }, f, indent=2, ensure_ascii=False)

        time.sleep(0.5)

    # Final summary
    final = metrics(results)
    print(
        f"\n{'='*90}\n"
        f"Config {config} ({CONFIG_MODELS.get(config, MODEL_NAME)}) FINAL RESULTS:\n"
        f"Accuracy:  {final['accuracy']:.3f}\n"
        f"Precision: {final['precision']:.3f}\n"
        f"Recall:    {final['recall']:.3f}\n"
        f"F1:        {final['f1']:.3f}\n"
        f"TP={final['tp']} TN={final['tn']} FP={final['fp']} FN={final['fn']}\n"
        f"{'='*90}\n"
    )

# ── Comparison table ──────────────────────────────────────────────────────────
def compare():
    """Print comparison table of all config results."""
    print(f"\n{'═'*95}")
    print("FMNV EVALUATION — Gemini Model Configurations")
    print(f"{'═'*95}")
    print(f"{'Cfg':<4} {'Model':<45} {'Acc':>6} {'Prec':>6} {'Rec':>6} {'F1':>6} {'TP':>3} {'TN':>3} {'FP':>3} {'FN':>3}")
    print(f"{'─'*95}")

    labels = {
        "A": "gemini-2.5-flash (text only)",
        "B": "gemini-2.5-flash (vision)",
        "C": "gemini-2.5-flash (vision + CLIP)",
        "D": "gemini-2.5-flash-lite (vision)",
        "E": "gemini-3-flash-preview (vision)",
        "F": "gemini-3.1-flash-lite-preview (vision) ★",
        "G": "gemini-2.5-pro (vision)",
    }

    for key in string.ascii_uppercase[:7]:
        f = RESULTS_DIR / f"fmnv_{key}.json"

        if not f.exists():
            if key in labels:
                print(f"  {key:<3} {labels[key]:<45} {'[Not yet run]':>35}")
            continue

        try:
            data = json.load(f.open())
            m = data.get("metrics", {})
            if not m:
                continue

            marker = " ←" if key == CONFIG_LABEL else ""
            print(
                f"  {key:<3} {labels.get(key, 'Unknown'):<45} "
                f"{m['accuracy']:>6.3f} {m['precision']:>6.3f} "
                f"{m['recall']:>6.3f} {m['f1']:>6.3f} "
                f"{m['tp']:>3} {m['tn']:>3} {m['fp']:>3} {m['fn']:>3}{marker}"
            )
        except Exception as e:
            print(f"  {key:<3} [Error reading results: {e}]")

    print(f"{'═'*95}\n")

# ── CLI ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FMNV evaluation with Gemini")
    parser.add_argument("--config", choices=list("ABCDEFG"), default="F",
                        help="Which config to run (default: F)")
    parser.add_argument("--compare", action="store_true",
                        help="Print comparison table of all saved configs")
    args = parser.parse_args()

    if args.compare:
        compare()
    else:
        CONFIG_LABEL = args.config
        MODEL_NAME = CONFIG_MODELS.get(CONFIG_LABEL, "gemini-3.1-flash-lite-preview")
        CONFIG_DESC = f"{MODEL_NAME} (max_scenes={MAX_SCENES}, τ={THRESHOLD})"
        run_fmnv(config=CONFIG_LABEL)
