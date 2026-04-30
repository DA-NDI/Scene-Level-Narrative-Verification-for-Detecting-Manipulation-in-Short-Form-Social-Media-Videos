"""
Run narrative manipulation pipeline on FakeTT with Qwen2.5-VL model.
- 50 real + 50 fake videos (balanced sample)
- Pipeline: scene detection -> keyframe OCR -> Qwen2.5-VL scoring
- Output: fakett_qwen_results.json
"""

import json
import os
import random
import time
from io import BytesIO
from pathlib import Path

import cv2
import easyocr
from PIL import Image
from scenedetect import SceneManager, open_video
from scenedetect.detectors import AdaptiveDetector

# Qwen imports
try:
    from qwen_vl_utils import process_vision_info
    from qwenclient import Client
    QWEN_AVAILABLE = True
except ImportError:
    QWEN_AVAILABLE = False
    print("⚠ Qwen libraries not installed. Install with: pip install qwen-vl")


DATA_FILE = Path("fakett/FakeTT_DATA_OPENSOURCE/data.json")
VIDEO_DIR = Path("fakett/FakeTT_DATA_OPENSOURCE/video")
RESULTS_FILE = Path("fakett_qwen_results.json")

N_REAL = 50
N_FAKE = 50
RANDOM_SEED = 42
MAX_SCENES_FOR_ANALYSIS = 4


def load_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def run_scene_detection(video_path: Path):
    try:
        video = open_video(str(video_path))
        scene_manager = SceneManager()
        scene_manager.add_detector(AdaptiveDetector(adaptive_threshold=3.0))  # CORRECTED threshold
        scene_manager.detect_scenes(video)
        scenes = scene_manager.get_scene_list()
        scenes = [s for s in scenes if (s[1] - s[0]).get_seconds() >= 0.5]
        return [(s[0].get_seconds(), s[1].get_seconds()) for s in scenes]
    except Exception as e:
        print(f"Scene detection failed: {e}")
        return []


def extract_frame_and_ocr(video_path: Path, timestamp_sec: float):
    try:
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(timestamp_sec * fps))
        ret, frame = cap.read()
        cap.release()
        if not ret:
            return None, ""

        ocr_reader = easyocr.Reader(["en", "uk", "ru"], gpu=False, verbose=False)
        ocr_texts = ocr_reader.readtext(frame, detail=0)
        ocr_text = " | ".join(ocr_texts) if ocr_texts else ""

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb)
        return pil_image, ocr_text
    except Exception as e:
        print(f"Frame extraction failed: {e}")
        return None, ""


def build_prompt(entry, scenes, ocr_texts):
    description = entry.get("description", "")
    user_description = entry.get("user_description", "")
    event = entry.get("event", "")

    scene_summary = ""
    for i, ((start_sec, end_sec), ocr_text) in enumerate(zip(scenes[:MAX_SCENES_FOR_ANALYSIS], ocr_texts[:MAX_SCENES_FOR_ANALYSIS])):
        scene_summary += f"  Scene {i + 1} ({start_sec:.1f}s-{end_sec:.1f}s): OCR='{ocr_text}'\n"
    if not scene_summary:
        scene_summary = "  (no scenes detected)\n"

    prompt = f"""You are a misinformation analyst specializing in short-form videos.

Assess whether this TikTok post is narratively manipulated (cheapfake-style recontextualization, misleading framing, false claim overlay, fabricated context) versus a normal authentic post.

POST DATA:
- Claimed event/topic: {event[:220]}
- Post description: {description[:450]}
- Account bio: {user_description[:220]}
- Scene count: {len(scenes)}
- OCR per scene:
{scene_summary}

You will also receive keyframe images from several scenes.

Return only valid JSON in this exact schema:
{{
  "score": <integer 0-100>,
  "label": "fake" or "real",
  "reasoning": "2-3 short sentences"
}}

Scoring guide:
- 0 to 30: likely real / low manipulation
- 31 to 69: uncertain or mixed signals
- 70 to 100: likely fake / strong manipulation signals
"""
    return prompt


def call_qwen_with_images(client, prompt, images, retries=3):
    """Call Qwen2.5-VL with images (local inference or API)."""
    for attempt in range(retries):
        try:
            # Prepare image content for Qwen
            content = [{"type": "text", "text": prompt}]
            for img in images:
                if img is not None:
                    # Convert PIL image to base64
                    import base64
                    from io import BytesIO
                    buffer = BytesIO()
                    img.save(buffer, format="PNG")
                    img_base64 = base64.b64encode(buffer.getvalue()).decode()
                    content.append({
                        "type": "image",
                        "image": f"data:image/png;base64,{img_base64}"
                    })
            
            # Call Qwen API
            response = client.chat.completions.create(
                model="qwen-vl-max",
                messages=[{"role": "user", "content": content}],
                temperature=0.3,
                top_p=0.8,
            )
            
            text = response.choices[0].message.content.strip()
            
            # Parse JSON response
            if "```json" in text:
                text = text.split("```json", 1)[1].split("```", 1)[0].strip()
            elif "```" in text:
                text = text.split("```", 1)[1].split("```", 1)[0].strip()
            
            parsed = json.loads(text)
            score = int(parsed["score"])
            label = str(parsed["label"]).strip().lower()
            reasoning = str(parsed.get("reasoning", ""))
            
            if label not in {"fake", "real"}:
                label = "fake" if score >= 50 else "real"
            
            score = max(0, min(100, score))
            return score, label, reasoning
            
        except Exception as exc:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                return None, None, f"Error: {exc}"


def binary_metrics(records):
    tp = sum(1 for r in records if r["gold_label"] == "fake" and r["pred_label"] == "fake")
    tn = sum(1 for r in records if r["gold_label"] == "real" and r["pred_label"] == "real")
    fp = sum(1 for r in records if r["gold_label"] == "real" and r["pred_label"] == "fake")
    fn = sum(1 for r in records if r["gold_label"] == "fake" and r["pred_label"] == "real")

    total = max(1, len(records))
    accuracy = (tp + tn) / total
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def main():
    if not QWEN_AVAILABLE:
        print("ERROR: Qwen libraries not available. Install with:")
        print("  pip install qwen-vl openai")
        return
    
    # Initialize Qwen client
    client = Client(api_key=os.environ.get("QWEN_API_KEY"))
    
    print("Loading FakeTT data...")
    rows = load_jsonl(DATA_FILE)

    eligible = []
    for row in rows:
        video_id = row.get("video_id")
        label = row.get("annotation")
        if not video_id or label not in {"fake", "real"}:
            continue
        video_path = VIDEO_DIR / f"{video_id}.mp4"
        if video_path.exists():
            eligible.append(row)

    real_pool = [r for r in eligible if r["annotation"] == "real"]
    fake_pool = [r for r in eligible if r["annotation"] == "fake"]

    if len(real_pool) < N_REAL or len(fake_pool) < N_FAKE:
        raise RuntimeError(f"Not enough data: real={len(real_pool)}, fake={len(fake_pool)}")

    random.seed(RANDOM_SEED)
    sample = random.sample(real_pool, N_REAL) + random.sample(fake_pool, N_FAKE)

    print(f"\n✓ Sample: {N_REAL} real + {N_FAKE} fake = {len(sample)} total videos\n")

    results = []
    for idx, entry in enumerate(sample, 1):
        video_id = entry["video_id"]
        gold_label = entry["annotation"]
        video_path = VIDEO_DIR / f"{video_id}.mp4"

        print(f"[{idx:3d}/{len(sample)}] {video_id}...", end=" ", flush=True)

        # Scene detection
        scenes = run_scene_detection(video_path)
        if not scenes:
            print("✗ (no scenes)")
            results.append({
                "video_id": video_id,
                "gold_label": gold_label,
                "pred_label": None,
                "score": None,
                "reasoning": "No scenes detected",
            })
            continue

        # Extract OCR + images
        ocr_texts = []
        images = []
        for start_sec, end_sec in scenes[:MAX_SCENES_FOR_ANALYSIS]:
            mid_sec = (start_sec + end_sec) / 2
            img, ocr_text = extract_frame_and_ocr(video_path, mid_sec)
            if img:
                images.append(img)
                ocr_texts.append(ocr_text)

        if not images:
            print("✗ (no frames)")
            results.append({
                "video_id": video_id,
                "gold_label": gold_label,
                "pred_label": None,
                "score": None,
                "reasoning": "No frames extracted",
            })
            continue

        # Call Qwen
        prompt = build_prompt(entry, scenes, ocr_texts)
        score, pred_label, reasoning = call_qwen_with_images(client, prompt, images)

        if pred_label is None:
            print(f"✗ (error: {reasoning})")
            results.append({
                "video_id": video_id,
                "gold_label": gold_label,
                "pred_label": None,
                "score": None,
                "reasoning": reasoning,
            })
        else:
            correct = "✓" if pred_label == gold_label else "✗"
            print(f"{correct} {pred_label:4s} (score={score:3d}, gold={gold_label})")
            results.append({
                "video_id": video_id,
                "gold_label": gold_label,
                "pred_label": pred_label,
                "score": score,
                "reasoning": reasoning,
            })

        time.sleep(1)  # Rate limiting

    # Compute metrics
    valid_results = [r for r in results if r["pred_label"] is not None]
    metrics = binary_metrics(valid_results)

    summary = {
        "model": "Qwen2.5-VL",
        "dataset": "FakeTT",
        "sample_size": len(sample),
        "valid_predictions": len(valid_results),
        "metrics": metrics,
        "results": results,
    }

    # Save results
    with RESULTS_FILE.open("w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Results saved to {RESULTS_FILE}")
    print(f"{'='*60}")
    print(f"Model: Qwen2.5-VL")
    print(f"Accuracy:  {metrics['accuracy']:.3f}")
    print(f"Precision: {metrics['precision']:.3f}")
    print(f"Recall:    {metrics['recall']:.3f}")
    print(f"F1:        {metrics['f1']:.3f}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
