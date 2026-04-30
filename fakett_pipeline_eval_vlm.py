"""
Run narrative manipulation pipeline on FakeTT with visual descriptions (VLM).
- 50 real + 50 fake videos (balanced sample)
- Pipeline: scene detection -> keyframe OCR + keyframe visual descriptions -> Gemini scoring
- Output: fakett_pipeline_results_vlm.json
"""

import json
import os
import random
import time
from io import BytesIO
from pathlib import Path

import cv2
import easyocr
import google.generativeai as genai
from PIL import Image
from scenedetect import SceneManager, open_video
from scenedetect.detectors import AdaptiveDetector


DATA_FILE = Path("fakett/FakeTT_DATA_OPENSOURCE/data.json")
VIDEO_DIR = Path("fakett/FakeTT_DATA_OPENSOURCE/video")
RESULTS_FILE = Path("fakett_pipeline_results_vlm.json")

N_REAL = 50
N_FAKE = 50
RANDOM_SEED = 42
MAX_SCENES_FOR_ANALYSIS = 4


genai.configure(api_key=os.environ["GEMINI_API_KEY"])
model = genai.GenerativeModel("gemini-2.5-flash")
ocr_reader = easyocr.Reader(["en", "uk", "ru"], gpu=False, verbose=False)


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
        scene_manager.add_detector(AdaptiveDetector(adaptive_threshold=27.0))
        scene_manager.detect_scenes(video)
        scenes = scene_manager.get_scene_list()
        scenes = [s for s in scenes if (s[1] - s[0]).get_seconds() >= 0.5]
        return [(s[0].get_seconds(), s[1].get_seconds()) for s in scenes]
    except Exception:
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

        ocr_texts = ocr_reader.readtext(frame, detail=0)
        ocr_text = " | ".join(ocr_texts) if ocr_texts else ""

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb)
        return pil_image, ocr_text
    except Exception:
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
  "visual_descriptions": ["short scene description 1", "..."],
  "reasoning": "2-3 short sentences"
}}

Scoring guide:
- 0 to 30: likely real / low manipulation
- 31 to 69: uncertain or mixed signals
- 70 to 100: likely fake / strong manipulation signals
"""
    return prompt


def call_gemini_with_images(prompt, images, retries=3):
    for attempt in range(retries):
        try:
            parts = [prompt]
            parts.extend(images)
            response = model.generate_content(parts)
            text = response.text.strip()

            if "```json" in text:
                text = text.split("```json", 1)[1].split("```", 1)[0].strip()
            elif "```" in text:
                text = text.split("```", 1)[1].split("```", 1)[0].strip()

            parsed = json.loads(text)
            score = int(parsed["score"])
            label = str(parsed["label"]).strip().lower()
            reasoning = str(parsed.get("reasoning", ""))
            visual_descriptions = parsed.get("visual_descriptions", [])
            if not isinstance(visual_descriptions, list):
                visual_descriptions = []

            if label not in {"fake", "real"}:
                label = "fake" if score >= 50 else "real"

            score = max(0, min(100, score))
            return score, label, reasoning, visual_descriptions
        except Exception as exc:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                return None, None, f"Error: {exc}", []


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
    random.shuffle(sample)

    print(f"Eligible videos: {len(eligible)}")
    print(f"Sample: {len(sample)} (real={N_REAL}, fake={N_FAKE})")

    if RESULTS_FILE.exists():
        with RESULTS_FILE.open("r", encoding="utf-8") as f:
            saved = json.load(f)
        if isinstance(saved, dict):
            results = saved.get("results", [])
        else:
            results = saved
        done_ids = {r["video_id"] for r in results}
        print(f"Resuming from existing results: {len(done_ids)} done")
    else:
        results = []
        done_ids = set()

    for idx, row in enumerate(sample, start=1):
        video_id = row["video_id"]
        if video_id in done_ids:
            continue

        video_path = VIDEO_DIR / f"{video_id}.mp4"
        gold = row["annotation"]

        scenes = run_scene_detection(video_path)

        key_images = []
        ocr_texts = []
        for (start_sec, end_sec) in scenes[:MAX_SCENES_FOR_ANALYSIS]:
            ts = (start_sec + end_sec) / 2
            image, ocr_text = extract_frame_and_ocr(video_path, ts)
            if image is not None:
                key_images.append(image)
            ocr_texts.append(ocr_text)

        prompt = build_prompt(row, scenes, ocr_texts)
        pred_score, pred_label, reasoning, visual_descriptions = call_gemini_with_images(prompt, key_images)

        if pred_label is None:
            pred_label = "fake" if (pred_score is not None and pred_score >= 50) else "real"

        print(
            f"[{idx:>3}/{len(sample)}] {video_id} | gold={gold} | "
            f"scenes={len(scenes)} images={len(key_images)} | pred={pred_label} score={pred_score}"
        )

        results.append(
            {
                "video_id": video_id,
                "gold_label": gold,
                "pred_label": pred_label,
                "pred_score": pred_score,
                "scene_count": len(scenes),
                "image_count": len(key_images),
                "event": row.get("event"),
                "description": row.get("description"),
                "reasoning": reasoning,
                "ocr_texts": ocr_texts,
                "visual_descriptions": visual_descriptions,
            }
        )
        done_ids.add(video_id)

        valid = [r for r in results if r.get("pred_score") is not None]
        metrics = binary_metrics(valid)

        out = {
            "config": {
                "dataset": str(DATA_FILE),
                "video_dir": str(VIDEO_DIR),
                "sample_real": N_REAL,
                "sample_fake": N_FAKE,
                "random_seed": RANDOM_SEED,
                "model": "gemini-2.5-flash",
                "max_scenes_for_analysis": MAX_SCENES_FOR_ANALYSIS,
                "uses_visual_descriptions": True,
            },
            "metrics_valid_only": metrics,
            "processed": len(results),
            "valid_predictions": len(valid),
            "results": results,
        }

        with RESULTS_FILE.open("w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)

        time.sleep(1.2)

    valid = [r for r in results if r.get("pred_score") is not None]
    metrics = binary_metrics(valid)

    print("\nFinal metrics (valid predictions):")
    print(
        "Accuracy={:.3f} Precision={:.3f} Recall={:.3f} F1={:.3f}".format(
            metrics["accuracy"], metrics["precision"], metrics["recall"], metrics["f1"]
        )
    )
    print("TP={tp} TN={tn} FP={fp} FN={fn}".format(**metrics))


if __name__ == "__main__":
    main()
