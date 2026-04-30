# Running Qwen2.5-VL on FakeTT (100 videos)

## Setup

### Option 1: Qwen API (Cloud - easiest)
```bash
# Install dependencies
pip install openai qwen-vl

# Set API key
export QWEN_API_KEY="your-qwen-api-key"

# Run
python fakett_qwen_eval.py
```

### Option 2: Local Inference (requires GPU)
```bash
# Install dependencies
pip install torch torchvision transformers qwen-vl

# Download model (auto on first run)
# Requires: 40GB+ VRAM recommended for Qwen2.5-VL

# Run
python fakett_qwen_eval.py
```

## What the script does

1. **Loads 100 balanced FakeTT videos:**
   - 50 real videos
   - 50 fake videos
   - Random seed: 42 (reproducible)

2. **For each video:**
   - Detects scenes (corrected threshold: 3.0)
   - Extracts OCR text from keyframes
   - Sends 3-4 keyframes to Qwen2.5-VL
   - Gets manipulation score (0-100) + label (fake/real)

3. **Outputs metrics:**
   - Accuracy, Precision, Recall, F1
   - Saves detailed results to: `fakett_qwen_results.json`

## Expected runtime

- **API (Cloud):** ~10-15 minutes (batch processing)
- **Local GPU:** ~5-10 minutes (faster inference)

## Comparing with your thesis results

Your Gemini Config F results on FakeTT:
```
Accuracy:  0.600
Precision: 0.558
Recall:    0.960
F1:        0.706
```

Compare Qwen results against this baseline.

## Troubleshooting

**"QWEN_API_KEY not set"**
```bash
export QWEN_API_KEY="sk-xxx..."
```

**"ModuleNotFoundError: qwen_vl_utils"**
```bash
pip install --upgrade openai
```

**Out of memory (local inference)**
- Use cloud API instead
- Or reduce MAX_SCENES_FOR_ANALYSIS in script

## Next steps

1. Run: `python fakett_qwen_eval.py`
2. Compare metrics: Qwen vs Gemini from thesis
3. Save results to thesis appendix
4. Update Chapter 2 (Related Work) with Qwen results
