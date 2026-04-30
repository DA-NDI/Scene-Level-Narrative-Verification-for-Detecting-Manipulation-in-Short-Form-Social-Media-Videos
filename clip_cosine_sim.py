import torch
import clip
from PIL import Image

# Use the MPS device on your M3 Max
device = "mps" if torch.backends.mps.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

def check_consistency(image_path, ocr_text):
    # 1. Preprocess the image and the OCR text
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    
    # Clip has a 77-token limit; we truncate OCR just in case
    text_tokens = clip.tokenize([ocr_text[:77]]).to(device)

    # 2. Extract features
    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text_tokens)

    # 3. Calculate Cosine Similarity (Result is between -1 and 1)
    # Higher score = More consistent
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    similarity = (image_features @ text_features.T).item()

    return similarity

# Example using your actual data
actual_ocr = "NV = КИЯНИ РОЗВАЖАЮТЬСЯ HA ОСТРОВІ ЕПШТЕЙНА ПіСЛЯ НІЧНИХ ОБСТРІЛІВ"
# image_path = "temp_frame.jpg" (from your previous script)

score = check_consistency("temp_frame.jpg", actual_ocr)
print(f"Consistency Score: {score:.4f}")

# Thresholding logic for your report
if score < 0.20:
    print("⚠️ Potential Narrative Manipulation: Low visual-textual alignment.")
else:
    print("✅ High alignment: Text matches visual context.")