import cv2
import torch
import numpy as np
from PIL import Image
from scenedetect import detect, ContentDetector
from facenet_pytorch import MTCNN
from transformers import pipeline

def load_deepfake_detector(device):
    print("Loading SOTA Deepfake model...")
    # Just swap the string here to whichever model you want to test
    detector = pipeline("image-classification", model="prithivMLmods/Deepfake-Detect-Siglip2", device=device)
    return detector

def analyze_video_artifacts(video_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Executing artifact detection on: {device}")

    # Force MTCNN to CPU to avoid the MPS pooling bug
    mtcnn_device = torch.device('cpu') if device.type == 'mps' else device
    mtcnn = MTCNN(keep_all=False, device=mtcnn_device) 
    
    # Load the trained deepfake model
    artifact_detector = load_deepfake_detector(device)

    print("Stage 1: Running PySceneDetect...")
    scene_list = detect(video_path, ContentDetector(threshold=27.0))
    
    if not scene_list:
        scene_list = [(None, None)]

    cap = cv2.VideoCapture(video_path)
    video_verdict = []

    for i, scene in enumerate(scene_list):
        start_frame = scene[0].get_frames() if scene[0] else 0
        end_frame = scene[1].get_frames() if scene[1] else int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        mid_frame_idx = start_frame + ((end_frame - start_frame) // 2)
        cap.set(cv2.CAP_PROP_POS_FRAMES, mid_frame_idx)
        ret, frame = cap.read()
        
        if not ret:
            continue
            
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes, _ = mtcnn.detect(frame_rgb)
        
        if boxes is not None:
            box = [int(b) for b in boxes[0]]
            # Crop face array
            face_array = frame_rgb[max(0, box[1]):box[3], max(0, box[0]):box[2]]
            
            if face_array.size == 0:
                continue
                
            # Convert to PIL Image for the Hugging Face pipeline
            face_pil = Image.fromarray(face_array)
            
            # Run inference
            result = artifact_detector(face_pil)
            
            # TWEAK: Check for both 'deepfake' and 'fake' to support different model labels
            is_fake = result[0]['label'].lower() in ['deepfake', 'fake']
            score = result[0]['score'] if is_fake else 1.0 - result[0]['score']
                
            print(f"Scene {i+1} Artifact Score: {score:.4f} (Closer to 1 = Likely Manipulated)")
            video_verdict.append(score)
        else:
            print(f"Scene {i+1}: No faces detected for pixel-level analysis.")

    cap.release()
    
    if video_verdict:
        final_score = np.mean(video_verdict)
        print(f"\n=> Final Video Pixel Artifact Score: {final_score:.4f}")
        return final_score
    else:
        print("\n=> Could not calculate a score (no faces found in any scene).")
        return None

if __name__ == "__main__":
    test_video = "./trump_gaza.mp4"
    analyze_video_artifacts(test_video)


