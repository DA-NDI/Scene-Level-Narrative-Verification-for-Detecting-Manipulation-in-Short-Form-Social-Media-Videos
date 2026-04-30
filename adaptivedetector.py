import torch
import os
from scenedetect import SceneManager, open_video
from scenedetect.detectors import AdaptiveDetector

# 1. Device Setup
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(f"✅ Using device: {device}")

def extract_narrative_segments(video_path):
    if not os.path.exists(video_path):
        print(f"❌ Error: Video not found at {video_path}")
        return []

    video = open_video(video_path)
    scene_manager = SceneManager()
    
    # Using your paper's specific threshold (cite: 413, 648)
    scene_manager.add_detector(AdaptiveDetector(adaptive_threshold=27.0))
    
    print(f"🚀 Processing: {os.path.basename(video_path)}...")
    scene_manager.detect_scenes(video)
    scene_list = scene_manager.get_scene_list()
    
    # Narrative Smoothing (cite: 649)
    filtered_scenes = [s for s in scene_list if (s[1] - s[0]).get_seconds() >= 0.5]
            
    print(f"🎬 Success: Detected {len(filtered_scenes)} narrative units.")
    for i, (start, end) in enumerate(filtered_scenes):
        print(f"  Unit {i+1}: {start.get_timecode()} -> {end.get_timecode()}")
    
    return filtered_scenes

# --- QUICK SEMINAR TEST BLOCK ---
if __name__ == "__main__":
    # Change 'test_video.mp4' to a real filename in your dyplom folder
    test_video = "./videos/4e51acc3-a58c-4e97-ae55-21083dc237a8.mp4" 
    extract_narrative_segments(test_video)