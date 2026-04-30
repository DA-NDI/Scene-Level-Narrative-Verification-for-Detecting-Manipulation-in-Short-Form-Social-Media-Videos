import cv2
import whisper
import easyocr
import torch
from google import genai
import os

# Initialize the new Gemini Client
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

def load_narrative_models(device):
    print("Loading Whisper (Audio) and EasyOCR (Text)...")
    # Whisper 'base' is fast and highly accurate for English/Russian/Ukrainian
    audio_model = whisper.load_model("base").to(device)
    
    # EasyOCR automatically targets the GPU if available
    reader = easyocr.Reader(['en', 'uk', 'ru'], gpu=(device.type != 'cpu'))
    
    return audio_model, reader

def analyze_narrative_consistency(video_path, scene_list):
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Executing narrative extraction on: {device}")
    
    audio_model, ocr_reader = load_narrative_models(device)

    print("\nStage 2: Extracting Audio Context...")
    transcription = audio_model.transcribe(video_path)
    audio_text = transcription['text'].strip()
    print(f"Audio Transcript: '{audio_text[:100]}...'")

    print("\nStage 3: Extracting On-Screen Text (OCR Keyframes)...")
    cap = cv2.VideoCapture(video_path)
    ocr_results = []

    for i, scene in enumerate(scene_list):
        # Grab the middle frame of each narrative unit
        start_frame = scene[0].get_frames() if scene[0] else 0
        end_frame = scene[1].get_frames() if scene[1] else int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        mid_frame_idx = start_frame + ((end_frame - start_frame) // 2)
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, mid_frame_idx)
        ret, frame = cap.read()
        
        if ret:
            # Read text from the frame
            text_detections = ocr_reader.readtext(frame, detail=0) 
            if text_detections:
                scene_text = " ".join(text_detections)
                ocr_results.append(f"Scene {i+1} Text: {scene_text}")

    cap.release()
    combined_ocr = "\n".join(ocr_results)

    print("\nStage 4: Gemini LLM Narrative Scoring...")
    prompt = f"""
    You are a forensic video analyst evaluating a short-form social media video for narrative manipulation (cheapfakes/recontextualization).
    
    Video Audio Transcript:
    "{audio_text}"
    
    On-Screen Text per Scene:
    {combined_ocr if combined_ocr else "No on-screen text detected."}
    
    Task:
    Analyze the semantic consistency between the Audio Transcript and the On-Screen Text, keeping in mind real-world geopolitical context. 
    Does the audio match the context of the text, or is there evidence of misleading recontextualization, satire taken out of context, or narrative mismatch?
    
    Provide:
    1. A consistency score from 0.0 to 1.0 (where 1.0 means highly manipulated/inconsistent/satirical, and 0.0 means perfectly coherent).
    2. A brief 2-3 sentence justification for your score.
    """

    # Generate the verdict using Gemini 1.5 Flash
    response = client.models.generate_content(
        model='gemini-3.1-flash-lite-preview',
        contents=prompt
    )
    
    print("\n=== Gemini Forensic Verdict ===")
    print(response.text)

if __name__ == "__main__":
    # Point this to the EXACT same Trump Gaza video you ran Stream A on
    test_video = "./trump_gaza.mp4"
    
    # For this standalone test, we treat the video as one continuous scene
    # In the final pipeline, you will pass the 32 scenes from PySceneDetect here
    mock_scene_list = [(None, None)] 
    analyze_narrative_consistency(test_video, mock_scene_list)