import whisper
import clip
import torch
import easyocr
from moviepy import VideoFileClip
from PIL import Image

# Ініціалізація пристрою MPS
device = "mps" if torch.backends.mps.is_available() else "cpu"

# Завантаження моделей
whisper_model = whisper.load_model("base", device="cpu")
clip_model, preprocess = clip.load("ViT-B/32", device=device)
ocr_reader = easyocr.Reader(['en', 'uk']) # Підтримка англійської та української

def analyze_segment(video_path, start_time, end_time):
    # .subclipped замість .subclip
    video = VideoFileClip(video_path).subclipped(start_time, end_time)
    
    audio_path = "temp_audio.mp3"
    # MoviePy 2.0 автоматично обробляє аудіо
    video.audio.write_audiofile(audio_path, logger=None)
    
    # 2. Whisper: Транскрипція
    result = whisper_model.transcribe(audio_path)
    transcript = result['text']
    
    # 3. CLIP & OCR: Беремо середній кадр сегмента
    mid_time = (start_time + end_time) / 2
    frame_path = "temp_frame.jpg"
    video.save_frame(frame_path, t=mid_time)
    
    # EasyOCR: Текст на екрані
    ocr_text = ocr_reader.readtext(frame_path, detail=0)
    
    return {
        "transcript": transcript,
        "ocr": " ".join(ocr_text)
    }

# Приклад для вашого Unit 1 (0:00 -> 0:04.92)
data = analyze_segment("./videos_manipulated/ostriv.mp4", 0, 4.92)
print(f"Segment Data: {data}")
