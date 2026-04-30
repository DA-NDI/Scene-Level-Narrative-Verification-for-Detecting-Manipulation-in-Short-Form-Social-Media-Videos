import os
import google.generativeai as genai

genai.configure(api_key=os.environ["GEMINI_API_KEY"])
model = genai.GenerativeModel('gemini-3-flash-preview')

# Дані з ваших попередніх тестів на M3 Max
visual_context = "Люди розважаються на пляжі, сонячна погода, хвойні дерева на фоні, прісна вода (Київське море)."
ocr_text = "КИЯНИ РОЗВАЖАЮТЬСЯ HA ОСТРОВІ ЕПШТЕЙНА ПіСЛЯ НІЧНИХ ОБСТРІЛІВ"
clip_score = 0.2632

prompt = f"""
Аналіз відео-маніпуляції (Cheapfake):
- Візуальний контекст: {visual_context}
- Текст на відео (OCR): {ocr_text}
- Бал відповідності CLIP: {clip_score}

ПИТАННЯ: Чи є цей контент маніпулятивним? Поясни, чому CLIP дав високий бал, але текст все одно є брехнею.
"""

response = model.generate_content(prompt)
print(response.text)