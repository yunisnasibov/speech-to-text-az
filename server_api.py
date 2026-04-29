import os
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

import torch
torch.set_num_threads(4)
import json
from google import genai
from fastapi import FastAPI, File, UploadFile, Header, HTTPException
from pyannote.audio import Pipeline
from faster_whisper import WhisperModel
import tempfile
import uvicorn

app = FastAPI(title="STT and Diarization Server")

# ═══════════════════════════════════════════════════════
# API KEY — bu açarı dəyişdirin və heç kimə verməyin!
# ═══════════════════════════════════════════════════════
API_KEY = os.getenv("STT_API_KEY", "stt-secret-key-2026")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# Modelləri server işə düşəndə bir dəfə yaddaşa yükləyirik
print("PyAnnote modeli yüklənir...")
HF_TOKEN = os.getenv("HF_TOKEN")
diarization_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
# Əgər serverdə NVIDIA GPU varsa:
if torch.cuda.is_available():
    diarization_pipeline.to(torch.device("cuda"))
    print("PyAnnote GPU-da işləyir.")

print("Whisper Large-v3 yüklənir...")
whisper_model = WhisperModel(
    "large-v3",
    device="cuda" if torch.cuda.is_available() else "cpu",
    compute_type="float16" if torch.cuda.is_available() else "int8"
)
print("Bütün modellər hazırdır!")


def gemini_correct_text(transcript_segments):
    """Gemini API ilə Whisper-in səhv yazdığı sözləri düzəldir."""
    if not GEMINI_API_KEY:
        print("GEMINI_API_KEY yoxdur, düzəliş edilmir.")
        return transcript_segments

    full_text = "\n".join([
        f"{seg['speaker']} ({seg['start']}s-{seg['end']}s): {seg['text']}"
        for seg in transcript_segments
    ])

    prompt = f"""Bu Azərbaycan dilində sinif monitorinq videosunun transkripsiyasıdır.
Whisper modeli bəzi sözləri, xüsusilə şəxs adlarını səhv yaza bilər.

Sənin vəzifən:
1. Səhv yazılmış Azərbaycan sözlərini düzəlt
2. Şəxs adlarını və soyadlarını düzgün formaya gətir
3. Qrammatik səhvləri düzəlt
4. Mənasını dəyişmə, yalnız yazılışı düzəlt

CAVABI YALNIZ JSON formatında qaytar, başqa heç nə yazma:
{{
  "corrected": [
    {{"speaker": "...", "start": ..., "end": ..., "text": "düzəldilmiş mətn"}},
    ...
  ],
  "teacher_name": "Müəllimin tam adı (əgər tapılarsa, yoxsa null)"
}}

Transkripsiya:
{full_text}"""

    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt
        )
        raw = response.text.strip()
        print(f"Gemini cavabı: {raw[:300]}")

        # JSON bloku tap (```json ... ``` və ya düz JSON)
        if "```json" in raw:
            raw = raw.split("```json")[1].split("```")[0].strip()
        elif "```" in raw:
            raw = raw.split("```")[1].split("```")[0].strip()

        result = json.loads(raw)
        corrected = result.get("corrected", [])
        teacher_name = result.get("teacher_name")

        if corrected and len(corrected) == len(transcript_segments):
            # Düzəldilmiş mətnləri əvəzlə
            for i, seg in enumerate(corrected):
                transcript_segments[i]["text"] = seg.get("text", transcript_segments[i]["text"])
            print("Gemini düzəlişləri tətbiq edildi.")
        
        return transcript_segments, teacher_name

    except Exception as e:
        print(f"Gemini xətası: {e}")
        return transcript_segments, None


@app.post("/transcribe")
async def transcribe_audio(
    file: UploadFile = File(...),
    x_api_key: str = Header(None, alias="X-API-Key")
):
    # API key yoxlanışı
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Yanlış API açarı")

    # Faylı serverdə müvəqqəti yaddaşa yazırıq
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
        temp_audio.write(await file.read())
        temp_audio_path = temp_audio.name

    try:
        # 1. Danışanların ayrılması (Diarization)
        print("Səslər ayrılır...")
        diarize_output = diarization_pipeline(temp_audio_path)
        diarization = diarize_output.speaker_diarization
        
        segments = []
        current_speaker = None
        current_start = None
        current_end = None

        for turn, _, speaker in diarization.itertracks(yield_label=True):
            if speaker == current_speaker:
                current_end = turn.end
            else:
                if current_speaker is not None:
                    segments.append({"speaker": current_speaker, "start": current_start, "end": current_end})
                current_speaker = speaker
                current_start = turn.start
                current_end = turn.end
        if current_speaker is not None:
            segments.append({"speaker": current_speaker, "start": current_start, "end": current_end})

        # 2. Bütün səsin eyni anda mətnə çevrilməsi və sözlərin vaxtla tapılması
        print("Səs mətnə çevrilir (Whisper)...")
        segments_whisper, _ = whisper_model.transcribe(temp_audio_path, language="az", word_timestamps=True)
        
        words = []
        for segment in segments_whisper:
            for word in segment.words:
                words.append({"word": word.word, "start": word.start, "end": word.end})

        # 3. Whisper sözlərini PyAnnote-un qeyd etdiyi danışanlarla sinxronizasiya edirik
        final_transcript = []
        
        for speaker_seg in segments:
            speaker_text = ""
            for w in words:
                # Əgər söz bu danışanın vaxt aralığına düşürsə
                if w["start"] >= speaker_seg["start"] and w["start"] <= speaker_seg["end"]:
                    speaker_text += w["word"]
            
            if speaker_text.strip():
                final_transcript.append({
                    "speaker": "Müfəttiş" if speaker_seg["speaker"] == "SPEAKER_00" else "Müəllim",
                    "start": round(speaker_seg["start"], 1),
                    "end": round(speaker_seg["end"], 1),
                    "text": speaker_text.strip()
                })

        # 4. Gemini ilə sözləri düzəlt
        teacher_name = None
        if GEMINI_API_KEY:
            print("Gemini ilə düzəliş edilir...")
            final_transcript, teacher_name = gemini_correct_text(final_transcript)

        return {"status": "success", "data": final_transcript, "teacher_name": teacher_name}

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"status": "error", "message": str(e)}
    finally:
        os.remove(temp_audio_path)

if __name__ == "__main__":
    # Serveri 8000 portunda işə salır
    uvicorn.run(app, host="0.0.0.0", port=8000)
