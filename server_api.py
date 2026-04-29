import os
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

import re
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


def extract_teacher_name(transcript_segments):
    """Regex ilə müəllimin adını tapır — API key tələb etmir."""
    full_text = " ".join([seg["text"] for seg in transcript_segments])
    # Whisper artefaktlarını təmizlə (nöqtə sözün ortasında/sonunda)
    clean_text = re.sub(r'\.(?=\s+[A-ZƏÖÜŞÇĞİ])', '', full_text)  # "Yuniş. İbtar" → "Yuniş İbtar"
    clean_text = re.sub(r'\s+', ' ', clean_text).strip()
    
    print(f"Ad axtarışı mətni: {clean_text[:200]}")

    # ═══════════════ ÖNCƏLİKLİ: "mən keçirəm/edirəm Ad Soyad" ═══════════════
    # "mən keçirəm Yunis Nəsibov" — müəllimin özünü tanıtması ən güclü siqnaldır
    men_verb = re.search(
        r'm[eə]n\s+\w+\s+([A-ZƏÖÜŞÇĞİ][a-zəöüşçğıi]{2,})\s+([A-ZƏÖÜŞÇĞİ][a-zəöüşçğıi]{2,})\s*[.,]',
        clean_text
    )
    if men_verb:
        w1, w2 = men_verb.group(1), men_verb.group(2)
        stop = {'Daha','Sonra','Indi','Burada','Sinif','Dərs','Onun','Bizim','Sizin'}
        if w1 not in stop and w2 not in stop:
            print(f"Pattern 'mən + feil + Ad Soyad': {w1} {w2}")
            return f"{w1} {w2}"

    # "mən keçirəm Yunis Nəsibov" — cümlə sonunda nöqtə ilə
    men_verb_end = re.search(
        r'm[eə]n\s+\w+\s+([A-ZƏÖÜŞÇĞİ][a-zəöüşçğıi]{2,})\s+([A-ZƏÖÜŞÇĞİ][a-zəöüşçğıi]{2,})\s*$',
        clean_text
    )
    if men_verb_end:
        w1, w2 = men_verb_end.group(1), men_verb_end.group(2)
        stop = {'Daha','Sonra','Indi','Burada','Sinif','Dərs','Onun','Bizim','Sizin'}
        if w1 not in stop and w2 not in stop:
            print(f"Pattern 'mən + feil + Ad Soyad (cümlə sonu)': {w1} {w2}")
            return f"{w1} {w2}"

    # ═══════════════ Pattern 1: "Soyad Ad, Ata-adı oğlu/qızı" ═══════════════
    # Məsələn: "Abbasov Rəhman, Mihman oğlu"
    oglu = re.search(
        r'([A-ZƏÖÜŞÇĞİa-zəöüşçğıi]{3,})\s+([A-ZƏÖÜŞÇĞİa-zəöüşçğıi]{3,})\s*[,.]?\s*[A-ZƏÖÜŞÇĞİa-zəöüşçğıi]+\s+o[gğ]lu',
        clean_text, re.IGNORECASE
    )
    if oglu:
        return f"{oglu.group(2).title()} {oglu.group(1).title()}"

    qizi = re.search(
        r'([A-ZƏÖÜŞÇĞİa-zəöüşçğıi]{3,})\s+([A-ZƏÖÜŞÇĞİa-zəöüşçğıi]{3,})\s*[,.]?\s*[A-ZƏÖÜŞÇĞİa-zəöüşçğıi]+\s+q[iıI]z[iıI]',
        clean_text, re.IGNORECASE
    )
    if qizi:
        return f"{qizi.group(2).title()} {qizi.group(1).title()}"

    # ═══════════════ Pattern 2: "adım X deyil, adım Y" ═══════════════
    # Müəllim səhv deyilmiş adı düzəldir
    deyil = re.search(
        r'ad[iıIİ]m\s+.{2,30}?deyil.{1,20}?ad[iıIİ]m\s+([A-ZƏa-zə][a-zəöüşçğıi]+(?:\s+[A-ZƏa-zə][a-zəöüşçğıi]+))',
        clean_text, re.IGNORECASE
    )
    if deyil:
        return re.sub(r'[dD][iıİI][rR]$', '', deyil.group(1)).strip().title()

    # ═══════════════ Pattern 3: vergüldən sonra ad ═══════════════
    # "... müəllimiyəm, Rəhman Abbasov" və ya "... sinifdir, adım Rəhman"
    vergul = re.search(
        r'[,.]\s*ad[iıIİ]m\s+([A-ZƏa-zə][a-zəöüşçğıi]{2,}(?:\s+[A-ZƏa-zə][a-zəöüşçğıi]{2,})?)',
        clean_text, re.IGNORECASE
    )
    if vergul:
        name = re.sub(r'[dD][iıİI][rR]$', '', vergul.group(1)).strip()
        if len(name) > 3:
            return name.title()

    # ═══════════════ Pattern 4: "mən ... yam/yəm, Ad Soyad" ═══════════════
    self_ref = re.search(
        r'm[eə]n\s+.{1,40}?[yY][eəaı]m\s*[,.]\s*([A-ZƏa-zə][a-zəöüşçğıi]{2,}\s+[A-ZƏa-zə][a-zəöüşçğıi]{2,})',
        clean_text, re.IGNORECASE
    )
    if self_ref:
        return self_ref.group(1).strip().title()

    # ═══════════════ Pattern 5: "adım Ad Soyad" ═══════════════
    adim = re.search(
        r'ad[iıIİ]m\s+([A-ZƏa-zə][a-zəöüşçğıi]{2,}\s+[A-ZƏa-zə][a-zəöüşçğıi]{2,})',
        clean_text, re.IGNORECASE
    )
    if adim:
        name = re.sub(r'[dD][iıİI][rR]$', '', adim.group(1)).strip()
        # "adım X deyil" deyilsə
        pos = clean_text.find(adim.group(0))
        rest = clean_text[pos:pos+40].lower()
        if 'deyil' not in rest and len(name) > 3:
            return name.title()
    
    # ═══════════════ Pattern 6: "adım Ad" (tək ad) ═══════════════
    adim_tek = re.search(
        r'ad[iıIİ]m\s+([A-ZƏa-zə][a-zəöüşçğıi]{2,})',
        clean_text, re.IGNORECASE
    )
    if adim_tek:
        name = re.sub(r'[dD][iıİI][rR]$', '', adim_tek.group(1)).strip()
        pos = clean_text.find(adim_tek.group(0))
        rest = clean_text[pos:pos+30].lower()
        if 'deyil' not in rest and len(name) > 2:
            return name.title()

    # ═══════════════ Pattern 7: "Ad sinif müəllimi" ═══════════════
    # "Yuniş İbtar sinif müəllimi" → "Yuniş İbtar"
    muellim = re.search(
        r'([A-ZƏÖÜŞÇĞİa-zəöüşçğıi]{3,})\s+([A-ZƏÖÜŞÇĞİa-zəöüşçğıi]{3,})\s+(?:sinif\s+)?m[uü][eə]llim',
        clean_text, re.IGNORECASE
    )
    if muellim:
        w1 = re.sub(r'[.,;:!?]', '', muellim.group(1)).strip()
        w2 = re.sub(r'[.,;:!?]', '', muellim.group(2)).strip()
        stop = {'və', 'bu', 'da', 'bir', 'o', 'ki', 'dərs', 'sinif', 'salam', 'xeyir', 'səbəb', 'sonra', 'indi'}
        if w1.lower() not in stop and w2.lower() not in stop and len(w1) > 2 and len(w2) > 2:
            return f"{w1.title()} {w2.title()}"

    # Tək ad + müəllim
    muellim_tek = re.search(
        r'([A-ZƏÖÜŞÇĞİ][a-zəöüşçğıi]{2,})\s+(?:sinif\s+)?m[uü][eə]llim',
        clean_text, re.IGNORECASE
    )
    if muellim_tek:
        name = re.sub(r'[.,;:!?]', '', muellim_tek.group(1)).strip()
        stop = {'Və', 'Bu', 'Da', 'Bir', 'Dərs', 'Sinif', 'Salam', 'Xeyir', 'Səbəb', 'Sonra', 'İndi', 'Mənim'}
        if name not in stop and len(name) > 2:
            return name.title()

    # ═══════════════ Pattern 8: İlk segmentlərdə Ad Soyad ═══════════════
    for seg in transcript_segments[:3]:
        seg_clean = re.sub(r'[.,;:!?]', '', seg["text"])
        name_match = re.search(
            r'([A-ZƏÖÜŞÇĞİ][a-zəöüşçğıi]{2,})\s+([A-ZƏÖÜŞÇĞİ][a-zəöüşçğıi]{2,})',
            seg_clean
        )
        if name_match:
            w1, w2 = name_match.group(1), name_match.group(2)
            stop = {'Salam','Xeyir','Sabah','Beli','Yaxsi','Bugun','Bize','Size','Ders',
                     'Sinif','Bilir','Olur','Edir','Gelir','Mənim','Sizin','Bizim','Onun'}
            if w1 not in stop and w2 not in stop:
                return f"{w1} {w2}"

    return None


def gemini_correct_text(transcript_segments, gemini_api_key):
    """Gemini API ilə Whisper-in səhv yazdığı sözləri düzəldir."""
    if not gemini_api_key:
        print("Gemini API açarı verilməyib, düzəliş edilmir.")
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
        client = genai.Client(api_key=gemini_api_key)
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
    x_api_key: str = Header(None, alias="X-API-Key"),
    x_gemini_key: str = Header(None, alias="X-Gemini-Key")
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

        # 4. Müəllim adını regex ilə tap (həmişə işləyir)
        teacher_name = extract_teacher_name(final_transcript)
        print(f"Regex ilə tapılan ad: {teacher_name}")

        # 5. Gemini ilə sözləri düzəlt + adı təkmilləşdir (yalnız API key varsa)
        if x_gemini_key:
            print("Gemini ilə düzəliş edilir...")
            final_transcript, gemini_name = gemini_correct_text(final_transcript, x_gemini_key)
            if gemini_name:
                teacher_name = gemini_name  # Gemini daha dəqiq ad tapır
                print(f"Gemini ilə düzəldilmiş ad: {teacher_name}")

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
