# 🎤 Azərbaycan Dilində Səsdən Mətnə (Speech-to-Text)

Sinif monitorinq videoları üçün Azərbaycan dilində səsdən mətnə çevirmə, danışanları ayırma və müəllim adını avtomatik tapma sistemi.

## İş Axını

1. 📁 Video/səs faylı yüklənir
2. 🎬 ffmpeg ilə səs çıxarılır (16kHz mono WAV)
3. 🧠 **PyAnnote 3.1** — danışanları ayırır (Müfəttiş / Müəllim)
4. 📝 **Whisper Large-v3** — səsi Azərbaycan dilində mətnə çevirir
5. 👩‍🏫 **Regex** — müəllimin adını avtomatik tapır (8 fərqli pattern)
6. ✨ **Gemini API** *(opsional)* — hərf səhvlərini düzəldir, adları təkmilləşdirir

## Xüsusiyyətlər

- **Whisper Large-v3** ilə Azərbaycan dilində yüksək dəqiqliklə səsdən mətnə çevirmə
- **PyAnnote 3.1** ilə danışanların ayrılması (diarization)
- **Müəllim adı avtomatik tapılır** — API key olmadan da regex ilə işləyir
- **Gemini API ilə düzəliş** — hərf səhvlərini, ad-soyadları düzəldir *(opsional)*
- **API Key** autentifikasiyası ilə təhlükəsiz giriş (`X-API-Key` header)
- **Streamlit** ilə sadə, 1 input-lu interfeys
- GPU dəstəyi (NVIDIA CUDA)

## Arxitektura

```
┌──────────────────┐         ┌──────────────────────────┐
│  Streamlit UI    │  HTTP   │    FastAPI Server         │
│  (app.py)        │────────▶│    (server_api.py)        │
│  Lokal Mac       │ Headers │    GPU Server             │
│                  │         │  ┌──────────────────────┐ │
│  📁 Video yüklə │ X-API-Key│  │ PyAnnote 3.1 (GPU)  │ │
│  🔑 Gemini key  │X-Gemini │  │ Whisper Large-v3     │ │
│                  │  -Key   │  │ Regex Ad Tapma       │ │
│                  │         │  │ Gemini API (opsional) │ │
└──────────────────┘         │  └──────────────────────┘ │
                             └──────────────────────────┘
```

## Quraşdırma

### Client (Lokal)
```bash
pip install -r requirements.txt
streamlit run app.py
```

### Server (GPU)
```bash
pip install -r requirements_server.txt
export STT_API_KEY="your-secret-key"
export HF_TOKEN="your-huggingface-token"
python server_api.py
```

### SSH Tunnel
Lokal maşından GPU serverə qoşulmaq üçün:
```bash
ssh -f -N -L 8000:127.0.0.1:8000 your-server
```

## Gemini API (Opsional)

Gemini API açarı daxil edildikdə sistem:
- Whisper-in səhv yazdığı sözləri düzəldir
- Şəxs adlarını və soyadlarını doğru formaya gətirir
- Qrammatik səhvləri aradan qaldırır

API açarını [Google AI Studio](https://aistudio.google.com/apikey)-dan pulsuz əldə edə bilərsiniz.

Sidebar-da "Gemini API Açarı" sahəsinə daxil edin — avtomatik işləyəcək.

## Müəllim Adı Tapma (Regex)

| # | Pattern | Nümunə |
|---|---------|--------|
| 1 | Soyad Ad, ata oğlu/qızı | `Abbasov Rəhman, Mihman oğlu` |
| 2 | adım X deyil, adım Y | `adım Rehman deyil, adım Rəhman` |
| 3 | Vergüldən sonra ad | `müəllimiyəm, adım Rəhman` |
| 4 | mən ... yam, Ad Soyad | `mən sinif müəllimiyəm, Rəhman Abbasov` |
| 5 | adım Ad Soyad | `adım Rəhman Abbasov` |
| 6 | adım Ad (tək) | `adım Rəhman` |
| 7 | Ad sinif müəllimi | `İbtar sinif müəllimi` |
| 8 | İlk seqmentdə Ad Soyad | İki böyük hərfli ardıcıl söz |

## Fayl Strukturu

| Fayl | Təsvir |
|------|--------|
| `app.py` | Streamlit web interfeysi (client) — 1 fayl, 1 input |
| `server_api.py` | FastAPI server — GPU-da Whisper + PyAnnote + Regex + Gemini |
| `requirements.txt` | Client asılılıqları (streamlit, requests) |
| `requirements_server.txt` | Server asılılıqları (torch, whisper, pyannote, genai) |
