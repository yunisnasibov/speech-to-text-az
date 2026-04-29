# 🎤 Azərbaycan Dilində Səsdən Mətnə (Speech-to-Text)

Azərbaycan dilində danışılan video/audio fayllarını mətnə çevirən və danışanları ayıran sistem.

## Xüsusiyyətlər

- **Whisper Large-v3** ilə Azərbaycan dilində yüksək dəqiqliklə səsdən mətnə çevirmə
- **PyAnnote 3.1** ilə danışanların ayrılması (diarization)
- **API Key** autentifikasiyası ilə təhlükəsiz giriş
- **Streamlit** ilə istifadəçi dostu web interfeys
- GPU dəstəyi (NVIDIA CUDA)

## Arxitektura

```
┌─────────────────┐         ┌──────────────────────┐
│  Streamlit UI   │  HTTP   │   FastAPI Server     │
│  (app.py)       │────────▶│   (server_api.py)    │
│  Lokal Mac      │  API Key│   GPU Server         │
└─────────────────┘         │  ┌────────────────┐  │
                            │  │ Whisper Large-v3│  │
                            │  │ PyAnnote 3.1   │  │
                            │  └────────────────┘  │
                            └──────────────────────┘
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

## SSH Tunnel
Lokal maşından serverə qoşulmaq üçün:
```bash
ssh -f -N -L 8000:127.0.0.1:8000 your-server
```

## Fayl Strukturu

| Fayl | Təsvir |
|------|--------|
| `app.py` | Streamlit web interfeysi (client) |
| `server_api.py` | FastAPI server (GPU-da işləyir) |
| `requirements.txt` | Client asılılıqları |
| `requirements_server.txt` | Server asılılıqları |
