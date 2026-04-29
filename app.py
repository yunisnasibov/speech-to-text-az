import streamlit as st
import requests
import os
import tempfile
import subprocess

# ═══════════════════════════════════════════════════════════
# KONFİQURASİYA — Bu dəyərləri öz mühitinizə uyğun dəyişin
# ═══════════════════════════════════════════════════════════
SERVER_URL = "http://127.0.0.1:8000/transcribe"
SERVER_API_KEY = "stt-secret-key-2026"

st.set_page_config(page_title="Sinif Monitorinq STT", page_icon="🎤", layout="centered")

st.title("🎤 Sinif Monitorinq — Səsdən Mətnə")
st.caption("Video yükləyin → Danışanlar ayrılır → Mətn düzəldilir → Müəllimin adı tapılır")

# Gemini API açarı (sidebar-da gizli)
gemini_key = st.sidebar.text_input("🔑 Gemini API Açarı", type="password", 
                                     help="Google AI Studio-dan pulsuz alın: aistudio.google.com/apikey")

uploaded_file = st.file_uploader(
    "📁 Video və ya səs faylını yükləyin",
    type=["mp4", "mov", "mp3", "wav", "m4a", "mpeg4"],
    help="Sinif monitorinq videosunu bura sürükləyin"
)


def extract_audio(file_path, output_path):
    """Video faylından səsi çıxarır (16kHz mono WAV)."""
    subprocess.run(
        ["ffmpeg", "-y", "-i", file_path, "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", output_path],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )


if uploaded_file is not None:
    if st.button("🚀 Təhlil Et", type="primary", use_container_width=True):

        # 1. Faylı müvəqqəti saxla
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_input:
            tmp_input.write(uploaded_file.read())
            input_path = tmp_input.name

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_wav:
            wav_path = tmp_wav.name

        # 2. Səsə çevir
        with st.spinner("🎬 Video səsə çevrilir..."):
            extract_audio(input_path, wav_path)

        # 3. Serverə göndər
        with st.spinner("🧠 Səs təhlil edilir (danışanlar ayrılır + mətnə çevrilir + düzəldilir)..."):
            try:
                headers = {"X-API-Key": SERVER_API_KEY}
                if gemini_key:
                    headers["X-Gemini-Key"] = gemini_key

                with open(wav_path, "rb") as f:
                    response = requests.post(SERVER_URL, files={"file": f}, headers=headers, timeout=600)

                if response.status_code == 200:
                    result = response.json()

                    if result.get("status") == "success":
                        st.success("✅ Təhlil tamamlandı!")

                        # Müəllimin adı
                        teacher_name = result.get("teacher_name")
                        if teacher_name:
                            st.markdown(f"## 👩‍🏫 Müəllim: **{teacher_name}**")
                        else:
                            st.warning("⚠️ Müəllimin adı transkripsiyada tapılmadı.")
                        
                        st.markdown("---")
                        st.markdown("### 📝 Transkripsiya")

                        final_text = ""
                        for item in result["data"]:
                            line = f"**{item['speaker']}** ({item['start']}s - {item['end']}s): {item['text']}"
                            st.markdown(line)
                            final_text += line + "\n\n"

                        st.download_button("📥 Nəticəni Yüklə", data=final_text, file_name="transkripsiya.txt",
                                           use_container_width=True)
                    else:
                        st.error(f"Server xətası: {result.get('message')}")

                elif response.status_code == 401:
                    st.error("❌ Yanlış server API açarı!")
                else:
                    st.error(f"Serverə qoşulmaq alınmadı. Status: {response.status_code}")

            except requests.exceptions.ConnectionError:
                st.error("❌ Serverə bağlantı yoxdur. SSH tunnel-in açıq olduğundan əmin olun.")
            except requests.exceptions.Timeout:
                st.error("⏰ Server cavab vermədi. Fayl çox böyük ola bilər.")
            except Exception as e:
                st.error(f"Xəta: {e}")

        # Təmizlik
        try:
            os.unlink(input_path)
            os.unlink(wav_path)
        except:
            pass
