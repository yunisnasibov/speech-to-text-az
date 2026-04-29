import streamlit as st
import requests
import os
import tempfile
import subprocess

st.set_page_config(page_title="Azərbaycan Dili STT", page_icon="🎤", layout="wide")

st.title("🎤 Azərbaycan Dilində Səsdən Mətnə (Server İnteqrasiyası)")
st.markdown("Bu interfeys faylı sizin qurduğunuz SSH GPU Serverinə göndərir və emal edilmiş mətni geri alır.")

st.sidebar.header("⚙️ Server Ayarları")
# Bura SSH serverinizin IP ünvanını və ya domainini yazacaqsınız
server_url = st.sidebar.text_input("Serverin API ünvanı", value="http://127.0.0.1:8000/transcribe")
api_key = st.sidebar.text_input("API Açarı", value="stt-secret-key-2026", type="password")

uploaded_file = st.file_uploader("Video və ya səs faylını yükləyin (mp4, mov, mp3, wav, m4a)", type=["mp4", "mov", "mp3", "wav", "m4a"])

def extract_audio(file_path, output_path):
    command = [
        "ffmpeg", "-y", "-i", file_path,
        "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", output_path
    ]
    subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

if uploaded_file is not None:
    if st.button("Serverə Göndər və Təhlil Et"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_input:
            tmp_input.write(uploaded_file.read())
            input_file_path = tmp_input.name
            
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_wav:
            wav_file_path = tmp_wav.name
            
        with st.spinner("Video səsə çevrilir..."):
            extract_audio(input_file_path, wav_file_path)
            
        with st.spinner("Səs serverə göndərilir (Danışanlar ayrılır və mətnə çevrilir)... Bu bir az vaxt apara bilər!"):
            try:
                with open(wav_file_path, "rb") as f:
                    response = requests.post(
                        server_url,
                        files={"file": f},
                        headers={"X-API-Key": api_key},
                        timeout=300
                    )
                
                if response.status_code == 200:
                    result = response.json()
                    if result.get("status") == "success":
                        st.success("Təhlil Tamamlandı!")
                        st.write("### 📝 Nəticə (Transkripsiya)")
                        
                        final_text = ""
                        for item in result["data"]:
                            line = f"**{item['speaker']}** ({item['start']}s - {item['end']}s): {item['text']}"
                            st.markdown(line)
                            final_text += line + "\n\n"
                        
                        # Müəllimin tam adını göstər (Gemini tərəfindən tapılır)
                        teacher_name = result.get("teacher_name")
                        if teacher_name:
                            st.markdown("---")
                            st.markdown(f"### 👩‍🏫 Müəllimin tam adı: **{teacher_name}**")
                        
                        st.download_button("Nəticəni Yüklə", data=final_text, file_name="transkripsiya.txt")
                    else:
                        st.error(f"Server xətası: {result.get('message')}")
                elif response.status_code == 401:
                    st.error("❌ Yanlış API açarı! Sidebar-dan düzgün açarı daxil edin.")
                else:
                    st.error(f"Serverə qoşulmaq alınmadı. Status kod: {response.status_code}")
                    
            except Exception as e:
                st.error(f"Bağlantı xətası. Serverin işlədiyindən və API ünvanının doğru olduğundan əmin olun: {e}")
                
        # Təmizlik
        try:
            os.unlink(input_file_path)
            os.unlink(wav_file_path)
        except:
            pass
