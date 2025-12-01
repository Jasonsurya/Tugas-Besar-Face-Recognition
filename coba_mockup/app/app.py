import streamlit as st
import cv2
import numpy as np
from PIL import Image
import torch
import time
from model_loader import load_all_components, DEVICE 

# --- A. Muat Komponen AI (Sekali di awal) ---
try:
    model, detector, idx_to_class, pytorch_transform = load_all_components() 
except Exception as e:
    st.error(f"Error memuat model: {e}")
    st.stop()

# --- Data Mahasiswa ---
DATA_MAHASISWA = {
    '119140157': {'nama': 'JOYAPUL HANSCALVIN PANJAITAN', 'nim': '119140157'},
    '120140156': {'nama': 'MUHAMMAD ZADA RIZKI', 'nim': '120140156'},
    '121140135': {'nama': 'DIMAS AZI RAJAB AIZAR', 'nim': '121140135'},
    '122140001': {'nama': 'GARLAND WIJAYA', 'nim': '122140001'},
    '122140005': {'nama': 'AHMAD FAQIH HASANI', 'nim': '122140005'},
    '122140006': {'nama': 'FEMMY APRILLIA PUTRI', 'nim': '122140006'},
    '122140008': {'nama': 'BINTANG FIKRI FAUZAN', 'nim': '122140008'},
    '122140009': {'nama': 'KAYLA CHIKA LATHISYA', 'nim': '122140009'},
    '122140010': {'nama': 'YOHANNA ANZELIKA SITEPU', 'nim': '122140010'},
    '122140012': {'nama': 'FERDANA AL HAKIM', 'nim': '122140012'},
    '122140016': {'nama': 'NASYA AULIA EFENDI', 'nim': '122140016'},
    '122140018': {'nama': 'FREDDY HARAHAP', 'nim': '122140018'},
    '122140027': {'nama': 'ALIEF FATHUR RAHMAN', 'nim': '122140027'},
    '122140037': {'nama': 'MUHAMMAD RIVELDO HERMAWAN PUTRA', 'nim': '122140037'},
    '122140038': {'nama': 'ARKAN HARIZ CHANDRAWINATA LIEM', 'nim': '122140038'},
    '122140039': {'nama': 'JOY DANIELLA V', 'nim': '122140039'},
    '122140043': {'nama': 'KENNETH AUSTIN WIJAYA', 'nim': '122140043'},
    '122140055': {'nama': 'FATHAN ANDI KARTAGAMA', 'nim': '122140055'},
    '122140056': {'nama': 'GABRIELLA NATALYA RUMAPEA', 'nim': '122140056'},
    '122140076': {'nama': 'DESTY ANANTA PURBA', 'nim': '122140076'},
    '122140077': {'nama': 'RAHMAT ALDI NASDA', 'nim': '122140077'},
    '122140087': {'nama': 'FESTUS MIKHAEL', 'nim': '122140087'},
    '122140095': {'nama': 'ABRAHAM GANDA NAPITU', 'nim': '122140095'},
    '122140098': {'nama': 'LOIS NOVEL E GURNING', 'nim': '122140098'},
    '122140100': {'nama': 'ZIDAN RAIHAN', 'nim': '122140100'},
    '122140101': {'nama': 'ZEFANYA DANOVANTA TARIGAN', 'nim': '122140101'},
    '122140103': {'nama': 'HAYYATUL FAJRI', 'nim': '122140103'},
    '122140104': {'nama': 'MYCHAEL DANIEL N', 'nim': '122140104'},
    '122140116': {'nama': 'REYNALDI CRISTIAN SIMAMORA', 'nim': '122140116'},
    '122140117': {'nama': 'ICHSAN KUNTADI BASKARA', 'nim': '122140117'},
    '122140118': {'nama': 'FAJRUL RAMADHAN AQSA', 'nim': '122140118'},
    '122140119': {'nama': 'MARTUA KEVIN A.M.H.LUBIS', 'nim': '122140119'},
    '122140122': {'nama': 'ALFAJAR', 'nim': '122140122'},
    '122140127': {'nama': 'RIZKY ABDILLAH', 'nim': '122140127'},
    '122140129': {'nama': 'BAYU EGA FERDANA', 'nim': '122140129'},
    '122140130': {'nama': 'WILLIAM CHAN', 'nim': '122140130'},
    '122140132': {'nama': 'FALIH DZAKWAN ZUHDI', 'nim': '122140132'},
    '122140134': {'nama': 'RAYHAN FATIH GUNAWAN', 'nim': '122140134'},
    '122140135': {'nama': 'ELSA ELISA YOHANA SIANTURI', 'nim': '122140135'},
    '122140137': {'nama': 'IKHSANNUDIN LATHIEF', 'nim': '122140137'},
    '122140138': {'nama': 'SHINTYA AYU WARDANI', 'nim': '122140138'},
    '122140140': {'nama': 'BEZALEL SAMUEL MANIK', 'nim': '122140140'},
    '122140141': {'nama': 'JOSHUA PALTI SINAGA', 'nim': '122140141'},
    '122140144': {'nama': 'DWI ARTHUR REVANGGA', 'nim': '122140144'},
    '122140145': {'nama': 'DWI DYO CAROL BUKIT', 'nim': '122140145'},
    '122140150': {'nama': 'ALDI SANJAYA', 'nim': '122140150'},
    '122140152': {'nama': 'FIQRI ALDIANSYAH', 'nim': '122140152'},
    '122140153': {'nama': 'DITO RIFKI IRAWAN', 'nim': '122140153'},
    '122140155': {'nama': 'RUSTIAN AFENCIUS MARBUN', 'nim': '122140155'},
    '122140160': {'nama': 'HAVIDZ RIDHO PRATAMA', 'nim': '122140160'},
    '122140163': {'nama': 'BOY SANDRO SIGIRO', 'nim': '122140163'},
    '122140164': {'nama': 'ABU BAKAR SIDDIQ SIREGAR', 'nim': '122140164'},
    '122140165': {'nama': 'EICHAL ELPHINDO GINTING', 'nim': '122140165'},
    '122140169': {'nama': 'JP RAFI RADIKTYA ARKAN', 'nim': '122140169'},
    '122140170': {'nama': 'JOSHIA FERNANDES SECTIO PURBA', 'nim': '122140170'},
    '122140171': {'nama': 'RANDY HENDRIYAWAN', 'nim': '122140171'},
    '122140172': {'nama': 'MACHZAUL HARMANSYAH', 'nim': '122140172'},
    '122140173': {'nama': 'MUHAMMAD NELWAN FAKHRI', 'nim': '122140173'},
    '122140182': {'nama': 'ZAKY AHMAD MAKARIM', 'nim': '122140182'},
    '122140187': {'nama': 'EDEN WIJAYA', 'nim': '122140187'},
    '122140198': {'nama': 'ZAKHI ALGIFARI', 'nim': '122140198'},
    '122140202': {'nama': 'FAYYADH ABDILLAH', 'nim': '122140202'},
    '122140207': {'nama': 'INTAN PERMATA SARI', 'nim': '122140207'},
    '122140208': {'nama': 'SIKAH NUBUAHTUL ILMI', 'nim': '122140208'},
    '122140209': {'nama': 'RADITYA ERZA FARANDI', 'nim': '122140209'},
    '122140219': {'nama': 'BAYU PRAMESWARA HARIS', 'nim': '122140219'},
    '122140222': {'nama': 'KEVIN NAUFAL DANY', 'nim': '122140222'},
    '122140236': {'nama': 'RAYHAN FADEL IRWANTO', 'nim': '122140236'},
    '122140239': {'nama': 'ROYFRAN ROGER VALENTINO', 'nim': '122140239'},
}

# --- KONFIGURASI THRESHOLD ---
# Diubah menjadi 0.0 agar SEMUA hasil prediksi ditampilkan
CONFIDENCE_THRESHOLD = 0.0 

# --- B. Konfigurasi Halaman ---
st.set_page_config(page_title="Kios Presensi AI (Upload)", layout="wide")

# --- C. Fungsi Pemrosesan Foto ---
def process_uploaded_image(uploaded_file):
    """Membaca file upload, mendeteksi wajah, dan mengenali mahasiswa."""
    
    # 1. Baca File Gambar
    image = Image.open(uploaded_file).convert('RGB')
    img_array = np.array(image)
    
    # Simpan hasil untuk ditampilkan
    result_image = img_array.copy()
    detection_results = []
    
    # 2. Deteksi Wajah (MTCNN)
    try:
        faces = detector.detect_faces(img_array)
    except Exception as e:
        st.error(f"Gagal menjalankan deteksi wajah: {e}")
        return image, []

    if len(faces) == 0:
        return image, []

    # 3. Proses Setiap Wajah yang Ditemukan
    for face in faces:
        x, y, w, h = face['box']
        
        # Validasi koordinat
        x, y = max(0, x), max(0, y)
        w, h = max(0, w), max(0, h)
        
        # Crop Wajah
        face_crop = img_array[y:y+h, x:x+w]
        
        if face_crop.size == 0: continue

        # 4. Pengenalan Wajah (EfficientNet)
        label_text = "Unknown"
        box_color = (255, 0, 0) # Merah (Unknown)
        status_text = "BUKAN MAHASISWA DEEP LEARNING"
        
        try:
            with torch.no_grad():
                face_tensor = pytorch_transform(face_crop).unsqueeze(0).to(DEVICE)
                outputs = model(face_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                
                conf, idx = torch.max(probabilities, 1)
                detected_id = idx_to_class[idx.item()]
                conf_val = conf.item()
                
                # --- LOGIKA BARU (Semua Ditampilkan) ---
                mhs = DATA_MAHASISWA.get(detected_id)
                
                if mhs:
                    label_text = f"{mhs['nama']}"
                    # Tambahkan info confidence di visualisasi juga
                    status_text = f"HADIR ({mhs['nim']})"
                    box_color = (0, 255, 0) # Hijau
                    
                    detection_results.append({
                        "nama": mhs['nama'],
                        "nim": mhs['nim'],
                        "confidence": f"{conf_val*100:.1f}%", # Tampilkan nilai asli
                        "status": "TERDETEKSI"
                    })
                else:
                    label_text = f"ID:{detected_id} (?)"
                    status_text = "DATA TIDAK LENGKAP"
                    box_color = (255, 255, 0) # Kuning
                    
                    detection_results.append({
                        "nama": "-",
                        "nim": f"ID: {detected_id}",
                        "confidence": f"{conf_val*100:.1f}%",
                        "status": "DATA TIDAK LENGKAP"
                    })

        except Exception as e:
            print(f"Error inference: {e}")
            
        # Gambar Kotak dan Label di Gambar Hasil
        cv2.rectangle(result_image, (x, y), (x+w, y+h), box_color, 3)
        # Tampilkan nama dan confidence di gambar
        label_display = f"{label_text} ({conf_val*100:.0f}%)"
        cv2.putText(result_image, label_display, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)

    return result_image, detection_results

# --- D. Tampilan Utama (UI) ---

st.title("📚 Sistem Presensi Mahasiswa Deep Learning")
st.markdown("Unggah foto mahasiswa untuk melakukan presensi otomatis.")
st.markdown("---")

col_upload, col_result = st.columns([1, 1])

with col_upload:
    st.subheader("1. Upload Foto")
    uploaded_file = st.file_uploader("Pilih gambar...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Tampilkan gambar asli
        st.image(uploaded_file, caption="Foto Diunggah", use_container_width=True)
        
        # Tombol Proses
        if st.button("🔍 PROSES PRESENSI", type="primary", use_container_width=True):
            with st.spinner("Mendeteksi wajah dan mencocokkan data..."):
                processed_img, results = process_uploaded_image(uploaded_file)
                
                # Simpan hasil ke session state agar tidak hilang saat refresh
                st.session_state['processed_img'] = processed_img
                st.session_state['results'] = results

with col_result:
    st.subheader("2. Hasil Identifikasi")
    
    if 'processed_img' in st.session_state:
        # Tampilkan gambar hasil deteksi (dengan kotak)
        st.image(st.session_state['processed_img'], caption="Hasil Deteksi", use_container_width=True)
        
        st.markdown("### Detail Presensi")
        
        results = st.session_state.get('results', [])
        
        if not results:
            st.warning("Wajah tidak terdeteksi dalam foto ini.")
        else:
            for res in results:
                with st.container(border=True):
                    c1, c2 = st.columns([3, 1])
                    with c1:
                        st.markdown(f"**Nama:** {res['nama']}")
                        st.markdown(f"**NIM:** {res['nim']}")
                    with c2:
                        st.info(f"Confidence: {res['confidence']}")
            
            # Tombol Simpan (Simulasi)
            if results:
                if st.button("💾 SIMPAN KE DATABASE", type="secondary", use_container_width=True):
                    st.toast("Data kehadiran berhasil disimpan!", icon="✅")
                    st.balloons()

st.markdown("---")
st.caption("Dibuat dengan Streamlit & PyTorch EfficientNet")