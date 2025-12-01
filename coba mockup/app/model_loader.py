import torch
import torch.nn as nn
from torchvision import transforms
from mtcnn.mtcnn import MTCNN
from timm import create_model 
import json
import streamlit as st
import numpy as np
import os 
import sys 

# --- Konfigurasi ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = 224

# --- Setup Path yang Aman (Absolut) ---
# Mengambil lokasi file model_loader.py ini berada
CURRENT_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
# Naik satu tingkat ke root proyek
ROOT_DIR = os.path.realpath(os.path.join(CURRENT_FILE_DIR, '..'))

# Definisikan path file relatif terhadap root
MODEL_PATH = os.path.join(ROOT_DIR, 'EfficientNet_B1_face_recognition_best.pth') 
LABEL_MAP_PATH = os.path.join(ROOT_DIR, 'models', 'label_map.json') 

@st.cache_resource
def load_all_components():
    """
    Memuat semua komponen AI (Model, Detektor, Peta Label) HANYA SEKALI.
    Menggunakan st.cache_resource agar tidak reload setiap kali ada interaksi user.
    """
    
    # Inisialisasi variabel untuk safety
    model = None
    detector = None
    idx_to_class = None
    pytorch_transform = None
    
    try:
        # 1. Muat Peta Label (JSON)
        # Digunakan untuk mengubah ID prediksi (0, 1, 2) menjadi NIM/Nama
        with open(LABEL_MAP_PATH, 'r') as f:
            label_map = json.load(f)
            # Balikkan mapping: {NIM: ID} -> {ID: NIM}
            idx_to_class = {v: k for k, v in label_map.items()}
            num_classes = len(label_map)

        # 2. Bangun Arsitektur Model EfficientNet B1
        # pretrained=False karena kita akan memuat bobot kita sendiri
        model = create_model('efficientnet_b1', pretrained=False, num_classes=0) 
        
        # Ambil jumlah fitur input untuk classification head
        # Menggunakan model.num_features adalah cara yang aman di timm
        num_ftrs = model.num_features 
        
        # Definisi Classification Head Kustom
        # Struktur ini harus sama dengan yang digunakan saat training
        model.classifier = nn.Sequential(
            nn.Linear(num_ftrs, 512), 
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes) 
        )

        # 3. Muat Bobot Model (.pth)
        # map_location memastikan model dimuat ke CPU jika GPU tidak ada
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
        
        # Logika ekstraksi state_dict yang fleksibel
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
            
        # Load bobot dengan strict=False untuk mengabaikan error mismatch minor
        model.load_state_dict(state_dict, strict=False) 
        
        # Pindahkan model ke device (CPU/GPU) dan set ke mode evaluasi
        model.to(DEVICE).eval()

        # 4. Muat Face Detector (MTCNN)
        # Digunakan untuk memotong wajah dari foto yang diupload
        detector = MTCNN()
        
        # 5. Definisikan Transformasi Input PyTorch
        # Standarisasi gambar agar sesuai dengan format input model saat training
        pytorch_transform = transforms.Compose([
            transforms.ToPILImage(), # Pastikan input dikonversi ke format PIL
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        return model, detector, idx_to_class, pytorch_transform
        
    except Exception as e:
        # Tampilkan error yang jelas di UI Streamlit jika gagal
        st.error(f"""
        FATAL ERROR: Gagal memuat komponen AI.
        
        Penyebab: {e}
        
        Path Model: {MODEL_PATH}
        Path Label: {LABEL_MAP_PATH}
        """)
        return None, None, None, None