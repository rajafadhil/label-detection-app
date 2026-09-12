import streamlit as st
import cv2
import numpy as np
import re
import time
import json
import requests
import tempfile
import os
import pandas as pd
from ultralytics import YOLO
from rapidfuzz import fuzz, process
from Levenshtein import distance as levenshtein_distance

# --- KONFIGURASI AWAL ---
st.set_page_config(page_title="Deteksi Label Gizi BPOM", layout="wide", page_icon="🥤")

# Konfigurasi Path & Model (Pastikan file best.pt ada di folder yang sama)
MODEL_PATH = 'best.pt' 
PADDLE_API_TOKEN = st.secrets.get("PADDLE_OCR_TOKEN", None) # Gunakan Secrets Streamlit untuk keamanan

if not PADDLE_API_TOKEN:
    st.error("⚠️ Token PaddleOCR API tidak ditemukan. Pastikan Anda menambahkan token di Settings > Secrets.")
    st.stop()

JOB_URL = "https://paddleocr.aistudio-app.com/api/v2/ocr/jobs"
MODEL_VERSION = "PP-OCRv6" 

HEADERS = {
    "Authorization": f"bearer {PADDLE_API_TOKEN}"
}

OPTIONAL_PAYLOAD = {
    "useDocOrientationClassify": False,
    "useDocUnwarping": False,
    "useTextlineOrientation": False,
}

# --- KAMUS BPOM & FUZZY MATCHING ---
BPOM_NUTRIENT_DICTIONARY = {
    'energi': ['energi', 'energy', 'tenaga', 'kalori', 'calories', 'kcal', 'kkal'],
    'lemak_total': ['lemak total', 'total fat', 'lipid', 'total lemak'],
    'lemak_jenuh': ['lemak jenuh', 'saturated fat', 'lemak sat', 'sat fat'],
    'protein': ['protein', 'proteine'],
    'karbohidrat_total': ['karbohidrat total', 'total carbohydrate', 'carbs', 'kohlenhydrate'],
    'gula': ['gula', 'sugars', 'zucker', 'sucre'],
    'natrium': ['natrium', 'sodium', 'garam', 'salt']
}

NUTRIENT_UNITS = {
    'energi': 'kkal', 'lemak_total': 'g', 'lemak_jenuh': 'g',
    'protein': 'g', 'karbohidrat_total': 'g', 'gula': 'g', 'natrium': 'mg'
}

MIN_TERM_LENGTH = 3

all_variations = []
mapping = {}
for cat, vars in BPOM_NUTRIENT_DICTIONARY.items():
    for v in vars:
        if len(v) < MIN_TERM_LENGTH:
            continue
        all_variations.append(v)
        mapping[v] = cat

# --- FUNGSI BANTUAN ---

@st.cache_resource
def load_model():
    """Memuat model YOLOv11 sekali saja saat aplikasi dimulai"""
    try:
        return YOLO(MODEL_PATH)
    except Exception as e:
        st.error(f"Gagal memuat model YOLO: {e}")
        return None

def clean_ocr_text(text):
    """Membersihkan teks dari karakter noise"""
    text = re.sub(r'[^a-zA-Z0-9\s.,%/-]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text.lower()

def extract_numeric_value_robust(text, expected_unit=''):
    """Ekstraksi nilai numerik yang robust"""
    text_clean = text.strip().replace(',', '.')
    text_clean = re.sub(r'(\d)O(\d)', r'\g<1>0\2', text_clean, flags=re.IGNORECASE)
    text_clean = re.sub(r'(\d)l(\d)', r'\g<1>1\2', text_clean, flags=re.IGNORECASE)

    pattern = r'(\d+\.?\d*)\s*(g|mg|kkal|kcal|%)?'
    match = re.search(pattern, text_clean, re.IGNORECASE)

    if match:
        val_str = match.group(1)
        unit = (match.group(2) or '').lower()
        try:
            val = float(val_str)
            final_unit = unit if unit else expected_unit
            # Koreksi sederhana jika satuan tidak sesuai ekspektasi (opsional)
            if expected_unit == 'g' and val > 100: 
                val = val / 10
            return val, final_unit
        except ValueError:
            pass

    numbers = re.findall(r'\d+\.?\d*', text_clean)
    if numbers:
        try:
            return float(numbers[0]), expected_unit
        except ValueError:
            pass

    return None, None

def process_fuzzy_matching(ocr_data):
    """Memproses data OCR mentah menjadi struktur nutrisi menggunakan Fuzzy Matching"""
    if not ocr_data:
        return "", {}

    raw_text = " ".join([d['text'] for d in ocr_data])

    # 1. Grouping Spasial (Berdasarkan Y-Center)
    tolerance = 10 
    items_sorted = sorted(ocr_data, key=lambda i: i['y_center'])
    sorted_lines = []
    current_group, current_y = [], None
    for item in items_sorted:
        if current_y is None or abs(item['y_center'] - current_y) <= tolerance:
            current_group.append(item)
            current_y = sum(i['y_center'] for i in current_group) / len(current_group)
        else:
            sorted_lines.append((current_y, current_group))
            current_group, current_y = [item], item['y_center']
    if current_group:
        sorted_lines.append((current_y, current_group))
    
    all_candidates = {}

    # 2. Fuzzy Matching per Baris
    for y_pos, items in sorted_lines:
        items.sort(key=lambda i: float(np.mean(np.array(i['box'])[:, 0])) if len(i['box']) > 0 else 0)
        line_text = " ".join([i['text'] for i in items])
        line_text_clean = clean_ocr_text(line_text)

        if not line_text_clean: continue

        match = process.extractOne(line_text_clean, all_variations, scorer=fuzz.partial_ratio)
        if match:
            best_match_term, score, idx = match
            if score >= 65: # Threshold similarity
                found_cat = mapping[best_match_term]
                expected_unit = NUTRIENT_UNITS.get(found_cat, '')
                val, unit = extract_numeric_value_robust(line_text, expected_unit)

                if val is not None:
                    # Ambil nilai terbaik jika ada duplikasi kategori
                    if found_cat not in all_candidates or score > all_candidates[found_cat]['score']:
                        all_candidates[found_cat] = {
                            'value': val, 'unit': unit if unit else expected_unit,
                            'orig': line_text.strip(), 'score': score
                        }

    corrected = {k: v for k, v in all_candidates.items()}
    # Hapus score dari output akhir agar rapi
    for k in corrected: 
        if 'score' in corrected[k]: del corrected[k]['score']

    return raw_text, corrected

def run_pipeline_api_paddle(img_input, model):
    """Pipeline End-to-End: Streamlit Input -> YOLOv11 -> Crop -> PaddleOCR API -> Fuzzy"""
    result = {'det_conf': 0, 'raw_text': '', 'nutrients': {}, 'error': None, 'roi_img': None}

    # 1. Baca file dari Streamlit UploadedFile ke OpenCV format
    file_bytes = np.asarray(bytearray(img_input.read()), dtype=np.uint8)
    img_cv = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    if img_cv is None:
        result['error'] = 'Gagal membaca/mendekode gambar'
        return result

    # 2. Deteksi YOLOv11
    results = model.predict(img_cv, conf=0.5, verbose=False)
    if len(results[0].boxes) == 0:
        result['error'] = 'No ROI detected (Label tidak terdeteksi)'
        return result

    box = results[0].boxes.xyxy[0].cpu().numpy()
    result['det_conf'] = float(results[0].boxes.conf[0].cpu().numpy())
    x1, y1, x2, y2 = map(int, box)
    
    # Validasi koordinat
    h, w = img_cv.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    
    roi = img_cv[y1:y2, x1:x2]
    result['roi_img'] = roi # Simpan untuk visualisasi

    # 3. Simpan ROI ke temporary file untuk dikirim ke API
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_file:
        cv2.imwrite(tmp_file.name, roi)
        temp_img_path = tmp_file.name

    try:
        # 4. Submit Job ke PaddleOCR API
        data = {"model": MODEL_VERSION, "optionalPayload": json.dumps(OPTIONAL_PAYLOAD)}
        with open(temp_img_path, "rb") as f:
            job_response = requests.post(JOB_URL, headers=HEADERS, data=data, files={"file": f})

        if job_response.status_code != 200:
            result['error'] = f"API Submit Failed: {job_response.text}"
            return result

        job_id = job_response.json()["data"]["jobId"]

        # 5. Polling hingga job selesai
        jsonl_url = ""
        max_retries = 15
        for _ in range(max_retries):
            time.sleep(2) # Wait 2 seconds per retry
            job_result_response = requests.get(f"{JOB_URL}/{job_id}", headers=HEADERS)
            if job_result_response.status_code != 200: break

            state = job_result_response.json()["data"]["state"]
            if state == 'done':
                jsonl_url = job_result_response.json()['data']['resultUrl']['jsonUrl']
                break
            elif state == "failed":
                result['error'] = f"API Job Failed"
                break

        if not jsonl_url:
            result['error'] = 'API Timeout or Failed'
            return result

        # 6. Parse Hasil JSONL
        jsonl_response = requests.get(jsonl_url)
        jsonl_response.raise_for_status()
        lines = jsonl_response.text.strip().split('\n')

        ocr_data = []
        for line in lines:
            if not line.strip(): continue
            try:
                data_json = json.loads(line)
            except json.JSONDecodeError:
                continue

            page_results = data_json.get("result", {}).get("ocrResults", [])
            if not page_results and isinstance(data_json.get("result"), list):
                page_results = data_json["result"]

            for page in page_results:
                if not isinstance(page, dict): continue

                pruned = page.get("prunedResult", page)
                rec_texts = pruned.get("rec_texts")

                if rec_texts:
                    rec_scores = pruned.get("rec_scores", [0.0] * len(rec_texts))
                    rec_polys = pruned.get("rec_polys", pruned.get("dt_polys", []))
                    rec_boxes = pruned.get("rec_boxes", [])
                    
                    for i, text in enumerate(rec_texts):
                        if not text: continue
                        score = rec_scores[i] if i < len(rec_scores) else 0.0
                        poly = rec_polys[i] if i < len(rec_polys) else None
                        
                        if not poly and i < len(rec_boxes):
                            x1b, y1b, x2b, y2b = rec_boxes[i]
                            poly = [[x1b, y1b], [x2b, y1b], [x2b, y2b], [x1b, y2b]]
                        
                        poly = poly or []
                        try:
                            poly_np = np.array(poly)
                            y_center = float(np.mean(poly_np[:, 1])) if poly_np.ndim == 2 and poly_np.shape[1] >= 2 else 0.0
                        except Exception:
                            y_center = 0.0
                            
                        ocr_data.append({'text': str(text), 'score': float(score) if score else 0.0, 'y_center': y_center, 'box': poly})

        if not ocr_data:
            result['error'] = 'OCR Data Empty (No text detected by PaddleOCR)'
            return result

        # 7. Proses Fuzzy Matching
        result['raw_text'], result['nutrients'] = process_fuzzy_matching(ocr_data)

    except Exception as e:
        result['error'] = f'Pipeline Error: {str(e)}'
    finally:
        if os.path.exists(temp_img_path): os.remove(temp_img_path)

    return result

# --- UI STREAMLIT ---

def main():
    st.title("🥤 Deteksi Label Gizi Minuman (YOLOv11 + PaddleOCR)")
    st.markdown("""
    Aplikasi ini mendeteksi tabel informasi nilai gizi pada kemasan minuman, 
    mengekstrak teks menggunakan OCR, dan memperbaiki hasil ekstraksi menggunakan 
    **Fuzzy Matching** berdasarkan standar **BPOM**.
    """)

    # Load Model
    with st.spinner("Memuat Model YOLOv11..."):
        model = load_model()
    
    if model is None:
        st.stop()

    # --- BAGIAN INPUT GAMBAR / KAMERA ---
    st.subheader("1. Masukkan Gambar Kemasan")
    
    # Buat tab untuk memisahkan opsi Upload dan Kamera
    tab_upload, tab_kamera = st.tabs(["📁 Upload Gambar", "📷 Ambil Foto Langsung"])

    img_input = None

    with tab_upload:
        uploaded_file = st.file_uploader("Unggah gambar dari perangkat Anda", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            img_input = uploaded_file

    with tab_kamera:
        camera_file = st.camera_input("Posisikan label informasi nilai gizi di tengah layar, lalu klik tombol kamera.")
        if camera_file is not None:
            img_input = camera_file

    # --- PROSES JIKA ADA INPUT (DARI UPLOAD ATAU KAMERA) ---
    if img_input is not None:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.image(img_input, caption="Gambar Input (Upload/Kamera)", use_column_width=True)
        
        with col2:
            with st.spinner("Sedang memproses deteksi dan ekstraksi... Mohon tunggu."):
                # Proses Pipeline
                start_time = time.time()
                result = run_pipeline_api_paddle(img_input, model)
                elapsed_time = time.time() - start_time
                
                if result['error']:
                    st.error(f"❌ Terjadi Kesalahan: {result['error']}")
                else:
                    st.success(f"✅ Berhasil diproses dalam {elapsed_time:.2f} detik!")
                    
                    # Tampilkan ROI (Area yang dideteksi YOLO)
                    if result['roi_img'] is not None:
                        roi_rgb = cv2.cvtColor(result['roi_img'], cv2.COLOR_BGR2RGB)
                        st.image(roi_rgb, caption="Area Tabel Gizi (ROI)", use_column_width=True)
                    
                    # Tampilkan Hasil Raw OCR
                    with st.expander("🔍 Lihat Hasil OCR Mentah (Sebelum Fuzzy)"):
                        st.text_area("Raw Text", result['raw_text'], height=150)
                    
                    # Tampilkan Hasil Akhir (Setelah Fuzzy)
                    st.subheader("📊 Hasil Ekstraksi Nilai Gizi (Validasi BPOM)")
                    
                    if result['nutrients']:
                        data_display = []
                        for key, val in sorted(result['nutrients'].items()):
                            # Mapping kunci internal ke nama tampil yang bagus
                            display_name = key.replace('_', ' ').title()
                            data_display.append({
                                "Nutrisi": display_name,
                                "Nilai": f"{val['value']} {val['unit']}",
                                "Teks Asli": val['orig']
                            })
                        
                        df_result = pd.DataFrame(data_display)
                        st.table(df_result)
                    else:
                        st.warning("Tidak ada nutrisi valid yang terdeteksi setelah proses Fuzzy Matching.")

if __name__ == "__main__":
    main()