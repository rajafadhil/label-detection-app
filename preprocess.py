"""Preprocessing CLAHE (LAB) + denoising Haar wavelet level 1 (soft threshold).
Non-destruktif: fungsi mengembalikan array baru, tidak menimpa file.
Dipakai identik pada training, evaluasi, dan inference agar tidak ada train/inference mismatch."""
import cv2
import numpy as np
import pywt

def clahe_lab(img_bgr, clip=3.0, tile=(8, 8)):
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=clip, tileGridSize=tile).apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

def haar_denoise_channel(ch, level=1):
    h, w = ch.shape
    coeffs = pywt.wavedec2(ch.astype(np.float32), 'haar', level=level)
    # estimasi sigma noise (MAD) dari detail diagonal level terhalus -> universal threshold
    cD = coeffs[-1][2]
    sigma = np.median(np.abs(cD)) / 0.6745
    thr = float(sigma * np.sqrt(2.0 * np.log(max(ch.size, 2))))
    new = [coeffs[0]] + [tuple(pywt.threshold(c, thr, mode='soft') for c in d) for d in coeffs[1:]]
    rec = pywt.waverec2(new, 'haar')[:h, :w]          # potong jika dimensi ganjil
    rec = np.nan_to_num(rec, nan=0.0, posinf=255.0, neginf=0.0)
    return np.clip(rec, 0, 255).astype(np.uint8)

def preprocess_image(img_bgr, clip=3.0, tile=(8, 8), level=1):
    """CLAHE pada kanal L (LAB), lalu denoising Haar pada kanal L; warna dipertahankan."""
    out = clahe_lab(img_bgr, clip, tile)
    lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = haar_denoise_channel(l, level)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
