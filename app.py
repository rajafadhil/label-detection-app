"""
Streamlit App: Beverage Nutrition Label Detection and Extraction
Pipeline: Image -> YOLOv11n -> ROI -> PaddleOCR (API) -> BPOM Validation (fuzzy) -> Structured nutrition data

Accompanying files that MUST be in the same folder as app.py:
  - best.pt                 (YOLO weights - Make sure this is the 'no prep' version)
  - preprocess.py           (same module as YOLO notebook)
  - nutrition_postproc.py   (same module as OCR notebook)
"""
import json
import os
import time

import cv2
import numpy as np
import pandas as pd
import requests
import streamlit as st
from ultralytics import YOLO

from preprocess import preprocess_image
import nutrition_postproc as npp

# ============================================================ CONFIGURATION
MODEL_PATH = "best.pt"
IMGSZ = 640
DET_CONF = 0.537                 # same as CONF_EVAL / DET_CONF in notebook
FUZZY_THRESHOLD = 60             # sensitivity analysis result (highest F1, precision > threshold 50)
LINE_TOL_MODE = "fixed"          # 'fixed' (10 px) | 'adaptive'
LINE_TOL_FIXED_PX = 10

# Preprocessing disabled (No Prep)
PREPROCESS_BEFORE_DETECTION = False  # Only resize, no CLAHE+Haar
PREPROCESS_ROI_BEFORE_OCR = False    # Raw ROI (no prep) directly to OCR

JOB_URL = "https://paddleocr.aistudio-app.com/api/v2/ocr/jobs"
MODEL_VERSION = "PP-OCRv6"
OPTIONAL_PAYLOAD = {
    "useDocOrientationClassify": False,
    "useDocUnwarping": False,
    "useTextlineOrientation": False,
}
API_MAX_RETRIES = 3
API_POLL_INTERVAL = 2.0
API_TIMEOUT_S = 150

# Keep the keys as Indonesian to match BPOM module logic, but change display values to English
DISPLAY_NAMES = {
    "energi": "Energy",
    "lemak_total": "Total Fat",
    "lemak_jenuh": "Saturated Fat",
    "protein": "Protein",
    "karbohidrat_total": "Total Carbohydrate",
    "gula": "Sugar",
    "natrium": "Sodium",
}

st.set_page_config(page_title="BPOM Nutrition Label Detection", layout="wide", page_icon="🥤")


# ============================================================ TOKEN & MODEL
def get_token():
    try:
        return st.secrets["PADDLE_OCR_TOKEN"]
    except Exception:
        return os.environ.get("PADDLE_OCR_TOKEN")


@st.cache_resource(show_spinner=False)
def load_model():
    model = YOLO(MODEL_PATH)
    # warm-up so first inference is not distorted
    model.predict(np.zeros((640, 480, 3), np.uint8), imgsz=IMGSZ, verbose=False)
    return model


# ============================================================ PREPROCESS & DETECTION
def resize_long_side(img, target=IMGSZ):
    """Same as training pool: longest side = 640, not upscaled."""
    h, w = img.shape[:2]
    s = target / max(h, w)
    if s >= 1:
        return img
    return cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)


def detect_roi(img_raw, model):
    """Return (det | None, t_pre_ms, t_det_ms). Box rescaled to original resolution."""
    H, W = img_raw.shape[:2]

    t = time.perf_counter()
    small = resize_long_side(img_raw)
    x_in = preprocess_image(small) if PREPROCESS_BEFORE_DETECTION else small
    t_pre = (time.perf_counter() - t) * 1000

    t = time.perf_counter()
    r = model.predict(x_in, conf=DET_CONF, imgsz=IMGSZ, verbose=False)[0]
    t_det = (time.perf_counter() - t) * 1000

    if len(r.boxes) == 0:
        return None, t_pre, t_det

    i = int(r.boxes.conf.argmax())  # highest confidence detection
    x1, y1, x2, y2 = r.boxes.xyxy[i].cpu().numpy()
    sh, sw = small.shape[:2]
    sx, sy = W / sw, H / sh
    box = [max(0, int(x1 * sx)), max(0, int(y1 * sy)),
           min(W, int(x2 * sx)), min(H, int(y2 * sy))]
    return dict(bbox=box, conf=float(r.boxes.conf[i])), t_pre, t_det


# ============================================================ PADDLEOCR API
def parse_ocr_jsonl(jsonl_text):
    out = []
    for line in jsonl_text.strip().split("\n"):
        if not line.strip():
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        res = d.get("result", {})
        pages = res.get("ocrResults", []) if isinstance(res, dict) else (res if isinstance(res, list) else [])
        for page in pages:
            if not isinstance(page, dict):
                continue
            pr = page.get("prunedResult", page)
            texts = pr.get("rec_texts") or []
            scores = pr.get("rec_scores") or []
            polys = pr.get("rec_polys") or pr.get("dt_polys") or []
            boxes = pr.get("rec_boxes") or []
            for i, t in enumerate(texts):
                if not t:
                    continue
                poly = polys[i] if i < len(polys) else None
                if poly is None and i < len(boxes):
                    x1, y1, x2, y2 = boxes[i]
                    poly = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
                arr = np.array(poly if poly is not None else [], dtype=float)
                ok = arr.ndim == 2 and arr.shape[1] >= 2 and len(arr) > 0
                out.append(dict(
                    text=str(t),
                    score=float(scores[i]) if i < len(scores) else 0.0,
                    y_center=float(arr[:, 1].mean()) if ok else 0.0,
                    box=arr.tolist() if ok else [],
                ))
    return out


def call_paddle_api(img_bgr, token):
    """Return dict(ocr_data, error, ms, attempts). ms includes submit + polling + download (including retries)."""
    headers = {"Authorization": f"bearer {token}"}
    ok, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not ok:
        return dict(ocr_data=[], error="Encoding failed", ms=0.0, attempts=0)

    payload = {"model": MODEL_VERSION, "optionalPayload": json.dumps(OPTIONAL_PAYLOAD)}
    t0, err = time.perf_counter(), None

    for attempt in range(1, API_MAX_RETRIES + 1):
        try:
            r = requests.post(JOB_URL, headers=headers, data=payload, timeout=60,
                              files={"file": ("roi.jpg", buf.tobytes(), "image/jpeg")})
            if r.status_code != 200:
                raise RuntimeError(f"Submit {r.status_code}: {r.text[:120]}")
            job_id = r.json()["data"]["jobId"]

            json_url, deadline = None, time.perf_counter() + API_TIMEOUT_S
            while time.perf_counter() < deadline:
                time.sleep(API_POLL_INTERVAL)
                s = requests.get(f"{JOB_URL}/{job_id}", headers=headers, timeout=30)
                if s.status_code != 200:
                    raise RuntimeError(f"Poll {s.status_code}")
                d = s.json()["data"]
                if d["state"] == "done":
                    json_url = d["resultUrl"]["jsonUrl"]
                    break
                if d["state"] == "failed":
                    raise RuntimeError(f"Job failed: {d.get('errorMsg')}")
            if json_url is None:
                raise TimeoutError("API timeout")

            jr = requests.get(json_url, timeout=60)
            jr.raise_for_status()
            return dict(ocr_data=parse_ocr_jsonl(jr.text), error=None,
                        ms=(time.perf_counter() - t0) * 1000, attempts=attempt)
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
            time.sleep(2 * attempt)

    return dict(ocr_data=[], error=err, ms=(time.perf_counter() - t0) * 1000, attempts=API_MAX_RETRIES)


# ============================================================ END-TO-END PIPELINE
def run_pipeline(file_bytes, model, token):
    res = dict(error=None, det_conf=None, bbox=None, vis=None, roi_raw=None,
               raw_text="", ocr_items=[], nutrients={}, timing={}, attempts=0)

    img = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        res["error"] = "Failed to read/decode image."
        return res

    det, t_pre, t_det = detect_roi(img, model)
    res["timing"].update({"Resize (detection input)": t_pre,
                          "YOLOv11n Detection": t_det})
    if det is None:
        res["error"] = "Nutrition facts table not detected. Try retaking the photo with more even lighting."
        return res

    x1, y1, x2, y2 = det["bbox"]
    if x2 - x1 < 5 or y2 - y1 < 5:
        res["error"] = "ROI is too small to process."
        return res

    vis = img.copy()
    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), max(3, img.shape[1] // 300))
    res.update(det_conf=det["conf"], bbox=det["bbox"], vis=vis)

    t = time.perf_counter()
    roi_raw = img[y1:y2, x1:x2].copy()
    res["timing"]["Crop ROI"] = (time.perf_counter() - t) * 1000

    t = time.perf_counter()
    roi_ocr = preprocess_image(roi_raw) if PREPROCESS_ROI_BEFORE_OCR else roi_raw
    res["timing"]["ROI Preprocessing"] = (time.perf_counter() - t) * 1000
    res["roi_raw"] = roi_ocr

    api = call_paddle_api(roi_ocr, token)
    res["timing"]["PaddleOCR API (submit+polling+download)"] = api["ms"]
    res["attempts"] = api["attempts"]
    if api["error"]:
        res["error"] = f"OCR failed: {api['error']}"
        return res
    if not api["ocr_data"]:
        res["error"] = "PaddleOCR found no text in the ROI."
        return res

    res["ocr_items"] = api["ocr_data"]
    lines = npp.group_lines(api["ocr_data"], LINE_TOL_MODE, LINE_TOL_FIXED_PX)
    res["raw_text"] = "\n".join(" ".join(i["text"] for i in g) for g in lines)

    t = time.perf_counter()
    res["nutrients"] = npp.extract_nutrients(
        api["ocr_data"], match_mode="fuzzy", numeric_mode="robust",
        threshold=FUZZY_THRESHOLD, line_mode=LINE_TOL_MODE, fixed_tol=LINE_TOL_FIXED_PX)
    res["timing"]["Regex + fuzzy matching"] = (time.perf_counter() - t) * 1000
    return res


# ============================================================ UI
def main():
    st.title("🥤 Beverage Nutrition Label Detection")
    st.caption("YOLOv11n → PaddleOCR → BPOM Dictionary Validation (BPOM Regulation No. 26 of 2021)")

    token = get_token()
    if not token:
        st.error("PaddleOCR token not set. Add `PADDLE_OCR_TOKEN` in Settings > Secrets "
                 "(or as an environment variable).")
        st.stop()
    if not os.path.exists(MODEL_PATH):
        st.error(f"Weight file `{MODEL_PATH}` not found in the application folder.")
        st.stop()

    with st.spinner("Loading YOLOv11n model..."):
        model = load_model()

    with st.sidebar:
        st.header("Pipeline configuration")
        st.write(f"- Detection confidence: **{DET_CONF}**")
        st.write(f"- Fuzzy threshold (partial_ratio): **{FUZZY_THRESHOLD}**")
        st.write(f"- Line grouping: **{LINE_TOL_MODE}** ({LINE_TOL_FIXED_PX} px)")
        st.write(f"- OCR: **{MODEL_VERSION}** (API)")
        st.info("Extraction results are automatic and may contain errors. Please verify the nutritional values "
                "against the physical label, especially if you are on a special diet.")

    st.subheader("1. Insert packaging image")
    tab_up, tab_cam = st.tabs(["📁 Upload image", "📷 Take a photo"])
    file_bytes = None
    with tab_up:
        up = st.file_uploader("Upload image from device", type=["jpg", "jpeg", "png"])
        if up is not None:
            file_bytes = up.getvalue()
    with tab_cam:
        cam = st.camera_input("Position the nutrition facts table in the center of the screen")
        if cam is not None:
            file_bytes = cam.getvalue()

    if file_bytes is None:
        return

    with st.spinner("Processing detection, OCR, and BPOM validation..."):
        t0 = time.perf_counter()
        res = run_pipeline(file_bytes, model, token)
        total_s = time.perf_counter() - t0

    col1, col2 = st.columns(2)
    with col1:
        if res["vis"] is not None:
            st.image(cv2.cvtColor(res["vis"], cv2.COLOR_BGR2RGB),
                     caption=f"YOLOv11n detection result (confidence {res['det_conf']:.2f})",
                     use_container_width=True)
        else:
            st.image(file_bytes, caption="Input image", use_container_width=True)

    with col2:
        if res["error"]:
            st.error(f"❌ {res['error']}")
            return

        st.success(f"✅ Completed in {total_s:.2f} seconds")
        if res["attempts"] > 1:
            st.warning(f"Request to OCR API was retried {res['attempts'] - 1} times.")
        st.image(cv2.cvtColor(res["roi_raw"], cv2.COLOR_BGR2RGB),
                 caption="Nutrition table ROI (OCR input)", use_container_width=True)

    st.subheader("2. Nutritional value extraction results (7 mandatory BPOM categories)")
    rows = []
    for cat in npp.REQUIRED_CATEGORIES:
        v = res["nutrients"].get(cat)
        if v:
            rows.append({
                "Nutrition": DISPLAY_NAMES[cat],
                "Value": f"{v['value']:g} {v['unit']}",
                "Status": "⚠️ Ambiguous match" if v.get("ambiguous") else "✅ Detected",
                "Match score": f"{v['score']:.0f}",
                "Original text (OCR line)": v["orig"],
            })
        else:
            rows.append({"Nutrition": DISPLAY_NAMES[cat], "Value": "-", "Status": "❌ Not detected",
                         "Match score": "-", "Original text (OCR line)": "-"})
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)

    n_det = sum(1 for c in npp.REQUIRED_CATEGORIES if c in res["nutrients"])
    n_amb = sum(1 for v in res["nutrients"].values() if v.get("ambiguous"))
    m1, m2, m3 = st.columns(3)
    m1.metric("Detected categories", f"{n_det} / {len(npp.REQUIRED_CATEGORIES)}")
    m2.metric("Ambiguous matches", n_amb)
    m3.metric("Total time", f"{total_s:.2f} s")
    if n_det < len(npp.REQUIRED_CATEGORIES):
        st.info("Some categories were not readable. Try retaking the photo with even lighting, "
                "no glare, and the label in a straight position.")

    with st.expander("🔍 Raw OCR text per line (before BPOM validation)"):
        st.text_area("Raw OCR", res["raw_text"], height=180)

    with st.expander("⏱️ Time breakdown per stage"):
        tdf = pd.DataFrame({"Stage": list(res["timing"]),
                            "Time (ms)": [round(v, 2) for v in res["timing"].values()]})
        st.dataframe(tdf, use_container_width=True, hide_index=True)
        st.caption("OCR time depends on internet connection and PaddleOCR API server load.")

    st.download_button("⬇️ Download results (CSV)", df.to_csv(index=False).encode("utf-8"),
                       file_name="nutrition_extraction_results.csv", mime="text/csv")


if __name__ == "__main__":
    main()
