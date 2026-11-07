"""Pasca-pemrosesan OCR -> data gizi terstruktur, beserta metrik evaluasi.
Dipakai identik di notebook evaluasi dan aplikasi Streamlit agar angka konsisten."""
import re
import numpy as np
from rapidfuzz import fuzz
from Levenshtein import distance as lev

BPOM_NUTRIENT_DICTIONARY = {
    'energi': ['energi', 'energy', 'tenaga', 'kalori', 'calories', 'kcal', 'kkal'],
    'lemak_total': ['lemak total', 'total fat', 'lipid', 'total lemak'],
    'lemak_jenuh': ['lemak jenuh', 'saturated fat', 'lemak sat', 'sat fat'],
    'protein': ['protein', 'proteine'],
    'karbohidrat_total': ['karbohidrat total', 'total carbohydrate', 'carbs', 'kohlenhydrate'],
    'gula': ['gula', 'sugars', 'zucker', 'sucre'],
    'natrium': ['natrium', 'sodium', 'garam', 'salt'],
}
NUTRIENT_UNITS = {'energi': 'kkal', 'lemak_total': 'g', 'lemak_jenuh': 'g', 'protein': 'g',
                  'karbohidrat_total': 'g', 'gula': 'g', 'natrium': 'mg'}
REQUIRED_CATEGORIES = list(BPOM_NUTRIENT_DICTIONARY)
MIN_TERM_LENGTH = 3
MIN_LINE_LEN = 3
AMBIG_MARGIN = 5          # selisih skor antar-kategori <= 5 dianggap ambigu (dicatat, bukan ditolak)

ALL_VARIATIONS = [v for vs in BPOM_NUTRIENT_DICTIONARY.values() for v in vs if len(v) >= MIN_TERM_LENGTH]
MAPPING = {v: c for c, vs in BPOM_NUTRIENT_DICTIONARY.items() for v in vs if len(v) >= MIN_TERM_LENGTH}

# ---------------------------------------------------------------- teks & angka
def clean_ocr_text(text):
    text = re.sub(r'[^a-zA-Z0-9\s.,%/-]', '', text)
    return re.sub(r'\s+', ' ', text).strip().lower()

_NUM_RE = re.compile(r'(?<![A-Za-z\d.])(\d+(?:\.\d+)?)\s*(?:(kkal|kcal|kj|mg|g|%)(?![A-Za-z]))?', re.I)
_UNIT_FAMILY = {'kkal': {'kkal', 'kcal'}, 'g': {'g'}, 'mg': {'mg'}}

def extract_numeric_value_naive(text, expected_unit=''):
    """Baseline tanpa normalisasi: angka pertama apa adanya."""
    m = re.search(r'\d+\.?\d*', text)
    return (float(m.group()), expected_unit) if m else (None, None)

def extract_numeric_value_robust(text, expected_unit='', fix_dropped_decimal=True):
    """Normalisasi regex: koma->titik, koreksi O/l di antara digit, abaikan digit yang menempel pada huruf
    (mis. 'Gu1a'), utamakan angka bersatuan sesuai kategori, abaikan %/kJ."""
    t = text.strip().replace(',', '.')
    t = re.sub(r'(\d)[oO](\d)', r'\g<1>0\2', t)
    t = re.sub(r'(\d)[lL](\d)', r'\g<1>1\2', t)
    cands = [(m.group(1), (m.group(2) or '').lower()) for m in _NUM_RE.finditer(t)]
    if not cands:
        return None, None
    fam = _UNIT_FAMILY.get(expected_unit, set())
    pick = next((c for c in cands if c[1] in fam), None) or next((c for c in cands if c[1] not in ('%', 'kj')), None)
    if pick is None:
        return None, None
    try:
        val = float(pick[0])
    except ValueError:
        return None, None
    if fix_dropped_decimal and expected_unit == 'g' and val > 100:   # titik desimal hilang (125 -> 12.5)
        val /= 10
    return val, (pick[1] or expected_unit)

# ---------------------------------------------------------------- grouping baris
def _poly(box):
    try:
        a = np.asarray(box, dtype=float)
        return a if a.ndim == 2 and a.shape[1] >= 2 and len(a) else None
    except Exception:
        return None
def _box_h(box):
    a = _poly(box); return float(a[:, 1].max() - a[:, 1].min()) if a is not None else 0.0
def _x_center(box):
    a = _poly(box); return float(a[:, 0].mean()) if a is not None else 0.0

def group_lines(ocr_data, mode='fixed', fixed_tol=10, ratio=0.5):
    if not ocr_data:
        return []
    tol = fixed_tol
    if mode == 'adaptive':
        hs = [h for h in (_box_h(i['box']) for i in ocr_data) if h > 0]
        if hs:
            tol = ratio * float(np.median(hs))
    lines, grp, cy = [], [], None
    for it in sorted(ocr_data, key=lambda i: i['y_center']):
        if cy is None or abs(it['y_center'] - cy) <= tol:
            grp.append(it); cy = sum(i['y_center'] for i in grp) / len(grp)
        else:
            lines.append(grp); grp, cy = [it], it['y_center']
    if grp:
        lines.append(grp)
    for g in lines:
        g.sort(key=lambda i: _x_center(i['box']))
    return lines

# ---------------------------------------------------------------- pencocokan kategori
def match_category(line_clean, mode='fuzzy', threshold=70):
    """Skor semua istilah kamus terhadap satu baris. Kategori = skor tertinggi; seri -> istilah terpanjang.
    Di bawah threshold -> None (tidak ditebak). Return (kategori, skor, istilah, ambigu) atau None."""
    if len(line_clean) < MIN_LINE_LEN:
        return None
    if mode == 'exact':
        hits = [(t, 100.0) for t in ALL_VARIATIONS if t in line_clean]
    else:
        hits = [(t, float(fuzz.partial_ratio(t, line_clean))) for t in ALL_VARIATIONS]
        hits = [(t, s) for t, s in hits if s >= threshold]
    if not hits:
        return None
    best = {}
    for t, s in hits:
        c = MAPPING[t]
        if c not in best or (s, len(t)) > (best[c][0], len(best[c][1])):
            best[c] = (s, t)
    ranked = sorted(best.items(), key=lambda kv: (kv[1][0], len(kv[1][1])), reverse=True)
    cat, (score, term) = ranked[0]
    ambiguous = len(ranked) > 1 and (score - ranked[1][1][0]) <= AMBIG_MARGIN
    return cat, score, term, ambiguous

def extract_nutrients(ocr_data, match_mode='fuzzy', numeric_mode='robust', threshold=70,
                      line_mode='fixed', fixed_tol=10):
    """ocr_data: list dict {text, score, y_center, box}. Return {kategori: {value, unit, orig, score, term, ambiguous}}"""
    cands = {}
    for grp in group_lines(ocr_data, line_mode, fixed_tol):
        line_text = ' '.join(i['text'] for i in grp)
        line_c = clean_ocr_text(line_text) if numeric_mode == 'robust' else line_text.lower()
        m = match_category(line_c, match_mode, threshold)
        if m is None:
            continue
        cat, score, term, amb = m
        exp = NUTRIENT_UNITS[cat]
        val, unit = (extract_numeric_value_robust(line_text, exp) if numeric_mode == 'robust'
                     else extract_numeric_value_naive(line_text, exp))
        if val is None:
            continue
        if cat not in cands or score > cands[cat]['score']:
            cands[cat] = dict(value=val, unit=unit or exp, orig=line_text.strip(),
                              score=score, term=term, ambiguous=bool(amb))
    return cands

# ---------------------------------------------------------------- evaluasi (level entitas per citra)
def norm_num(x):
    return ('%.6f' % float(x)).rstrip('0').rstrip('.')           # 50.0 -> '50', 1.50 -> '1.5'

def values_match(p, g, rel_tol=0.0, abs_tol=1e-6):
    return abs(p - g) <= max(abs_tol, rel_tol * abs(g))

def evaluate_image(pred, gt, rel_tol=0.0):
    """TP: entitas ada di prediksi & nilai benar | wrong (FP): ada tetapi nilai salah |
    missing (FN): ada di GT tetapi tidak terdeteksi | extra (FP): terdeteksi tetapi tidak ada di GT."""
    detail, terms, terms_det = [], [], []
    tp = wrong = missing = 0
    for k, gv in gt.items():
        try:
            g = float(gv)
        except (TypeError, ValueError):
            continue
        if k in pred:
            try:
                p = float(pred[k]['value']); ok = values_match(p, g, rel_tol)
            except (TypeError, ValueError, KeyError):
                p, ok = None, False
            status = 'tp' if ok else 'wrong'
            tp += ok; wrong += (not ok)
            if p is None:
                term = 1.0
            else:
                gn = norm_num(g); term = min(lev(norm_num(p), gn) / max(len(gn), 1), 1.0)
            terms.append(term); terms_det.append(term)
            detail.append(dict(key=k, status=status, gt=g, pred=p, orig=pred[k].get('orig')))
        else:
            missing += 1; terms.append(1.0)
            detail.append(dict(key=k, status='missing', gt=g, pred=None, orig=None))
    gt_keys = set(gt.keys())
    extra_keys = [k for k in pred if k not in gt_keys]
    for k in extra_keys:
        detail.append(dict(key=k, status='extra', gt=None, pred=pred[k].get('value'), orig=pred[k].get('orig')))
    return dict(tp=tp, wrong=wrong, missing=missing, extra=len(extra_keys), n_gt=tp + wrong + missing,
                terms=terms, terms_det=terms_det, detail=detail)

def aggregate(per_image):
    S = lambda k: int(sum(e[k] for e in per_image))
    TP, WR, MI, EX, N_GT = S('tp'), S('wrong'), S('missing'), S('extra'), S('n_gt')
    FP, FN = WR + EX, MI
    P = TP / (TP + FP) if TP + FP else 0.0
    R = TP / (TP + FN) if TP + FN else 0.0
    F1 = 2 * P * R / (P + R) if P + R else 0.0
    def prf(e):
        fp = e['wrong'] + e['extra']
        p = e['tp'] / (e['tp'] + fp) if e['tp'] + fp else 0.0
        r = e['tp'] / (e['tp'] + e['missing']) if e['tp'] + e['missing'] else 0.0
        return p, r, (2 * p * r / (p + r) if p + r else 0.0)
    mP, mR, mF = np.array([prf(e) for e in per_image]).mean(axis=0)
    terms = [t for e in per_image for t in e['terms']]
    terms_det = [t for e in per_image for t in e['terms_det']]
    return dict(
        n_images=len(per_image), TP=TP, WRONG=WR, MISSING=MI, EXTRA=EX, N_GT=N_GT, FP=FP, FN=FN,
        precision=P, recall=R, f1=F1,
        macro_precision=float(mP), macro_recall=float(mR), macro_f1=float(mF),
        harmonic_of_macro=float(2 * mP * mR / (mP + mR)) if mP + mR else 0.0,
        nter_micro=float(np.mean(terms)) if terms else 0.0,
        nter_detected_only=float(np.mean(terms_det)) if terms_det else 0.0,
        nter_image_macro=float(np.mean([np.mean(e['terms']) if e['terms'] else 0.0 for e in per_image])),
        fully_correct_images=int(sum(1 for e in per_image if e['n_gt'] > 0 and e['wrong'] == 0 and e['missing'] == 0)),
    )
