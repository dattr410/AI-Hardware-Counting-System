import os
# TCP cho RTSP: tranh mat goi UDP gay vo hinh/nhay frame. Phai dat truoc khi import cv2.
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")

import cv2
import json
import math
import shutil
import time
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from threading import Lock, Thread
from urllib.parse import quote

from flask import Flask, Response, jsonify, redirect, render_template, send_file
try:
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont   # ve chu tieng Viet co dau (cv2.putText khong ho tro)
except ImportError:                               # thieu Pillow -> tu dong dung chu khong dau
    np = Image = ImageDraw = ImageFont = None
try:
    from scipy.optimize import linear_sum_assignment   # di kem Ultralytics
except ImportError:
    linear_sum_assignment = None
from ultralytics import YOLO

app = Flask(__name__)


def _env_bool(name, default):
    v = os.getenv(name)
    return default if v is None else v.strip().lower() in ("1", "true", "yes", "on")


def _env_float(name, default):
    try:
        return float(os.getenv(name, default))
    except ValueError:
        return default


def _env_int_list(name, default):
    v = os.getenv(name)
    if v is None:
        return default
    try:
        return [int(x) for x in v.split(",") if x.strip()]
    except ValueError:
        return default


def log(msg):
    try:
        print(msg, flush=True)
    except UnicodeEncodeError:   # console Windows cu khong in duoc tieng Viet co dau
        print(str(msg).encode("ascii", "replace").decode("ascii"), flush=True)


# ==================================================
# 1. CAU HINH
# ==================================================
MODEL_PATH = Path(__file__).resolve().parent / "best.pt"

CAMERA_IP = os.getenv("CAMERA_IP", "192.168.1.233")
CAMERA_USERNAME = os.getenv("CAMERA_USERNAME", "admin")
# KHONG hard-code mat khau trong source. Dat bang: setx CAMERA_PASSWORD "..."  (roi mo lai terminal)
CAMERA_PASSWORD = os.getenv("CAMERA_PASSWORD", "") # điền mật khẩu camera vào biến môi trường CAMERA_PASSWORD
RTSP_PORT = int(os.getenv("RTSP_PORT", "554"))
RTSP_URL = (
    f"rtsp://{quote(CAMERA_USERNAME, safe='')}:{quote(CAMERA_PASSWORD, safe='')}"
    f"@{CAMERA_IP}:{RTSP_PORT}/Streaming/Channels/101"
)
RTSP_RECONNECT_DELAY = 2.0
PREVIEW_FPS = 20.0
PREVIEW_MAX_W = 1280                  # preview thu nho truoc khi encode JPEG -> do ton CPU cua luong doc camera

# v4 la profile mac dinh (on dinh hoa). v3 = hanh vi cu, giu lai de so sanh A/B: COUNT_PROFILE=v3
PROFILE = os.getenv("COUNT_PROFILE", "v4").strip().lower()
DEBUG = _env_bool("COUNT_DEBUG", False)

CONFIDENCE = 0.5                      # nguong "da xac nhan" tu dong
CONF_LOW = float(os.getenv("CONF_LOW", "0.30"))   # nguong suy luan cua profile v2/v3/v4
RECOVER_MIN_OBS = 3                   # track conf thap can xuat hien >= 3/5 frame
RECOVER_MIN_MEAN = 0.40               # va conf trung binh >= 0.40
IMAGE_SIZE = int(os.getenv("IMAGE_SIZE", "1600"))
USE_PREPROCESS = _env_bool("USE_PREPROCESS", True)
AGNOSTIC_NMS = _env_bool("AGNOSTIC_NMS", False)
NMS_IOU = float(os.getenv("NMS_IOU", "0.5"))       # chi co tac dung khi model chay che do NMS
SCALES = _env_int_list("SCALES", [1280, 1600, 1920, 1600, 1280])   # CHI v3: imgsz moi frame
SCALES_V4 = _env_int_list("SCALES_V4", [])          # v4: rong = 1 imgsz co dinh (IMAGE_SIZE) cho moi frame
REFINE = _env_bool("REFINE", True)

USE_ROI = True
ROI_X1, ROI_Y1, ROI_X2, ROI_Y2 = 0.05, 0.10, 0.95, 0.85

USE_SHAPE_CHECK = True
SHAPE_DEBUG = True
SHAPE_CORRECT = _env_bool("SHAPE_CORRECT", False)  # chi ap dung cho profile v2/v3/v4
SHAPE_MIN_SIZE = 18
DEDUP_IOU = 0.65                                   # dedup cu (baseline / v2)

# ---- geometry-aware dedup (can hieu chinh theo du lieu that) ----
DUP_IOU_SAME = 0.85     # cung class: chi coi la trung khi IoU >= nguong nay ...
DUP_CDIST = 0.08        # ... VA khoang cach tam / duong cheo nho nhat <= nguong nay ...
DUP_SIZE = 0.80         # ... VA ti le dien tich nho/lon >= nguong nay
DUP_IOU_CROSS = 0.80    # khac class cung 1 vat (vd nut/washer nham nhau)
FILL_X_MAX = 0.45       # minAreaRect fill < nguong -> hinh chu X / nhieu vat (phu quyet dedup)
OVERLAP_VETO = _env_bool("OVERLAP_VETO", True)   # tat = near-identical same-class luon bi coi la trung

# ---- ghep frame bang Hungarian ----
MATCH_MAX_COST = 0.90
_BIG = 1e6
# v4: vat dung yen -> cong ghep chat hon v3 (v3 chi chan khi iou<0.15 va cd>0.35)
MATCH_MIN_IOU = 0.30          # cung class
MATCH_MIN_IOU_CROSS = 0.50    # khac class: chi ghep khi gan nhu cung 1 vat (nut<->washer lat class)
MATCH_MIN_SIZE = 0.40         # ti le dien tich nho/lon toi thieu
MIN_CONFIRM_FRAC = 0.60       # v4: "da xac nhan" can thay >= 60% so frame (5 frame -> 3)

# ---- re-inference vung nghi ngo ----
REFINE_IMGSZ = 960
MAX_REFINE = 6
SUSPECT_AREA_K = 1.6    # box screw > 1.6 x trung vi dien tich screw cua phien -> nghi gop

COUNTDOWN_SECONDS = 3
SAMPLE_COUNT = 5
SAMPLE_INTERVAL = 0.15
IOU_THRESHOLD = 0.30

# ---- v4: thu frame on dinh ----
SETTLE_SECONDS = _env_float("SETTLE_SECONDS", 0.4)   # bo qua frame ngay sau countdown (decoder/AE on dinh)
SAMPLE_TIMEOUT = _env_float("SAMPLE_TIMEOUT", 8.0)   # qua thoi gian nay thi khong cho canh yen nua
MOTION_MAX = _env_float("MOTION_MAX", 3.0)           # |diff| TB (muc xam, anh 320px) giua 2 frame lien tiep -> dang chuyen dong
EXPOSURE_DRIFT = _env_float("EXPOSURE_DRIFT", 0.03)  # do sang TB lech > 3% giua cac frame -> canh bao

SAVE_DIR = Path(__file__).resolve().parent / "static" / "saved_images"
SAVE_DIR.mkdir(parents=True, exist_ok=True)
REVIEW_DIR = SAVE_DIR / "review"
REVIEW_DIR.mkdir(parents=True, exist_ok=True)
DEBUG_DIR = SAVE_DIR / "debug"
REVIEW_CONFIDENCE = 0.70
REVIEW_COUNT_SPREAD = 2
REVIEW_ON_SHAPE_DISAGREEMENT = True
CLASS_SHARE_REVIEW = 0.80
file_lock = Lock()

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Khong tim thay model: {MODEL_PATH}")

model = YOLO(str(MODEL_PATH))
lock = Lock()


def detect_e2e_kwargs():
    """Tham so ep che do one-to-one (NMS-free) cua YOLO26.
    SUA: ban cu thu predict(nms=False) roi predict(end2end=True) tren anh den; nhung `nms` la tham so EXPORT
    hop le trong cfg Ultralytics nen predict(nms=False) KHONG nem loi -> ham luon tra {'nms': False}
    du no khong lam gi o predict. Ban moi doc DEFAULT_CFG_DICT: chi tra {'end2end': True} khi key do ton tai,
    nguoc lai tra {} (dung mac dinh cua model; YOLO26 .pt mac dinh da la head end2end). Xem log [BOOT]."""
    info = {}
    keys = set()
    try:
        import ultralytics
        from ultralytics.cfg import DEFAULT_CFG_DICT
        info["ultralytics"] = getattr(ultralytics, "__version__", "?")
        keys = set(DEFAULT_CFG_DICT)
    except Exception:
        pass
    try:
        info["head_end2end"] = getattr(model.model.model[-1], "end2end", None)
    except Exception:
        info["head_end2end"] = None
    kw = {"end2end": True} if "end2end" in keys else {}
    info["end2end_in_cfg"] = "end2end" in keys
    info["nms_in_cfg(export-only)"] = "nms" in keys
    return kw, info


E2E_KW, E2E_INFO = detect_e2e_kwargs()

camera_enabled = False
camera_ready = False
camera_thread = None
camera_status = "Đã dừng"

session_state = "idle"
session_message = "Sẵn sàng đếm"
session_id = 0
countdown_deadline = 0.0

latest_frame = None
latest_jpeg = None

latest_counts = {}
latest_total = 0
latest_detections = 0
latest_recovered = 0
latest_result_path = None
latest_result_filename = None
latest_raw_path = None
latest_review_frame = None
latest_review_metadata = None
latest_review_reasons = []
latest_report = None

sample_frames = []
sample_meta = []
sample_info = {"restarts": 0, "duplicates": 0, "flags": []}
sampling_t0 = 0.0
prev_sig = None
last_sample_time = 0.0


def make_cfg(profile=None, **override):
    """baseline = hanh vi cu; v2 = nguong thap + xac nhan theo track;
    v3 = v2 + geo-dedup + NMS-free + da scale + Hungarian + refine;
    v4 = v3 + imgsz co dinh + cong ghep chat + xac nhan theo SO FRAME (khong theo max conf)
         + refine dong thuan nhieu frame + trace day du."""
    profile = (profile or PROFILE).lower()
    cfg = dict(name=profile, conf_infer=CONFIDENCE, recover=False,
               shape_correct=True, legacy_merge=True,
               preprocess=USE_PREPROCESS, imgsz=IMAGE_SIZE, agnostic=AGNOSTIC_NMS,
               nms_iou=NMS_IOU, geo=False, nms_free=False, scales=None,
               hungarian=False, refine=False, stable=False)
    if profile != "baseline":
        cfg.update(conf_infer=min(CONFIDENCE, CONF_LOW), recover=True,
                   shape_correct=SHAPE_CORRECT, legacy_merge=False)
    if profile == "v3":
        cfg.update(geo=True, nms_free=True, scales=(SCALES or None),
                   hungarian=True, refine=REFINE)
    if profile == "v4":
        cfg.update(geo=True, nms_free=True, scales=(SCALES_V4 or None),
                   hungarian=True, refine=REFINE, stable=True)
    cfg.update(override)
    return cfg


def cname(class_id):
    return str(model.names[class_id]).lower()


def brief(d):
    out = {"class": cname(d["class_id"]), "conf": round(d["confidence"], 3),
           "box": [int(round(v)) for v in d["box"]]}
    if "id" in d:
        out["id"] = d["id"]
    if "frame" in d:
        out["frame"] = d["frame"] + 1
    if "imgsz" in d:
        out["imgsz"] = d["imgsz"]
    return out


def by_class(dets):
    return dict(Counter(cname(d["class_id"]) for d in dets))


# ==================================================
# 2. XU LY ANH
# ==================================================
def calculate_iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    x1, y1 = max(ax1, bx1), max(ay1, by1)
    x2, y2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = max(0, ax2 - ax1) * max(0, ay2 - ay1) + max(0, bx2 - bx1) * max(0, by2 - by1) - inter
    return inter / union if union > 0 else 0.0


def calculate_sharpness(frame):
    """(legacy) Laplacian var tren vung giua 70% anh - KHONG trung ROI that, va bi nhieu/nen tang diem."""
    h, w = frame.shape[:2]
    roi = frame[int(h * .15):int(h * .85), int(w * .15):int(w * .85)]
    gray = cv2.GaussianBlur(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY), (3, 3), 0)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def select_best_frame(frames):
    """(legacy v2/v3)"""
    scores = [calculate_sharpness(f) for f in frames]
    idx = max(range(len(scores)), key=lambda i: scores[i])
    log(f"[FRAME SELECT] scores={[round(v, 2) for v in scores]} selected={idx + 1}/{len(frames)}")
    return frames[idx], idx, scores


# ---------- v4: chat luong frame + phat hien frame lap / chuyen dong ----------
def frame_quality(frame):
    """Chat luong tren DUNG vung ROI: do net (Laplacian tren anh thu nho 0.5x de bot nhay cam voi nhieu nen),
    do sang, tuong phan, ti le chay sang/chay toi."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = get_roi_px(w, h)
    g = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
    small = cv2.resize(g, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    return {"sharp": float(cv2.Laplacian(small, cv2.CV_64F).var()),
            "brightness": float(g.mean()), "contrast": float(g.std()),
            "clip_hi": float((g >= 250).mean()), "clip_lo": float((g <= 5).mean())}


def frame_signature(frame):
    """Anh xam ROI thu nho ~320px dung de so sanh frame (lap / chuyen dong)."""
    if np is None:
        return None
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = get_roi_px(w, h)
    g = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, (320, max(1, int(320 * g.shape[0] / g.shape[1]))), interpolation=cv2.INTER_AREA)
    return g.astype(np.float32)


def signature_diff(a, b):
    return float(np.mean(np.abs(a - b)))


def select_best_frame_v4(frames, quality):
    """Diem = do net (chuan hoa theo trung vi) - phat do lech sang - phat chay sang/toi.
    Frame nay CHI de hien thi / shape-check; moi quyet dinh dem deu dung bang chung cua TAT CA frame."""
    med_s = sorted(q["sharp"] for q in quality)[len(quality) // 2]
    med_b = sorted(q["brightness"] for q in quality)[len(quality) // 2]
    scores = []
    for q in quality:
        pen = abs(q["brightness"] - med_b) / max(med_b, 1.0) + 5.0 * (q["clip_hi"] + q["clip_lo"])
        scores.append(q["sharp"] / max(med_s, 1e-6) - pen)
    idx = max(range(len(scores)), key=lambda i: (round(scores[i], 3), -i))
    log(f"[FRAME SELECT v4] quality={[round(v, 3) for v in scores]} selected={idx + 1}/{len(frames)}")
    return frames[idx], idx, scores


def get_detections(result):
    detections = []
    if result.boxes is None:
        return detections
    for box in result.boxes:
        detections.append({
            "box": [float(v) for v in box.xyxy[0].cpu().numpy().tolist()],
            "class_id": int(box.cls[0].item()),
            "confidence": float(box.conf[0].item()),
        })
    return detections


def preprocess_frame(frame):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=1.6, tileGridSize=(8, 8)).apply(l)
    improved = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
    blur = cv2.GaussianBlur(improved, (0, 0), 0.7)
    return cv2.addWeighted(improved, 1.25, blur, -0.25, 0)


def get_roi_px(w, h):
    """Mot ham duy nhat cho ROI: dung chung cho inference, preview va bao cao."""
    if not USE_ROI:
        return 0, 0, w, h
    x1 = max(0, min(w - 1, int(round(w * ROI_X1))))
    y1 = max(0, min(h - 1, int(round(h * ROI_Y1))))
    x2 = max(x1 + 1, min(w, int(round(w * ROI_X2))))
    y2 = max(y1 + 1, min(h, int(round(h * ROI_Y2))))
    return x1, y1, x2, y2


def deduplicate(detections, threshold=DEDUP_IOU, log_list=None, stage="dedup"):
    """(baseline/v2) Greedy NMS khong phan biet class, ghi lai box bi loai + ly do."""
    ordered = sorted(detections, key=lambda d: d["confidence"], reverse=True)
    kept = []
    for item in ordered:
        clash = None
        for old in kept:
            iou = calculate_iou(item["box"], old["box"])
            if iou >= threshold:
                clash = (old, iou)
                break
        if clash is None:
            kept.append(item)
        elif log_list is not None:
            log_list.append({"stage": stage, "reason": f"iou={clash[1]:.2f}>={threshold}",
                             "dropped": brief(item), "kept": brief(clash[0])})
    return kept


# ---------- geometry-aware dedup ----------
def pair_metrics(a, b):
    """(iou, khoang cach tam chuan hoa theo duong cheo nho nhat, ti le dien tich nho/lon)."""
    iou = calculate_iou(a, b)
    aw, ah, bw, bh = a[2] - a[0], a[3] - a[1], b[2] - b[0], b[3] - b[1]
    dist = math.hypot((a[0] + a[2] - b[0] - b[2]) / 2.0, (a[1] + a[3] - b[1] - b[3]) / 2.0)
    diag = max(min(math.hypot(aw, ah), math.hypot(bw, bh)), 1e-6)
    size = min(aw * ah, bw * bh) / max(aw * ah, bw * bh, 1e-6)
    return iou, dist / diag, size


def looks_like_two_objects(frame, a, b):
    """Vung hop cua 2 box co hinh chu X / nhieu vat khong? (minAreaRect fill thap).
    CHI dung lam 'phu quyet dedup' cho cap box gan nhu trung nhau, nen sai so nghieng ve giu lai
    (va phien dem bi gan co review 'kept_overlap_pair'). v4: cluster co co nay chi duoc la 'cho xac minh'."""
    if frame is None or np is None:
        return False
    h, w = frame.shape[:2]
    x1, y1 = max(0, int(min(a[0], b[0]))), max(0, int(min(a[1], b[1])))
    x2, y2 = min(w, int(max(a[2], b[2]))), min(h, int(max(a[3], b[3])))
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0 or min(roi.shape[:2]) < 12:
        return False
    gray = cv2.GaussianBlur(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY), (3, 3), 0)
    _, m = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if m.mean() / 255.0 > 0.5:            # vat the la phan nho hon cua ROI
        m = 255 - m
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return False
    c = max(cnts, key=cv2.contourArea)
    (_, _), (rw, rh), _ = cv2.minAreaRect(c)
    fill = cv2.contourArea(c) / max(rw * rh, 1.0)
    return fill < FILL_X_MAX              # 1 vit don: fill cao; chu X: fill thap


def geo_dedup(dets, frame=None, log_list=None, stage="geo_dedup"):
    """Chi loai box THUC SU trung (gan nhu giong het). Hai box chi giao nhau -> giu ca hai."""
    kept = []
    for d in sorted(dets, key=lambda x: x["confidence"], reverse=True):
        drop = None
        for k in kept:
            iou, cd, size = pair_metrics(d["box"], k["box"])
            if d["class_id"] == k["class_id"]:
                dup = iou >= DUP_IOU_SAME and cd <= DUP_CDIST and size >= DUP_SIZE
                if dup and OVERLAP_VETO and looks_like_two_objects(frame, d["box"], k["box"]):
                    dup = False
                    d["kept_overlap_pair"] = True
                    k["kept_overlap_pair"] = True
            else:
                dup = iou >= DUP_IOU_CROSS          # cung 1 vat bi gan 2 class
            if dup:
                drop = (k, iou)
                break
        if drop is None:
            kept.append(d)
        elif log_list is not None:
            log_list.append({"stage": stage, "reason": f"near_identical iou={drop[1]:.2f}",
                             "dropped": brief(d), "kept": brief(drop[0])})
    return kept


def predict_frame(frame, cfg, dropped, frame_index=0):
    t0 = time.perf_counter()
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = get_roi_px(w, h)
    crop = frame[y1:y2, x1:x2]
    source = preprocess_frame(crop) if cfg["preprocess"] else crop
    scales = cfg.get("scales")
    imgsz = scales[frame_index % len(scales)] if scales else cfg["imgsz"]
    kw = dict(source=source, conf=cfg["conf_infer"], imgsz=imgsz,
              agnostic_nms=cfg["agnostic"], iou=cfg["nms_iou"], max_det=300, verbose=False)
    if cfg.get("nms_free"):
        kw.update(E2E_KW)
    result = model.predict(**kw)[0]
    raw = get_detections(result)
    # thu tu on dinh -> det id on dinh giua cac lan chay
    raw.sort(key=lambda d: (-d["confidence"], d["box"][1], d["box"][0]))
    for k, det in enumerate(raw, 1):  # crop -> frame goc
        bx1, by1, bx2, by2 = det["box"]
        det["box"] = [bx1 + x1, by1 + y1, bx2 + x1, by2 + y1]
        det["id"], det["frame"], det["imgsz"] = f"F{frame_index + 1}-{k}", frame_index, imgsz
    local_drop = []
    if cfg.get("geo"):
        kept = geo_dedup(raw, frame, local_drop, f"frame{frame_index + 1}_geo")
    else:
        kept = deduplicate(raw, log_list=local_drop, stage=f"frame{frame_index + 1}_dedup")
    dropped.extend(local_drop)
    info = {"raw": len(raw), "kept": len(kept), "imgsz": imgsz,
            "raw_by_class": by_class(raw), "kept_by_class": by_class(kept),
            "low_conf": sum(1 for d in kept if d["confidence"] < CONFIDENCE),
            "t": round(time.perf_counter() - t0, 3),
            "dets": [brief(d) for d in kept] if (DEBUG or cfg.get("stable")) else None}
    return kept, info


def shape_evidence(frame, box):
    """Bang chung hinh dang nut/washer; None neu phan doan khong dang tin."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    pad = max(3, int(max(x2 - x1, y2 - y1) * 0.18))
    x1, y1, x2, y2 = max(0, x1 - pad), max(0, y1 - pad), min(w, x2 + pad), min(h, y2 + pad)
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0 or min(roi.shape[:2]) < SHAPE_MIN_SIZE:
        return None
    gray = cv2.GaussianBlur(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY), (3, 3), 0)
    candidates = []
    for polarity in (cv2.THRESH_BINARY, cv2.THRESH_BINARY_INV):
        _, mask = cv2.threshold(gray, 0, 255, polarity | cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            roi_area = roi.shape[0] * roi.shape[1]
            if not 0.08 * roi_area < area < 0.78 * roi_area:
                continue
            bx, by, bw, bh = cv2.boundingRect(contour)
            if bx <= 1 or by <= 1 or bx + bw >= roi.shape[1] - 1 or by + bh >= roi.shape[0] - 1:
                continue
            perimeter = cv2.arcLength(contour, True)
            if perimeter <= 0:
                continue
            circularity = 4 * math.pi * area / (perimeter * perimeter)
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area else 0
            approx = cv2.approxPolyDP(hull, 0.035 * cv2.arcLength(hull, True), True)
            if not 0.78 <= solidity <= 1.02:
                continue
            candidates.append((area / roi_area, circularity, len(approx)))
    if not candidates:
        return None
    _, circularity, vertices = max(candidates)
    if 5 <= vertices <= 7 and 0.78 <= circularity <= 0.94:
        return "nut", circularity, vertices
    if vertices >= 8 and circularity >= 0.95:
        return "washer", circularity, vertices
    return None


def apply_shape_check(frame, detections, cfg):
    """Luon tinh bang chung hinh dang (de canh bao/review).
    Chi DOI class khi cfg['shape_correct'] bat (baseline) hoac SHAPE_CORRECT=1."""
    corrected = []
    class_lookup = {str(v).lower(): int(k) for k, v in model.names.items()}
    for index, item in enumerate(detections, start=1):
        d = dict(item)
        name = cname(d["class_id"])
        d["original_class"] = name
        d["shape"] = None
        if USE_SHAPE_CHECK and name in ("nut", "washer"):
            evidence = shape_evidence(frame, d["box"])
            if SHAPE_DEBUG:
                log(f"[SHAPE DEBUG] #{index} YOLO={name} conf={d['confidence']:.3f} "
                    f"box={[round(v) for v in d['box']]} OpenCV={evidence}")
            if evidence is not None:
                shape_class, circularity, vertices = evidence
                d["shape"] = {"class": shape_class, "circularity": round(circularity, 3),
                              "vertices": vertices}
                if shape_class != name:
                    strong = ((shape_class == "nut" and circularity <= 0.90 and vertices == 6)
                              or (shape_class == "washer" and circularity >= 0.97 and vertices >= 9))
                    if strong and cfg["shape_correct"] and shape_class in class_lookup:
                        d["class_id"] = class_lookup[shape_class]
                        d["shape_changed_class"] = True
                        log(f"[SHAPE CORRECTION] #{index}: {name} -> {shape_class}")
                    elif strong:
                        log(f"[SHAPE SUGGEST] #{index}: {name} -> {shape_class} (khong ap dung, chi canh bao)")
        corrected.append(d)
    return corrected


# ==================================================
# 3. GHEP 5 FRAME
# ==================================================
def merge_detections_legacy(frame_detections, dropped):
    """Giu nguyen logic cu (baseline). Luu y: nguong 'conf<0.60' o day khong con y nghia
    vi CONFIDENCE hien la 0.5 (moi detection da >= nguong infer)."""
    tracks = []
    for frame_index, detections in enumerate(frame_detections):
        used = set()
        for det in sorted(detections, key=lambda d: d["confidence"], reverse=True):
            best_idx, best_iou = None, 0.0
            for i, track in enumerate(tracks):
                if i in used or frame_index - track["last_frame"] > 2:
                    continue
                iou = calculate_iou(det["box"], track["last_box"])
                if iou > best_iou:
                    best_idx, best_iou = i, iou
            if best_idx is not None and best_iou >= IOU_THRESHOLD:
                track = tracks[best_idx]
                track["observations"].append(det)
                track["last_box"], track["last_frame"] = det["box"], frame_index
                used.add(best_idx)
            else:
                tracks.append({"observations": [det], "last_box": det["box"], "last_frame": frame_index})
                used.add(len(tracks) - 1)
    merged = []
    for track in tracks:
        obs = track["observations"]
        if len(obs) < 2 and max(d["confidence"] for d in obs) < 0.60:
            dropped.append({"stage": "merge", "reason": "single_obs_low_conf", "dropped": brief(obs[0])})
            continue
        merged.append(_finalize_track(obs, "confirmed"))
    return deduplicate(merged, log_list=dropped, stage="post_merge_dedup")


def merge_detections_v2(frame_detections, cfg, dropped):
    """Matching toan cuc theo IoU (cap IoU cao nhat truoc). Track xac nhan neu:
      - max conf >= CONFIDENCE, HOAC
      - xuat hien >= RECOVER_MIN_OBS frame va conf TB >= RECOVER_MIN_MEAN (status=recovered)."""
    tracks = []
    for fi, dets in enumerate(frame_detections):
        pairs = []
        for di, d in enumerate(dets):
            for ti, t in enumerate(tracks):
                if t["last_frame"] == fi or fi - t["last_frame"] > 2:
                    continue
                iou = calculate_iou(d["box"], t["last_box"])
                if iou >= IOU_THRESHOLD:
                    pairs.append((iou, di, ti))
        pairs.sort(reverse=True)
        used_d, used_t = set(), set()
        for iou, di, ti in pairs:
            if di in used_d or ti in used_t:
                continue
            tracks[ti]["observations"].append(dets[di])
            tracks[ti]["last_box"], tracks[ti]["last_frame"] = dets[di]["box"], fi
            used_d.add(di)
            used_t.add(ti)
        for di, d in enumerate(dets):
            if di not in used_d:
                tracks.append({"observations": [d], "last_box": d["box"], "last_frame": fi})
    merged = []
    for t in tracks:
        obs = t["observations"]
        confs = [d["confidence"] for d in obs]
        mx, mean = max(confs), sum(confs) / len(confs)
        if mx >= CONFIDENCE:
            merged.append(_finalize_track(obs, "confirmed"))
        elif cfg["recover"] and len(obs) >= RECOVER_MIN_OBS and mean >= RECOVER_MIN_MEAN:
            merged.append(_finalize_track(obs, "recovered"))
        else:
            dropped.append({"stage": "merge",
                            "reason": f"unconfirmed(n={len(obs)},max={mx:.2f},mean={mean:.2f})",
                            "dropped": brief(max(obs, key=lambda d: d["confidence"]))})
    return deduplicate(merged, log_list=dropped, stage="post_merge_dedup")


# ---------- v3: Hungarian + chi phi hinh hoc (GIU NGUYEN de so sanh A/B) ----------
def match_cost(det, ref):
    """Chi phi ghep 1 detection voi 1 cum (box tham chieu = trung binh). None = khong the ghep.
    Class khac chi bi PHAT mem (khong chan cung)."""
    iou, cd, size = pair_metrics(det["box"], ref["box"])
    if iou < 0.15 and cd > 0.35:
        return None
    cls_pen = 0.0 if det["class_id"] == ref["class_id"] else 0.4
    return (1 - iou) + 0.5 * cd + 0.3 * (1 - size) + cls_pen


def _assign(C):
    """Tra ve list (hang, cot). Dung scipy neu co, neu khong thi greedy theo chi phi tang dan."""
    if linear_sum_assignment is not None:
        r, c = linear_sum_assignment(C)
        return list(zip(r.tolist(), c.tolist()))
    cells = sorted((C[i, j], i, j) for i in range(C.shape[0]) for j in range(C.shape[1]))
    used_r, used_c, out = set(), set(), []
    for _, i, j in cells:
        if i not in used_r and j not in used_c:
            used_r.add(i)
            used_c.add(j)
            out.append((i, j))
    return out


def merge_detections_v3(frame_detections, cfg, dropped, frame=None):
    """Vat dung yen -> cum theo vi tri. Moi frame ghep MOT-MOT (Hungarian) nen hai vit cung frame
    khong bao gio bi gop thanh mot. Doi cho 2 box gan trung nhau khong lam sai so dem."""
    clusters = []
    for fi, dets in enumerate(frame_detections):
        matched = {}
        if clusters and dets and np is not None:
            C = np.full((len(dets), len(clusters)), _BIG)
            for i, d in enumerate(dets):
                for j, c in enumerate(clusters):
                    cost = match_cost(d, c)
                    if cost is not None and cost <= MATCH_MAX_COST:
                        C[i, j] = cost
            for i, j in _assign(C):
                if C[i, j] < _BIG:
                    matched[i] = j
        for i, d in enumerate(dets):
            if i in matched:
                c = clusters[matched[i]]
                c["obs"].append(d)
                n = len(c["obs"])
                c["box"] = [sum(o["box"][k] for o in c["obs"]) / n for k in range(4)]
                c["class_id"] = max(c["obs"], key=lambda o: o["confidence"])["class_id"]
            else:
                clusters.append({"obs": [d], "box": list(d["box"]), "class_id": d["class_id"]})
    merged = []
    for c in clusters:
        obs = c["obs"]
        confs = [o["confidence"] for o in obs]
        mx, mean = max(confs), sum(confs) / len(confs)
        if mx >= CONFIDENCE:
            merged.append(_finalize_track(obs, "confirmed"))
        elif cfg["recover"] and len(obs) >= RECOVER_MIN_OBS and mean >= RECOVER_MIN_MEAN:
            merged.append(_finalize_track(obs, "recovered"))
        else:
            dropped.append({"stage": "merge",
                            "reason": f"unconfirmed(n={len(obs)},max={mx:.2f},mean={mean:.2f})",
                            "dropped": brief(max(obs, key=lambda o: o["confidence"]))})
    return geo_dedup(merged, frame, dropped, "post_merge_geo")


def _finalize_track(obs, status):
    votes = defaultdict(float)
    for d in obs:
        votes[d["class_id"]] += d["confidence"]
    chosen = max(votes, key=lambda k: (votes[k], -k))
    selected = [d for d in obs if d["class_id"] == chosen]
    return {
        "box": [sum(d["box"][j] for d in selected) / len(selected) for j in range(4)],
        "class_id": chosen,
        "confidence": sum(d["confidence"] for d in selected) / len(selected),
        "observations": len(obs),
        "status": status,
        "class_share": votes[chosen] / sum(votes.values()),
        "votes": {cname(k): round(v, 3) for k, v in votes.items()},
        "kept_overlap_pair": any(d.get("kept_overlap_pair") for d in obs),
        "frame_idx": sorted({d["frame"] for d in obs if d.get("frame") is not None}),
    }


# ---------- v4: ghep cho vat dung yen + xac nhan theo so frame ----------
def match_cost_v4(det, ref):
    """Vat dung yen => cung 1 vat thi box gan nhu trung nhau o moi frame. Cong cung:
    - dien tich phai tuong duong (>= MATCH_MIN_SIZE)
    - cung class: IoU >= MATCH_MIN_IOU ; khac class: IoU >= MATCH_MIN_IOU_CROSS (gan nhu cung 1 vat,
      chi la model lat nut<->washer). Hai vat khac class ke nhau co IoU thap nen khong bi ghep."""
    iou, cd, size = pair_metrics(det["box"], ref["box"])
    same = det["class_id"] == ref["class_id"]
    if size < MATCH_MIN_SIZE or iou < (MATCH_MIN_IOU if same else MATCH_MIN_IOU_CROSS):
        return None
    return (1 - iou) + 0.5 * cd + 0.3 * (1 - size) + (0.0 if same else 0.4)


def classify_cluster(obs, nframes):
    """Quyet dinh trang thai tu BANG CHUNG NHIEU FRAME (v3 dung max conf cua 1 frame bat ky -> 1 box ma
    co conf 0.5 o dung 1 frame cung 'da xac nhan'; con vat that conf~0.45 o ca 5 frame thi bi loai)."""
    n = len({o.get("frame") for o in obs})
    confs = [o["confidence"] for o in obs]
    mx, mean = max(confs), sum(confs) / len(confs)
    need = max(2, math.ceil(round(MIN_CONFIRM_FRAC * nframes, 6)))
    if n >= need and mean >= CONFIDENCE:
        return "confirmed", f"seen {n}/{nframes} frames (>= {need}), mean_conf={mean:.2f} >= {CONFIDENCE}"
    if n >= RECOVER_MIN_OBS and mean >= RECOVER_MIN_MEAN:
        return "recovered", f"stable_low_conf: seen {n}/{nframes}, mean_conf={mean:.2f} in [{RECOVER_MIN_MEAN},{CONFIDENCE})"
    if mx >= CONFIDENCE:
        return "recovered", f"weak_support: seen only {n}/{nframes} frames, max_conf={mx:.2f}"
    return None, f"unconfirmed: seen {n}/{nframes}, max_conf={mx:.2f}<{CONFIDENCE}, mean={mean:.2f}"


def merge_detections_v4(frame_dets, cfg, dropped, nframes):
    """Tra ve (merged, trace). Hungarian 1-1 moi frame (cluster khong nhan 2 detection cung frame),
    cost matrix cung cho vat dung yen, trang thai theo so frame, va dedup sau ghep dua tren
    'dong xuat hien': 2 cluster tung xuat hien CUNG 1 frame la 2 vat khac nhau (detector da thay ca hai)."""
    clusters, matching = [], []
    for fi, dets in enumerate(frame_dets):
        dets = sorted(dets, key=lambda d: ((d["box"][1] + d["box"][3]) / 2, (d["box"][0] + d["box"][2]) / 2))
        n_before = len(clusters)
        matched, C = {}, None
        if n_before and dets and np is not None:
            C = np.full((len(dets), n_before), _BIG)
            for i, d in enumerate(dets):
                for j, c in enumerate(clusters):
                    cost = match_cost_v4(d, c)
                    if cost is not None and cost <= MATCH_MAX_COST:
                        C[i, j] = cost
            for i, j in _assign(C):
                if C[i, j] < _BIG:
                    matched[i] = j
        flog = {"frame": fi + 1, "matches": [], "new_clusters": [], "missed_clusters": []}
        seen = set()
        for i, d in enumerate(dets):
            if i in matched:
                j = matched[i]
                c = clusters[j]
                c["obs"].append(d)
                n = len(c["obs"])
                c["box"] = [sum(o["box"][k] for o in c["obs"]) / n for k in range(4)]
                votes = defaultdict(float)
                for o in c["obs"]:
                    votes[o["class_id"]] += o["confidence"]
                c["class_id"] = max(votes, key=lambda k: (votes[k], -k))
                seen.add(j)
                flog["matches"].append({"det": d.get("id"), "cluster": c["id"], "cost": round(float(C[i, j]), 3)})
            else:
                cid = f"C{len(clusters) + 1}"
                clusters.append({"id": cid, "obs": [d], "box": list(d["box"]), "class_id": d["class_id"]})
                flog["new_clusters"].append({"det": d.get("id"), "cluster": cid})
        flog["missed_clusters"] = [clusters[j]["id"] for j in range(n_before) if j not in seen]
        matching.append(flog)

    merged, info = [], {}
    for c in clusters:
        obs = c["obs"]
        status, reason = classify_cluster(obs, nframes)
        info[c["id"]] = {
            "id": c["id"], "class": cname(c["class_id"]), "status": "rejected" if status is None else
            ("confirmed" if status == "confirmed" else "pending"), "reason": reason,
            "frames": sorted({o["frame"] + 1 for o in obs if o.get("frame") is not None}),
            "box": [int(round(v)) for v in c["box"]],
            "obs": [{"det": o.get("id"), "frame": o.get("frame", 0) + 1, "class": cname(o["class_id"]),
                     "conf": round(o["confidence"], 3), "imgsz": o.get("imgsz"),
                     "box": [int(round(v)) for v in o["box"]]} for o in obs]}
        if status is None:
            dropped.append({"stage": "merge", "reason": reason, "cluster": c["id"],
                            "dropped": brief(max(obs, key=lambda o: o["confidence"]))})
            continue
        t = _finalize_track(obs, status)
        t.update(id=c["id"], reason=reason)
        info[c["id"]]["votes"] = t["votes"]
        info[c["id"]]["class_share"] = round(t["class_share"], 3)
        merged.append(t)

    # dedup sau ghep: near-identical chi bi loai neu 2 cluster KHONG BAO GIO cung xuat hien trong 1 frame
    order = sorted(merged, key=lambda x: (x["status"] != "confirmed", -x["confidence"], x["id"]))
    kept = []
    for d in order:
        drop = None
        for k in kept:
            iou, cd, size = pair_metrics(d["box"], k["box"])
            if d["class_id"] == k["class_id"]:
                dup = iou >= DUP_IOU_SAME and cd <= DUP_CDIST and size >= DUP_SIZE
            else:
                dup = iou >= DUP_IOU_CROSS
            if not dup:
                continue
            if set(d["frame_idx"]) & set(k["frame_idx"]):      # cung 1 frame -> 2 vat khac nhau
                d["kept_overlap_pair"] = k["kept_overlap_pair"] = True
            else:
                drop = (k, iou)
                break
        if drop is None:
            kept.append(d)
        else:
            why = f"near_identical_never_cooccur iou={drop[1]:.2f} kept={drop[0]['id']}"
            dropped.append({"stage": "post_merge_dedup", "reason": why, "cluster": d["id"],
                            "dropped": {"class": cname(d["class_id"]), "conf": round(d["confidence"], 3),
                                        "box": [int(round(v)) for v in d["box"]]}})
            info[d["id"]].update(status="rejected", reason=why)
    kept.sort(key=lambda x: int(x["id"][1:]))
    for d in kept:                                 # vat nghi ngo chong lap khong bao gio tu 'da xac nhan'
        if d.get("kept_overlap_pair") and d["status"] == "confirmed":
            d["status"], d["reason"] = "recovered", "overlap_pair_needs_check; " + d["reason"]
            info[d["id"]].update(status="pending", reason=d["reason"])
    return kept, {"matching": matching, "info": info}


# ---------- v3: re-inference vung nghi ngo (GIU NGUYEN) ----------
def refine_suspects(frame, merged, cfg, dropped):
    """Box screw to bat thuong (nghi 2 vit bi gop) -> chay lai YOLO tren vung crop phong lon.
    Neu detector tu tach duoc >=2 screw conf>=CONFIDENCE thi thay the, nhung danh dau
    'recovered' (CHO XAC MINH) - khong tu tang so 'da xac nhan'. Toi da MAX_REFINE vung/phien."""
    if not cfg.get("refine"):
        return merged
    screw_areas = sorted((d["box"][2] - d["box"][0]) * (d["box"][3] - d["box"][1])
                         for d in merged if cname(d["class_id"]) == "screw")
    if len(screw_areas) < 3:
        return merged
    med = screw_areas[len(screw_areas) // 2]
    H, W = frame.shape[:2]
    out, n_ref = [], 0
    for d in merged:
        x1, y1, x2, y2 = d["box"]
        area = (x2 - x1) * (y2 - y1)
        suspect = cname(d["class_id"]) == "screw" and area > SUSPECT_AREA_K * med
        if not suspect or n_ref >= MAX_REFINE:
            out.append(d)
            continue
        n_ref += 1
        pad = 0.15 * max(x2 - x1, y2 - y1)
        cx1, cy1 = max(0, int(x1 - pad)), max(0, int(y1 - pad))
        cx2, cy2 = min(W, int(x2 + pad)), min(H, int(y2 + pad))
        if cx2 - cx1 < 8 or cy2 - cy1 < 8:
            out.append(d)
            continue
        r = model.predict(frame[cy1:cy2, cx1:cx2], conf=cfg["conf_infer"], imgsz=REFINE_IMGSZ,
                          max_det=50, verbose=False, **E2E_KW)[0]
        subs = get_detections(r)
        for s in subs:
            s["box"] = [s["box"][0] + cx1, s["box"][1] + cy1, s["box"][2] + cx1, s["box"][3] + cy1]
        subs = geo_dedup([s for s in subs if cname(s["class_id"]) == "screw"], frame)
        # chi giu cac box co tam nam trong box nghi ngo goc
        subs = [s for s in subs
                if x1 <= (s["box"][0] + s["box"][2]) / 2 <= x2 and y1 <= (s["box"][1] + s["box"][3]) / 2 <= y2]
        if len(subs) >= 2 and all(s["confidence"] >= CONFIDENCE for s in subs):
            for s in subs:
                s.update(status="recovered", observations=1, class_share=1.0,
                         votes={"screw": round(s["confidence"], 3)}, from_refine=True)
                out.append(s)
            dropped.append({"stage": "refine", "reason": f"split_into_{len(subs)}", "dropped": brief(d)})
            log(f"[REFINE] box {brief(d)['box']} tach thanh {len(subs)} screw (cho xac minh)")
        else:
            out.append(d)
    return out


def _refine_one(frame, d, cfg):
    """Chay lai YOLO tren crop quanh 1 box nghi ngo cua 1 frame -> list screw conf>=CONFIDENCE, tam nam trong box."""
    x1, y1, x2, y2 = d["box"]
    H, W = frame.shape[:2]
    pad = 0.15 * max(x2 - x1, y2 - y1)
    cx1, cy1 = max(0, int(x1 - pad)), max(0, int(y1 - pad))
    cx2, cy2 = min(W, int(x2 + pad)), min(H, int(y2 + pad))
    if cx2 - cx1 < 8 or cy2 - cy1 < 8:
        return []
    r = model.predict(frame[cy1:cy2, cx1:cx2], conf=cfg["conf_infer"], imgsz=REFINE_IMGSZ,
                      max_det=50, verbose=False, **E2E_KW)[0]
    subs = get_detections(r)
    for s in subs:
        s["box"] = [s["box"][0] + cx1, s["box"][1] + cy1, s["box"][2] + cx1, s["box"][3] + cy1]
    subs = geo_dedup([s for s in subs if cname(s["class_id"]) == "screw"], None)
    return sorted((s for s in subs if s["confidence"] >= CONFIDENCE
                   and x1 <= (s["box"][0] + s["box"][2]) / 2 <= x2
                   and y1 <= (s["box"][1] + s["box"][3]) / 2 <= y2), key=lambda s: s["box"][0])


def refine_suspects_v4(frames, merged, cfg, dropped, rtrace, info):
    """Nhu refine_suspects nhung chay tren MOI frame ma cluster nghi ngo xuat hien va chi tach khi da so
    frame dong y cung so luong >=2 (v3 chay tren 'best frame' - frame nay doi theo nhieu -> ket qua dao dong)."""
    if not cfg.get("refine"):
        return merged
    areas = sorted((d["box"][2] - d["box"][0]) * (d["box"][3] - d["box"][1])
                   for d in merged if cname(d["class_id"]) == "screw")
    if len(areas) < 3:
        return merged
    med = areas[len(areas) // 2]
    out, n_ref = [], 0
    for d in merged:
        area = (d["box"][2] - d["box"][0]) * (d["box"][3] - d["box"][1])
        if cname(d["class_id"]) != "screw" or area <= SUSPECT_AREA_K * med or n_ref >= MAX_REFINE:
            out.append(d)
            continue
        n_ref += 1
        runs = [(fi, _refine_one(frames[fi], d, cfg)) for fi in (d.get("frame_idx") or [0])]
        counts = [len(s) for _, s in runs]
        mode = max(set(counts), key=lambda c: (counts.count(c), -c))      # hoa -> chon so nho (than trong)
        agree = [r for r in runs if len(r[1]) == mode]
        ok = mode >= 2 and len(agree) * 2 > len(runs)
        rtrace.append({"cluster": d["id"], "box": [int(round(v)) for v in d["box"]],
                       "per_frame_split_counts": {fi + 1: len(s) for fi, s in runs},
                       "mode": mode, "agree": f"{len(agree)}/{len(runs)}", "accepted": ok})
        if not ok:
            out.append(d)
            continue
        fi_best, subs = max(agree, key=lambda r: sum(s["confidence"] for s in r[1]))
        kids = []
        for k, s in enumerate(subs, 1):
            sid = f"{d['id']}.{k}"
            s.update(status="recovered", observations=len(agree), class_share=1.0, from_refine=True,
                     votes={"screw": round(s["confidence"], 3)}, id=sid, frame_idx=[fi_best],
                     reason=f"refine_split_{mode}: consensus {len(agree)}/{len(runs)} frames")
            info[sid] = {"id": sid, "class": "screw", "status": "pending", "reason": s["reason"],
                         "frames": [fi_best + 1], "box": [int(round(v)) for v in s["box"]], "obs": []}
            kids.append(sid)
            out.append(s)
        info[d["id"]].update(status="replaced_by_refine", reason=f"split into {kids}")
        dropped.append({"stage": "refine", "reason": f"split_into_{mode}", "cluster": d["id"], "dropped": brief(d)})
        log(f"[REFINE v4] {d['id']} tach thanh {mode} screw (dong thuan {len(agree)}/{len(runs)} frame, cho xac minh)")
    return out


def review_reasons_for_session(frame_detections, merged_before_shape, merged_after_shape):
    reasons = []
    frame_counts = [len(dets) for dets in frame_detections]
    if frame_counts and max(frame_counts) - min(frame_counts) >= REVIEW_COUNT_SPREAD:
        reasons.append(f"frame_count_variation:{frame_counts}")
    for idx, det in enumerate(merged_after_shape, 1):
        name = cname(det["class_id"])
        if name in ("nut", "washer") and det["confidence"] < REVIEW_CONFIDENCE:
            reasons.append(f"low_confidence:object_{idx}:{name}:{det['confidence']:.3f}")
        if det.get("observations", 0) < 3:
            reasons.append(f"few_observations:object_{idx}:{det.get('observations', 0)}")
        if det.get("status") == "recovered":
            reasons.append(f"recovered_low_conf:object_{idx}:{name}:{det['confidence']:.3f}")
        if det.get("class_share", 1.0) < CLASS_SHARE_REVIEW:
            reasons.append(f"class_unstable:object_{idx}:{det['votes']}")
        if det.get("kept_overlap_pair"):
            reasons.append(f"kept_overlap_pair:object_{idx}:{name}")
        if det.get("from_refine"):
            reasons.append(f"split_suspect:object_{idx}:{name}")
        if REVIEW_ON_SHAPE_DISAGREEMENT and det.get("shape"):
            if det["shape"]["class"] != det.get("original_class"):
                reasons.append(f"shape_disagreement:object_{idx}")
    for idx, det in enumerate(merged_before_shape, 1):
        votes = Counter()
        for frame_dets in frame_detections:
            matches = [(calculate_iou(det["box"], x["box"]), x) for x in frame_dets]
            if matches:
                iou, match = max(matches, key=lambda p: p[0])
                if iou >= IOU_THRESHOLD:
                    votes[cname(match["class_id"])] += 1
        if votes.get("nut", 0) and votes.get("washer", 0):
            reasons.append(f"nut_washer_vote_conflict:object_{idx}:{dict(votes)}")
    return reasons


def run_counting_pipeline(frames, cfg=None, capture=None):
    """Pipeline thuan (khong dung camera/Flask) -> dung chung cho app va cac script offline."""
    cfg = cfg or make_cfg()
    stable = bool(cfg.get("stable"))
    dropped, per_frame, frame_dets, timing = [], [], [], {}
    t0 = time.perf_counter()
    for i, frame in enumerate(frames):
        dets, info = predict_frame(frame, cfg, dropped, frame_index=i)
        frame_dets.append(dets)
        per_frame.append(info)
        log(f"[FRAME {i + 1}/{len(frames)}] imgsz={info['imgsz']} raw={info['raw']} "
            f"after_dedup={info['kept']} below_{CONFIDENCE}={info['low_conf']}")
    t_infer = time.perf_counter() - t0
    timing["infer"] = round(t_infer, 3)

    t = time.perf_counter()
    quality = [frame_quality(f) for f in frames]
    if stable:
        best_frame, best_index, _ = select_best_frame_v4(frames, quality)
        sharp = [q["sharp"] for q in quality]
    else:
        best_frame, best_index, sharp = select_best_frame(frames)   # can truoc khi ghep (geo-dedup dung anh)
    timing["quality_select"] = round(time.perf_counter() - t, 3)

    mtrace, rtrace = None, []
    t = time.perf_counter()
    if stable:
        merged, mtrace = merge_detections_v4(frame_dets, cfg, dropped, len(frames))
        timing["merge"] = round(time.perf_counter() - t, 3)
        t = time.perf_counter()
        merged = refine_suspects_v4(frames, merged, cfg, dropped, rtrace, mtrace["info"])
        timing["refine"] = round(time.perf_counter() - t, 3)
    else:
        if cfg["legacy_merge"]:
            merged = merge_detections_legacy(frame_dets, dropped)
        elif cfg.get("hungarian"):
            merged = merge_detections_v3(frame_dets, cfg, dropped, best_frame)
        else:
            merged = merge_detections_v2(frame_dets, cfg, dropped)
        timing["merge"] = round(time.perf_counter() - t, 3)
        t = time.perf_counter()
        merged = refine_suspects(best_frame, merged, cfg, dropped)
        timing["refine"] = round(time.perf_counter() - t, 3)

    n_rec = sum(1 for d in merged if d.get("status") == "recovered")
    log(f"[MERGE] profile={cfg['name']} tracks_final={len(merged)} recovered={n_rec} "
        f"dropped={len(dropped)} infer={t_infer:.2f}s")
    if DEBUG:
        for item in dropped:
            log(f"[DROP] {item['stage']} {item['reason']} {item['dropped']}")

    t = time.perf_counter()
    baseline = [dict(d) for d in merged]
    final = apply_shape_check(best_frame, merged, cfg)
    reasons = review_reasons_for_session(frame_dets, baseline, final)
    timing["shape_review"] = round(time.perf_counter() - t, 3)

    # ---- v4: canh bao chat luong thu frame ----
    frame_diffs = None
    if stable:
        sigs = [frame_signature(f) for f in frames]
        if all(s is not None for s in sigs):
            frame_diffs = [round(signature_diff(sigs[i], sigs[i + 1]), 4) for i in range(len(sigs) - 1)]
            if any(d < 1e-6 for d in frame_diffs):
                reasons.append(f"duplicate_frames:{frame_diffs}")
            if any(d > MOTION_MAX for d in frame_diffs):
                reasons.append(f"motion_between_frames:{frame_diffs}")
        bs = [q["brightness"] for q in quality]
        bmed = sorted(bs)[len(bs) // 2]
        if bmed > 0 and max(abs(b - bmed) for b in bs) / bmed > EXPOSURE_DRIFT:
            reasons.append(f"exposure_drift:{[round(b, 1) for b in bs]}")
        if capture:
            reasons += [f"capture:{f}" for f in capture.get("flags", [])]

    counts = Counter(cname(d["class_id"]) for d in final)
    counts_confirmed = Counter(cname(d["class_id"]) for d in final if d.get("status") != "recovered")
    counts_pending = Counter(cname(d["class_id"]) for d in final if d.get("status") == "recovered")
    n_pending = sum(counts_pending.values())
    log(f"[COMPARE] YOLO+merge: {dict(Counter(cname(d['class_id']) for d in baseline))}")
    log(f"[COUNT] total={len(final)} confirmed={len(final) - n_pending} pending={n_pending} "
        f"by_class={dict(counts)} pending_by_class={dict(counts_pending)}")
    timing["total"] = round(time.perf_counter() - t0, 3)

    trace = None
    if stable:
        trace = {
            "capture": capture,
            "frame_quality": [{k: round(v, 4) for k, v in q.items()} for q in quality],
            "frame_diffs": frame_diffs, "best_frame": best_index + 1,
            "frames": [{"frame": i + 1, **{k: v for k, v in inf.items()}} for i, inf in enumerate(per_frame)],
            "matching": mtrace["matching"],
            "clusters": list(mtrace["info"].values()),
            "refine": rtrace,
            "dropped": dropped,
            "timing_s": timing,
        }
    return {"cfg": cfg, "merged": final, "baseline": baseline, "best_frame": best_frame,
            "best_index": best_index, "sharpness": sharp, "frame_detections": frame_dets,
            "per_frame": per_frame, "dropped": dropped, "reasons": reasons,
            "counts": counts, "counts_confirmed": counts_confirmed, "counts_pending": counts_pending,
            "total": len(final), "confirmed": len(final) - n_pending, "pending": n_pending,
            "t_infer": t_infer, "t_total": timing["total"], "timing": timing,
            "quality": quality, "trace": trace, "capture": capture}


# ---------- chu tieng Viet tren anh ket qua ----------
_BOLD_FONTS = [
    r"C:\Windows\Fonts\segoeuib.ttf", r"C:\Windows\Fonts\arialbd.ttf", r"C:\Windows\Fonts\tahomabd.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
]
_font_cache = {}


def _load_font(px):
    """Tim font dam ho tro tieng Viet. Co the chi dinh bang bien moi truong OVERLAY_FONT."""
    if px in _font_cache or ImageFont is None:
        return _font_cache.get(px)
    font = None
    env = os.getenv("OVERLAY_FONT")
    for path in ([env] if env else []) + _BOLD_FONTS:
        try:
            font = ImageFont.truetype(path, px)
            break
        except Exception:
            continue
    _font_cache[px] = font
    return font


def strip_accents(text):
    text = text.replace("đ", "d").replace("Đ", "D")
    return "".join(c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn")


def draw_summary_panel(img, lines, k):
    """Ve bang tong ket goc tren-trai (nen den mo + chu mau). lines = [(text, bgr), ...].
    Dung font TrueType nen hien thi duoc dau tieng Viet; neu khong tim thay font thi ve chu khong dau."""
    h, w = img.shape[:2]
    size, pad, step = int(round(28 * k)), int(round(14 * k)), int(round(40 * k))
    x0, y0 = int(15 * k), int(10 * k)
    font = _load_font(size)
    if font is None:                                    # du phong: chu khong dau bang cv2
        fb = cv2.FONT_HERSHEY_SIMPLEX
        fs, th = 0.75 * k, max(2, int(round(2 * k)))
        txt = [(strip_accents(t), c) for t, c in lines]
        width = max(cv2.getTextSize(t, fb, fs, th)[0][0] for t, _ in txt) + 2 * pad
        x1, y1 = min(w, x0 + width), min(h, y0 + step * len(txt) + pad)
        img[y0:y1, x0:x1] = (img[y0:y1, x0:x1] * 0.35).astype(np.uint8) if np is not None else img[y0:y1, x0:x1]
        for i, (t, c) in enumerate(txt):
            cv2.putText(img, t, (x0 + pad, y0 + int(round(30 * k)) + i * step), fb, fs, c, th, cv2.LINE_AA)
        return
    probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    width = int(max(probe.textlength(t, font=font) for t, _ in lines)) + 2 * pad
    x1, y1 = min(w, x0 + width), min(h, y0 + step * len(lines) + pad)
    roi = (img[y0:y1, x0:x1] * 0.35).astype(np.uint8)   # nen den, do dac 65%
    pil = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    for i, (text, bgr) in enumerate(lines):
        draw.text((pad, pad // 2 + i * step), text, font=font, fill=(bgr[2], bgr[1], bgr[0]))
    img[y0:y1, x0:x1] = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


PENDING_BGR = (0, 140, 255)   # cam - trung voi the "Cho xac minh" tren giao dien
CLASS_BGR = {"nut": (255, 255, 0), "screw": (255, 160, 80), "washer": (255, 255, 255)}


def pending_items(merged):
    """Danh sach vat cho xac minh, danh so ?1, ?2... theo thu tu trong merged
    (cung thu tu duoc ve tren anh ket qua)."""
    items = []
    for d in merged:
        if d.get("status") == "recovered":
            items.append({"index": len(items) + 1, "class": cname(d["class_id"]),
                          "conf": round(d["confidence"], 3), "obs": d.get("observations"),
                          "box": [int(round(v)) for v in d["box"]],
                          "id": d.get("id"), "reason": d.get("reason")})
    return items


def annotate_result(frame, merged, res):
    """Xanh la = da xac nhan; CAM + nhan '?n' = cho xac minh (conf thap / tach tu refine)."""
    annotated = frame.copy()
    k = max(1.0, frame.shape[1] / 1920.0)       # anh 4K -> chu va net ve to hon cho de doc
    font = cv2.FONT_HERSHEY_SIMPLEX
    pend_no = 0
    for det in merged:
        x1, y1, x2, y2 = [int(round(v)) for v in det["box"]]
        pend = det.get("status") == "recovered"
        name = str(model.names[det["class_id"]])
        if pend:
            pend_no += 1
            label, color = f"?{pend_no} {name} {det['confidence']:.2f}", PENDING_BGR
        else:
            label, color = f"{name}: {det['confidence']:.2f}", (0, 255, 0)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, int(round((3 if pend else 2) * k)))
        cv2.putText(annotated, label, (x1, max(int(20 * k), y1 - int(8 * k))),
                    font, 0.55 * k, color, max(2, int(round(2 * k))))

    counts, pend_c = res["counts"], res["counts_pending"]

    def cls_line(c, label):
        extra = f"  (chờ {pend_c[c]})" if pend_c.get(c) else ""
        return (f"{label}: {counts.get(c, 0)}{extra}", CLASS_BGR[c])

    lines = [(f"Tổng phát hiện: {res['total']}", (255, 255, 255)),
             (f"Đã xác nhận: {res['confirmed']}", (0, 255, 0)),
             (f"Chờ xác minh: {res['pending']}", PENDING_BGR),
             cls_line("nut", "Đai ốc"), cls_line("screw", "Ốc vít"), cls_line("washer", "Long đền")]
    draw_summary_panel(annotated, lines, k)
    return annotated, lines


def save_debug_images(ddir, frames, res):
    """(DEBUG, v4) dets_F<n>.jpg: moi detection sau dedup kem id; cluster_map.jpg: cluster + trang thai
    (xanh = da xac nhan, cam = cho xac minh, do = bi loai) tren frame tot nhat."""
    tr = res.get("trace")
    if not tr:
        return
    for i, fr in enumerate(frames):
        img = fr.copy()
        k = max(1.0, img.shape[1] / 1920.0)
        for d in res["frame_detections"][i]:
            x1, y1, x2, y2 = [int(round(v)) for v in d["box"]]
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 200, 0), max(1, int(round(2 * k))))
            cv2.putText(img, f"{d['id']} {d['confidence']:.2f}", (x1, max(12, y1 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45 * k, (255, 200, 0), max(1, int(round(k))))
        cv2.imwrite(str(ddir / f"dets_F{i + 1}.jpg"), img)
    img = res["best_frame"].copy()
    k = max(1.0, img.shape[1] / 1920.0)
    colors = {"confirmed": (0, 255, 0), "pending": PENDING_BGR, "rejected": (0, 0, 255)}
    for c in tr["clusters"]:
        if c["status"] not in colors:
            continue
        x1, y1, x2, y2 = c["box"]
        cv2.rectangle(img, (x1, y1), (x2, y2), colors[c["status"]], max(1, int(round(2 * k))))
        cv2.putText(img, f"{c['id']} n={len(c['frames'])}", (x1, max(12, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5 * k, colors[c["status"]], max(1, int(round(k))))
    cv2.imwrite(str(ddir / "cluster_map.jpg"), img)


def build_report(res, filename, frame_shape):
    h, w = frame_shape[:2]
    x1, y1, x2, y2 = get_roi_px(w, h)
    cfg = res["cfg"]
    tr = res.get("trace")
    report = {
        "result_filename": filename, "time": datetime.now().isoformat(timespec="seconds"),
        "profile": cfg["name"], "params": {k: cfg[k] for k in cfg},
        "e2e_kwargs": E2E_KW, "e2e_info": E2E_INFO,
        "confidence_final": CONFIDENCE, "dedup_iou": DEDUP_IOU, "merge_iou": IOU_THRESHOLD,
        "geo_dedup": {"iou_same": DUP_IOU_SAME, "cdist": DUP_CDIST, "size": DUP_SIZE,
                      "iou_cross": DUP_IOU_CROSS, "fill_x_max": FILL_X_MAX, "overlap_veto": OVERLAP_VETO},
        "match_v4": {"min_iou": MATCH_MIN_IOU, "min_iou_cross": MATCH_MIN_IOU_CROSS,
                     "min_size": MATCH_MIN_SIZE, "max_cost": MATCH_MAX_COST,
                     "confirm_frac": MIN_CONFIRM_FRAC},
        "roi_px": [x1, y1, x2, y2], "frame_size": [w, h],
        "counts": dict(res["counts"]), "counts_confirmed": dict(res["counts_confirmed"]),
        "counts_pending": dict(res["counts_pending"]),
        "total": res["total"], "confirmed": res["confirmed"], "pending": res["pending"],
        "rejected_count": sum(1 for c in tr["clusters"] if c["status"] == "rejected") if tr else None,
        "pending_items": pending_items(res["merged"]),
        # pending_weak = chi 1-2 frame ung ho / nghi chong lap; core_total = total - pending_weak (chi so on dinh de theo doi)
        "pending_weak": sum(1 for d in res["merged"] if d.get("status") == "recovered"
                            and str(d.get("reason", "")).startswith(("weak_support", "overlap_pair", "refine_split"))),
        "core_total": res["total"] - sum(1 for d in res["merged"] if d.get("status") == "recovered"
                                         and str(d.get("reason", "")).startswith(("weak_support", "overlap_pair", "refine_split"))),
        "sharpness": [round(s, 2) for s in res["sharpness"]], "best_frame": res["best_index"] + 1,
        "timing_s": {"infer": round(res["t_infer"], 3), "total": round(res["t_total"], 3), **res.get("timing", {})},
        "review_reasons": res["reasons"],
        "objects": [{**brief(d), "status": d.get("status"), "obs": d.get("observations"),
                     "id": d.get("id"), "reason": d.get("reason"),
                     "frames": [f + 1 for f in d.get("frame_idx", [])],
                     "class_share": round(d.get("class_share", 1.0), 3), "votes": d.get("votes"),
                     "original_class": d.get("original_class"), "shape": d.get("shape"),
                     "shape_changed_class": d.get("shape_changed_class", False),
                     "kept_overlap_pair": d.get("kept_overlap_pair", False),
                     "from_refine": d.get("from_refine", False)}
                    for d in res["merged"]],
        "per_frame": [{k: v for k, v in i.items() if k != "dets"} for i in res["per_frame"]],
        "dropped_count": len(res["dropped"]),
    }
    if tr:                                   # v4: truy vet day du, LUON ghi (khong phu thuoc DEBUG)
        report["trace"] = tr
    elif DEBUG:
        report["debug"] = {"dropped": res["dropped"],
                           "per_frame_dets": [i["dets"] for i in res["per_frame"]]}
    return report


# ==================================================
# 4. LUU FILE
# ==================================================
def save_raw_png(path, frame):
    if not cv2.imwrite(str(path), frame):
        raise RuntimeError(f"Khong luu duoc anh RAW: {path}")


def next_image_number(folder, prefix):
    numbers = []
    for path in folder.glob(f"{prefix}_*.jpg"):
        suffix = path.stem[len(prefix) + 1:]
        if suffix.isdigit():
            numbers.append(int(suffix))
    return max(numbers, default=0) + 1


def save_review_pair(raw_frame, result_path):
    with file_lock:
        number = next_image_number(REVIEW_DIR, "review")
        raw_path = REVIEW_DIR / f"review_{number}_raw.png"
        annotated_path = REVIEW_DIR / f"review_{number}.jpg"
        save_raw_png(raw_path, raw_frame)
        try:
            shutil.copy2(result_path, annotated_path)
            sidecar = Path(result_path).with_suffix(".json")
            if sidecar.is_file():
                shutil.copy2(sidecar, REVIEW_DIR / f"review_{number}.json")
        except Exception:
            raw_path.unlink(missing_ok=True)
            raise
    return raw_path, annotated_path


def process_count_session(frames, current_session_id, capture=None):
    global session_state, session_message
    global latest_counts, latest_total, latest_detections, latest_recovered
    global latest_result_path, latest_result_filename
    global latest_raw_path, latest_review_frame, latest_review_metadata, latest_review_reasons
    global latest_report
    try:
        if not frames:
            raise RuntimeError("Khong thu thap duoc frame.")
        res = run_counting_pipeline(frames, make_cfg(), capture)
        merged, best_frame, counts, total = res["merged"], res["best_frame"], res["counts"], res["total"]
        annotated, lines = annotate_result(best_frame, merged, res)

        log(f"\n========== KET QUA DEM ({len(frames)} FRAME, profile={res['cfg']['name']}) ==========")
        for line, _ in lines:
            log(line)
        log("============================================\n")

        with file_lock:
            number = next_image_number(SAVE_DIR, "result")
            filename = f"result_{number}.jpg"
            output_path = SAVE_DIR / filename
            if not cv2.imwrite(str(output_path), annotated):
                raise RuntimeError("Khong luu duoc anh ket qua.")

        report = build_report(res, filename, best_frame.shape)
        try:
            ddir = None
            if DEBUG:
                ddir = DEBUG_DIR / f"session_{current_session_id}_{datetime.now():%Y%m%d_%H%M%S}"
                ddir.mkdir(parents=True, exist_ok=True)
                report["debug_dir"] = str(ddir)
                for i, fr in enumerate(frames, 1):
                    save_raw_png(ddir / f"frame_{i}.png", fr)    # dau vao cho tools/test_*.py
                save_debug_images(ddir, frames, res)
            output_path.with_suffix(".json").write_text(
                json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
            if ddir is not None:
                (ddir / "report.json").write_text(
                    json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
        except Exception as exc:  # log phu khong duoc lam hong phien dem
            log(f"[WARN] Khong ghi duoc JSON/debug: {exc}")

        reasons = res["reasons"]
        raw_path = None
        if reasons:
            raw_path, _ = save_review_pair(best_frame, output_path)
            log(f"[REVIEW] {len(reasons)} flags: {reasons}")
        else:
            log("[REVIEW] No warning. RAW kept in RAM for manual review.")

        n_rec = res["pending"]
        with lock:
            if current_session_id != session_id or not camera_enabled:
                return
            latest_raw_path = raw_path
            latest_review_frame = best_frame.copy()
            latest_review_metadata = {"result_filename": filename}
            latest_review_reasons = reasons
            latest_counts = dict(counts)
            latest_total = total
            latest_detections = res["total"]
            latest_recovered = n_rec
            latest_report = report
            latest_result_path = output_path.resolve()
            latest_result_filename = filename
            session_state = "done"
            session_message = (f"Hoan tat: {total} phat hien ({res['confirmed']} da xac nhan, "
                               f"{n_rec} cho xac minh). Da luu anh {filename}")
    except Exception as exc:
        log(f"[ERROR] process_count_session: {exc!r}")
        with lock:
            if current_session_id == session_id:
                session_state = "error"
                session_message = f"Loi nhan dien: {exc}"


# ==================================================
# 5. CAMERA RTSP
# ==================================================
def open_ip_camera():
    cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # luu y: backend FFmpeg co the bo qua tham so nay
    return cap


def _abort_session_locked(message):
    """Goi khi dang giu lock: huy phien dang countdown/sampling neu camera rot."""
    global session_state, session_message, sample_frames
    if session_state in ("countdown", "sampling"):
        session_state = "error"
        session_message = message
        sample_frames = []


def camera_worker():
    global camera_enabled, camera_ready, camera_status
    global latest_frame, latest_jpeg
    global session_state, session_message
    global sample_frames, sample_meta, sample_info, sampling_t0, prev_sig, last_sample_time

    cap = None
    ready_reported = False
    last_preview = 0.0
    try:
        while True:
            with lock:
                if not camera_enabled:
                    break

            if cap is None or not cap.isOpened():
                ready_reported = False   # reset de camera_ready bat lai duoc sau khi rot mang
                with lock:
                    camera_ready = False
                    camera_status = "Đang kết nối camera IP..."
                    session_message = "Đang kết nối Hikvision RTSP..."
                    _abort_session_locked("Mất kết nối camera trong lúc đếm - hãy đếm lại")
                if cap is not None:
                    cap.release()
                cap = open_ip_camera()
                if not cap.isOpened():
                    cap.release()
                    cap = None
                    with lock:
                        camera_status = "Mất kết nối camera IP - đang thử lại"
                    time.sleep(RTSP_RECONNECT_DELAY)
                    continue

            # grab() LUON chay moi vong (giai ma + bo goi, giu do tre thap); retrieve() (chuyen mau,
            # cap phat 1 frame day du) chi khi that su can cho preview hoac mau dem.
            if not cap.grab():
                cap.release()
                cap = None
                ready_reported = False
                with lock:
                    camera_ready = False
                    camera_status = "Mất tín hiệu RTSP - đang kết nối lại"
                    _abort_session_locked("Mất tín hiệu camera trong lúc đếm - hãy đếm lại")
                time.sleep(RTSP_RECONNECT_DELAY)
                continue

            if not ready_reported:
                ready_reported = True
                with lock:
                    camera_ready = True
                    camera_status = "Camera IP đang bật"
                    session_message = "Camera IP đã sẵn sàng"

            now = time.monotonic()
            frames_to_process = None
            process_session_id = None
            capture_info = None
            with lock:
                if session_state == "countdown" and now >= countdown_deadline:
                    session_state = "sampling"
                    session_message = f"Đang lấy {SAMPLE_COUNT} frame..."
                    sample_frames, sample_meta = [], []
                    sample_info = {"restarts": 0, "duplicates": 0, "flags": []}
                    sampling_t0, last_sample_time, prev_sig = now, 0.0, None
                sampling = session_state == "sampling"
                s_t0, s_last = sampling_t0, last_sample_time
            want_preview = now - last_preview >= 1.0 / PREVIEW_FPS
            # bo SETTLE_SECONDS dau (decoder/auto-exposure on dinh sau countdown), sau do cach nhau SAMPLE_INTERVAL
            want_sample = (sampling and now - s_t0 >= SETTLE_SECONDS
                           and (s_last == 0.0 or now - s_last >= SAMPLE_INTERVAL))

            frame = None
            if want_preview or want_sample:
                ok, frame = cap.retrieve()
                if not ok or frame is None:
                    frame, want_sample, want_preview = None, False, False

            if frame is not None:
                sig = frame_signature(frame) if want_sample else None
                with lock:
                    latest_frame = frame   # khong copy: moi retrieve la mot mang moi; noi can sua thi tu copy
                    if want_sample and session_state == "sampling":
                        diff = None if (prev_sig is None or sig is None) else signature_diff(sig, prev_sig)
                        motion_check = (now - sampling_t0) < SAMPLE_TIMEOUT
                        if diff is not None and diff < 1e-6:
                            sample_info["duplicates"] += 1            # frame lap (decoder lap lai) -> doi frame moi
                        elif diff is not None and diff > MOTION_MAX and motion_check:
                            sample_info["restarts"] += 1              # canh dang dong -> thu lai tu dau
                            sample_frames, sample_meta = [], []
                            prev_sig, last_sample_time = sig, now
                        else:
                            if diff is not None and diff > MOTION_MAX and "motion_timeout" not in sample_info["flags"]:
                                sample_info["flags"].append("motion_timeout")
                            sample_frames.append(frame)
                            sample_meta.append({"t": round(now - sampling_t0, 3),
                                                "diff_prev": None if diff is None else round(diff, 4)})
                            prev_sig, last_sample_time = sig, now
                        if len(sample_frames) >= SAMPLE_COUNT:
                            frames_to_process = list(sample_frames)
                            capture_info = {"frames": list(sample_meta), "settle_s": SETTLE_SECONDS,
                                            "interval_s": SAMPLE_INTERVAL, **sample_info,
                                            "flags": list(sample_info["flags"])}
                            process_session_id = session_id
                            session_state = "processing"
                            session_message = f"Đang chạy YOLO trên {SAMPLE_COUNT} frame và ghép kết quả..."

            # Preview: thu nho truoc khi encode (imencode 4K/1080p moi frame rat ton CPU)
            if want_preview and frame is not None:
                last_preview = now
                ph, pw = frame.shape[:2]
                s = min(1.0, PREVIEW_MAX_W / float(pw))
                preview = cv2.resize(frame, (int(pw * s), int(ph * s)), interpolation=cv2.INTER_AREA) if s < 1.0 else frame.copy()
                if USE_ROI:
                    rx1, ry1, rx2, ry2 = get_roi_px(pw, ph)
                    sx1, sy1, sx2, sy2 = [int(round(v * s)) for v in (rx1, ry1, rx2, ry2)]
                    cv2.rectangle(preview, (sx1, sy1), (sx2 - 1, sy2 - 1), (0, 255, 0), 2, cv2.LINE_AA)
                    label = f"ROI - YOLO AREA ({rx2 - rx1}x{ry2 - ry1}) [{PROFILE}]"
                    font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                    (tw, th), base = cv2.getTextSize(label, font, scale, thick)
                    ly = max(th + 10, sy1 - 8)
                    cv2.rectangle(preview, (sx1, ly - th - 8), (sx1 + tw + 10, ly + base - 4), (0, 160, 0), -1)
                    cv2.putText(preview, label, (sx1 + 5, ly - 3), font, scale, (255, 255, 255), thick, cv2.LINE_AA)
                ok, encoded = cv2.imencode(".jpg", preview, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if ok:
                    with lock:
                        latest_jpeg = encoded.tobytes()

            if frames_to_process is not None:
                Thread(target=process_count_session,
                       args=(frames_to_process, process_session_id, capture_info), daemon=True).start()
            time.sleep(0.001)
    finally:
        if cap is not None:
            cap.release()
        with lock:
            camera_enabled = False
            camera_ready = False
            camera_status = "Đã dừng"
            session_state = "idle"
            session_message = "Camera IP đã dừng"


# ==================================================
# 6. FLASK ROUTES
# ==================================================
@app.route("/pick")
def pick():
    """Giao dien chinh nam o /pick."""
    return render_template("index.html")


@app.route("/")
def index():
    # Giu "/" de ban cu/duong dan cu van vao duoc: chuyen sang /pick.
    return redirect("/pick")


@app.route("/start", methods=["POST"])
def start_camera():
    global camera_enabled, camera_thread
    if not CAMERA_PASSWORD:
        return jsonify(ok=False, message="Chưa đặt biến môi trường CAMERA_PASSWORD"), 400
    with lock:
        if camera_enabled:
            return jsonify(ok=True, message="Camera đang bật")
        camera_enabled = True
        camera_thread = Thread(target=camera_worker, daemon=True)
        camera_thread.start()
    return jsonify(ok=True, message="Đang khởi động camera...")


@app.route("/stop", methods=["POST"])
def stop_camera():
    global camera_enabled, session_id, session_state, session_message
    with lock:
        camera_enabled = False
        session_id += 1
        session_state = "idle"
        session_message = "Đã dừng camera"
    return jsonify(ok=True, message="Đã yêu cầu dừng camera")


@app.route("/start_count", methods=["POST"])
def start_count():
    global session_id, session_state, session_message
    global countdown_deadline, sample_frames
    global latest_counts, latest_total, latest_detections, latest_recovered
    global latest_result_path, latest_result_filename
    global latest_raw_path, latest_review_frame, latest_review_metadata, latest_review_reasons
    global latest_report
    with lock:
        if not camera_enabled or not camera_ready:
            return jsonify(ok=False, message="Hãy bật camera và đợi camera sẵn sàng."), 400
        if session_state in ("countdown", "sampling", "processing"):
            return jsonify(ok=False, message="Phiên đếm hiện tại chưa kết thúc."), 400
        session_id += 1
        session_state = "countdown"
        session_message = "Chuẩn bị đếm ngược 3 giây..."
        countdown_deadline = time.monotonic() + COUNTDOWN_SECONDS
        sample_frames = []
        latest_counts = {}
        latest_total = 0
        latest_detections = 0
        latest_recovered = 0
        latest_result_path = None
        latest_result_filename = None
        latest_raw_path = None
        latest_review_frame = None
        latest_review_metadata = None
        latest_review_reasons = []
        latest_report = None
    return jsonify(ok=True, message="Bắt đầu đếm ngược 3 giây")


def generate_frames():
    while True:
        with lock:
            frame = latest_jpeg
            active = camera_enabled
        if frame is not None and active:
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
            time.sleep(1.0 / PREVIEW_FPS)   # tranh vong lap quay 100% CPU khi frame khong doi
        else:
            time.sleep(0.05)


@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/counts")
def get_counts():
    with lock:
        countdown_remaining = 0
        if session_state == "countdown":
            countdown_remaining = max(0, math.ceil(countdown_deadline - time.monotonic()))
        rep = latest_report or {}
        return jsonify(
            counts=latest_counts, total=latest_total, detections=latest_detections,
            confirmed=rep.get("confirmed", 0), pending=latest_recovered,
            rejected=rep.get("rejected_count"),
            counts_confirmed=rep.get("counts_confirmed", {}),
            counts_pending=rep.get("counts_pending", {}),
            pending_items=rep.get("pending_items", []),
            recovered=latest_recovered, profile=PROFILE,
            camera_status=camera_status, camera_ready=camera_ready,
            session_state=session_state, session_message=session_message,
            countdown_remaining=countdown_remaining, result_filename=latest_result_filename,
            raw_filename=latest_raw_path.name if latest_raw_path else None,
            review_required=bool(latest_review_reasons), review_reasons=latest_review_reasons,
        )


@app.route("/last_report")
def last_report():
    """Bao cao JSON phien gan nhat (ROI, tham so, tung vat the, vote, ly do loai, trace v4)."""
    with lock:
        report = latest_report
    if report is None:
        return jsonify(ok=False, message="Chưa có báo cáo"), 404
    return jsonify(report)


@app.route("/latest_result")
def latest_result():
    with lock:
        path = latest_result_path
    if path is None or not Path(path).is_file():
        return jsonify(ok=False, message="Chưa có ảnh kết quả"), 404
    return send_file(path, mimetype="image/jpeg")


@app.route("/save", methods=["POST"])
def save_image():
    with lock:
        frame = latest_frame.copy() if latest_frame is not None else None
    if frame is None:
        return jsonify(ok=False, message="Chưa có hình ảnh để lưu"), 400
    filename = datetime.now().strftime("manual_%Y%m%d_%H%M%S_%f.jpg")
    path = SAVE_DIR / filename
    if not cv2.imwrite(str(path), frame):
        return jsonify(ok=False, message="Không lưu được ảnh"), 500
    return jsonify(ok=True, message="Đã lưu ảnh camera hiện tại", filename=filename)


@app.route("/save_review", methods=["POST"])
def save_review():
    global latest_raw_path, latest_review_reasons
    with lock:
        raw_frame = latest_review_frame.copy() if latest_review_frame is not None else None
        result_path = latest_result_path
        existing_raw = latest_raw_path
    if raw_frame is None or result_path is None or not result_path.is_file():
        return jsonify(ok=False, message="Chua co phien dem de review"), 404
    if existing_raw is not None and existing_raw.is_file():
        return jsonify(ok=True, message="Phien nay da duoc luu review", raw_filename=existing_raw.name)
    raw_path, annotated_path = save_review_pair(raw_frame, result_path)
    with lock:
        if latest_result_path == result_path:
            latest_raw_path = raw_path
            if "manual_review" not in latest_review_reasons:
                latest_review_reasons = latest_review_reasons + ["manual_review"]
    return jsonify(ok=True, message="Da luu 2 anh vao review",
                   raw_filename=raw_path.name, annotated_filename=annotated_path.name)


@app.route("/latest_raw")
def latest_raw():
    with lock:
        frame = latest_review_frame.copy() if latest_review_frame is not None else None
    if frame is None:
        return jsonify(ok=False, message="Chua co RAW"), 404
    ok, encoded = cv2.imencode(".png", frame)
    if not ok:
        return jsonify(ok=False, message="Khong ma hoa duoc RAW"), 500
    return Response(encoded.tobytes(), mimetype="image/png")


if __name__ == "__main__":
    cfg0 = make_cfg()
    log(f"[BOOT] profile={PROFILE} debug={DEBUG} imgsz={IMAGE_SIZE} scales={cfg0['scales']} "
        f"preprocess={USE_PREPROCESS} conf_infer={cfg0['conf_infer']} "
        f"shape_correct={cfg0['shape_correct']} geo={cfg0['geo']} refine={cfg0['refine']} "
        f"overlap_veto={OVERLAP_VETO}")
    log(f"[BOOT] e2e_kwargs={E2E_KW} e2e_info={E2E_INFO}")
    if cfg0["nms_free"] and not E2E_KW:
        log("[BOOT][WARN] Khong co tham so 'end2end' trong cfg Ultralytics - dung che do mac dinh cua model. "
            "Kiem tra e2e_info.head_end2end: True = model dang chay NMS-free; False/None = can xem lai phien ban.")
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)