
from flask import Flask, render_template, Response, jsonify, send_file
from ultralytics import YOLO
from collections import Counter, defaultdict
from pathlib import Path
from threading import Thread, Lock
import cv2
import time
import math
import shutil

app = Flask(__name__)

# ==================================================
# 1. CAU HINH
# ==================================================
MODEL_PATH = Path(__file__).resolve().parent / "best3.pt"

CAMERA_INDEX = 0
CONFIDENCE = 0.6
IMAGE_SIZE = 1280
USE_PREPROCESS = True
USE_SHAPE_CHECK = True 
SHAPE_DEBUG = True  # In thong tin contour cua TAT CA nut/washer
SHAPE_MIN_SIZE = 18
SHAPE_MARGIN = 0.10
DEDUP_IOU = 0.65

COUNTDOWN_SECONDS = 3
SAMPLE_COUNT = 5
SAMPLE_INTERVAL = 0.15

# IoU càng cao thì hai bounding box càng giống nhau.
# Có thể điều chỉnh nếu các linh kiện nằm sát nhau.
IOU_THRESHOLD = 0.30

SAVE_DIR = Path(__file__).resolve().parent / "static" / "saved_images"
SAVE_DIR.mkdir(parents=True, exist_ok=True)
REVIEW_DIR = SAVE_DIR / "review"
REVIEW_DIR.mkdir(parents=True, exist_ok=True)
REVIEW_CONFIDENCE = 0.70
REVIEW_COUNT_SPREAD = 2
REVIEW_ON_SHAPE_DISAGREEMENT = True
file_lock = Lock()  # Bao ve danh so khi nhieu request cung luu


if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Khong tim thay model: {MODEL_PATH}"
    )

model = YOLO(str(MODEL_PATH))

lock = Lock()

# Trạng thái camera
camera_enabled = False
camera_ready = False
camera_thread = None
camera_status = "Đã dừng"

# Trạng thái phiên đếm
session_state = "idle"
session_message = "Sẵn sàng đếm"
session_id = 0
countdown_deadline = 0.0

# Dữ liệu camera
latest_frame = None
latest_jpeg = None

# Kết quả phiên đếm gần nhất
latest_counts = {}
latest_total = 0
latest_detections = 0
latest_result_path = None
latest_result_filename = None
latest_raw_path = None
latest_review_frame = None  # RAW in memory for manual review
latest_review_metadata = None
latest_review_reasons = []

# Lấy 5 frame mới sau khi hết đếm ngược
sample_frames = []
last_sample_time = 0.0


# ==================================================
# 2. CAC HAM XU LY ANH
# ==================================================
def calculate_iou(box_a, box_b):
    """Tính IoU giữa hai bounding box dạng x1, y1, x2, y2."""

    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    x1 = max(ax1, bx1)
    y1 = max(ay1, by1)
    x2 = min(ax2, bx2)
    y2 = min(ay2, by2)

    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)

    union = area_a + area_b - intersection

    if union <= 0:
        return 0.0

    return intersection / union


def calculate_sharpness(frame):
    """Do net trong vung giua khay, tranh vien camera lam sai diem."""
    h, w = frame.shape[:2]
    roi = frame[int(h*.15):int(h*.85), int(w*.15):int(w*.85)]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    # Blur nhe de giam diem net gia do nhieu sensor
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def select_best_frame(frames):
    """Chon frame net nhat truoc khi inference; khong tong hop detection 5 frame."""
    scores = [calculate_sharpness(f) for f in frames]
    idx = max(range(len(scores)), key=lambda i: scores[i])
    print(f"[FRAME SELECT] scores={[round(v, 2) for v in scores]} "
          f"selected={idx + 1}/{len(frames)}", flush=True)
    return frames[idx], idx, scores


def get_detections(result):
    """Chuyển kết quả YOLO về danh sách detection."""

    detections = []

    if result.boxes is None:
        return detections

    for box in result.boxes:
        xyxy = box.xyxy[0].cpu().numpy().tolist()
        class_id = int(box.cls[0].item())
        confidence = float(box.conf[0].item())

        detections.append({
            "box": [float(v) for v in xyxy],
            "class_id": class_id,
            "confidence": confidence
        })

    return detections


def preprocess_frame(frame):
    """CLAHE on luminance and conservative sharpening; preserve image dimensions."""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=1.6, tileGridSize=(8, 8)).apply(l)
    improved = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
    blur = cv2.GaussianBlur(improved, (0, 0), 0.7)
    return cv2.addWeighted(improved, 1.25, blur, -0.25, 0)


def predict_frame(frame):
    """Chi YOLO tren TOAN ANH, khong chia 4 vung."""
    source = preprocess_frame(frame) if USE_PREPROCESS else frame
    result = model.predict(
        source=source,
        conf=CONFIDENCE,
        imgsz=IMAGE_SIZE,
        agnostic_nms=True,
        iou=0.5,
        max_det=300,
        verbose=False,
    )[0]
    return deduplicate(get_detections(result))


def deduplicate(detections, threshold=DEDUP_IOU):
    """Class-independent greedy NMS; only suppress highly overlapping boxes."""
    ordered = sorted(detections, key=lambda d: d['confidence'], reverse=True)
    kept = []
    for item in ordered:
        if all(calculate_iou(item['box'], old['box']) < threshold for old in kept):
            kept.append(item)
    return kept


def shape_evidence(frame, box):
    """Return nut/washer shape evidence, or None if segmentation is unreliable."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    pad = max(3, int(max(x2-x1, y2-y1) * 0.18))
    x1, y1, x2, y2 = max(0,x1-pad), max(0,y1-pad), min(w,x2+pad), min(h,y2+pad)
    roi = frame[y1:y2, x1:x2]
    if min(roi.shape[:2]) < SHAPE_MIN_SIZE:
        return None
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3,3), 0)
    # Try both polarities: reflective hardware can be brighter or darker.
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
            # Reject background contours touching ROI border.
            if bx <= 1 or by <= 1 or bx+bw >= roi.shape[1]-1 or by+bh >= roi.shape[0]-1:
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
            score = area / roi_area
            candidates.append((score, circularity, len(approx)))
    if not candidates:
        return None
    _, circularity, vertices = max(candidates)
    # Strict rules: only strong evidence can override YOLO.
    if 5 <= vertices <= 7 and 0.78 <= circularity <= 0.94:
        return 'nut', circularity, vertices
    if vertices >= 8 and circularity >= 0.95:
        return 'washer', circularity, vertices
    return None


def apply_shape_check(frame, detections):
    """Kiem tra hinh dang; ghi log ca khi contour khong dat dieu kien."""
    corrected = []
    class_lookup = {str(v).lower(): int(k) for k, v in model.names.items()}
    for index, item in enumerate(detections, start=1):
        d = dict(item)
        name = str(model.names[d['class_id']]).lower()
        d['original_class'] = name
        d['shape'] = None
        if USE_SHAPE_CHECK and name in ('nut', 'washer'):
            evidence = shape_evidence(frame, d['box'])
            if SHAPE_DEBUG:
                print(
                    f"[SHAPE DEBUG] #{index} YOLO={name} "
                    f"conf={d['confidence']:.3f} "
                    f"box={[round(v) for v in d['box']]} "
                    f"OpenCV={evidence}",
                    flush=True
                )
            if evidence is not None:
                shape_class, circularity, vertices = evidence
                d['shape'] = {
                    'class': shape_class,
                    'circularity': round(circularity, 3),
                    'vertices': vertices
                }
                if shape_class != name:
                    strong = (
                        (shape_class == 'nut' and circularity <= 0.90 and vertices == 6)
                        or (shape_class == 'washer' and circularity >= 0.97 and vertices >= 9)
                    )
                    if strong and shape_class in class_lookup:
                        d['class_id'] = class_lookup[shape_class]
                        if SHAPE_DEBUG:
                            print(f"[SHAPE CORRECTION] #{index}: {name} -> {shape_class}", flush=True)
                    elif SHAPE_DEBUG:
                        print(f"[SHAPE KEEP] #{index}: hinh hoc chua du chac, giu {name}", flush=True)
        corrected.append(d)
    return corrected


# ==================================================
# 4. NHAN DIEN MOT PHIEN
# ==================================================
def merge_detections(frame_detections):
    """Ghep cung mot vat qua 5 frame; moi frame chi dong gop 1 lan/track."""
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
                track["last_box"] = det["box"]
                track["last_frame"] = frame_index
                used.add(best_idx)
            else:
                tracks.append({"observations": [det], "last_box": det["box"],
                               "last_frame": frame_index})
                used.add(len(tracks) - 1)

    merged = []
    for track in tracks:
        obs = track["observations"]
        if len(obs) < 2 and max(d["confidence"] for d in obs) < 0.60:
            continue
        votes = defaultdict(float)
        for d in obs:
            votes[d["class_id"]] += d["confidence"]
        chosen = max(votes, key=votes.get)
        selected = [d for d in obs if d["class_id"] == chosen]
        merged.append({
            "box": [sum(d["box"][j] for d in selected) / len(selected)
                    for j in range(4)],
            "class_id": chosen,
            "confidence": sum(d["confidence"] for d in selected) / len(selected),
            "observations": len(obs),
        })
    return deduplicate(merged)


def review_reasons_for_session(frame_detections, merged_before_shape, merged_after_shape):
    """Heuristic flags, not proof of misclassification."""
    reasons = []
    frame_counts = [len(dets) for dets in frame_detections]
    if frame_counts and max(frame_counts) - min(frame_counts) >= REVIEW_COUNT_SPREAD:
        reasons.append(f"frame_count_variation:{frame_counts}")

    for idx, det in enumerate(merged_after_shape, 1):
        name = str(model.names[det["class_id"]]).lower()
        if name in ("nut", "washer") and det["confidence"] < REVIEW_CONFIDENCE:
            reasons.append(f"low_confidence:object_{idx}:{name}:{det['confidence']:.3f}")
        if det.get("observations", 0) < 3:
            reasons.append(f"few_observations:object_{idx}:{det.get('observations', 0)}")
        if REVIEW_ON_SHAPE_DISAGREEMENT and det.get("shape"):
            if det["shape"]["class"] != det.get("original_class"):
                reasons.append(f"shape_disagreement:object_{idx}")

    # Check per-frame class disagreement for each merged object (location match).
    for idx, det in enumerate(merged_before_shape, 1):
        votes = Counter()
        for frame_dets in frame_detections:
            matches = [(calculate_iou(det["box"], x["box"]), x)
                       for x in frame_dets]
            if matches:
                iou, match = max(matches, key=lambda pair: pair[0])
                if iou >= IOU_THRESHOLD:
                    votes[str(model.names[match["class_id"]]).lower()] += 1
        if votes.get("nut", 0) and votes.get("washer", 0):
            reasons.append(f"nut_washer_vote_conflict:object_{idx}:{dict(votes)}")

    return reasons


def save_raw_png(path, frame):
    if not cv2.imwrite(str(path), frame):
        raise RuntimeError(f"Khong luu duoc anh RAW: {path}")


def next_image_number(folder, prefix):
    """Tim so tiep theo, khong ghi de file da ton tai sau khi khoi dong lai."""
    numbers = []
    for path in folder.glob(f"{prefix}_*.jpg"):
        suffix = path.stem[len(prefix) + 1:]
        if suffix.isdigit():
            numbers.append(int(suffix))
    return max(numbers, default=0) + 1


def save_review_pair(raw_frame, result_path):
    """Luu 2 anh cung so: review_N_raw.png va review_N.jpg."""
    with file_lock:
        number = next_image_number(REVIEW_DIR, "review")
        raw_path = REVIEW_DIR / f"review_{number}_raw.png"
        annotated_path = REVIEW_DIR / f"review_{number}.jpg"
        save_raw_png(raw_path, raw_frame)
        try:
            shutil.copy2(result_path, annotated_path)
        except Exception:
            raw_path.unlink(missing_ok=True)
            raise
    return raw_path, annotated_path


def process_count_session(frames, current_session_id):
    global session_state, session_message
    global latest_counts, latest_total, latest_detections
    global latest_result_path, latest_result_filename
    global latest_raw_path, latest_review_frame, latest_review_metadata, latest_review_reasons

    try:
        if not frames:
            raise RuntimeError("Khong thu thap duoc frame.")

        # YOLO chay TAT CA 5 frame, khong chia 4 vung.
        started = time.perf_counter()
        frame_detections = []
        for i, frame in enumerate(frames):
            detections = predict_frame(frame)
            frame_detections.append(detections)
            print(f"[FRAME {i + 1}/{len(frames)}] detections={len(detections)}", flush=True)

        merged = merge_detections(frame_detections)
        print(f"[INFERENCE] mode=5-frame/full-image "
              f"merged={len(merged)} time={time.perf_counter()-started:.2f}s", flush=True)

        # Frame net nhat chi dung cho shape check va ve anh ket qua.
        best_frame, best_index, scores = select_best_frame(frames)
        baseline = [dict(d) for d in merged]
        merged = apply_shape_check(best_frame, merged)
        print("[COMPARE] YOLO (preprocessed):",
              dict(Counter(model.names[d["class_id"]] for d in baseline)), flush=True)
        print("[COMPARE] YOLO+OpenCV:",
              dict(Counter(model.names[d["class_id"]] for d in merged)), flush=True)

        reasons = review_reasons_for_session(frame_detections, baseline, merged)
        counts = Counter(str(model.names[d["class_id"]]).lower() for d in merged)
        total = len(merged)
        annotated = best_frame.copy()

        for detection in merged:
            x1, y1, x2, y2 = [int(round(v)) for v in detection["box"]]
            class_name = str(model.names[detection["class_id"]])
            label = f"{class_name}: {detection['confidence']:.2f}"
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated, label, (x1, max(20, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

        summary_lines = [
            (f"TOTAL: {total}", (0, 255, 0)),
            (f"NUT: {counts.get('nut', 0)}", (0, 255, 255)),
            (f"SCREW: {counts.get('screw', 0)}", (255, 200, 0)),
            (f"WASHER: {counts.get('washer', 0)}", (255, 255, 255)),
        ]
        overlay = annotated.copy()
        cv2.rectangle(overlay, (15, 10), (275, 180), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.65, annotated, 0.35, 0, annotated)
        for i, (line, color) in enumerate(summary_lines):
            cv2.putText(annotated, line, (28, 43 + i * 36),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2, cv2.LINE_AA)

        print("\n========== KET QUA DEM (5 FRAME) ==========", flush=True)
        for line, _ in summary_lines:
            print(line, flush=True)
        print("============================================\n", flush=True)

        with file_lock:
            number = next_image_number(SAVE_DIR, "result")
            filename = f"result_{number}.jpg"
            output_path = SAVE_DIR / filename
            if not cv2.imwrite(str(output_path), annotated):
                raise RuntimeError("Khong luu duoc anh ket qua.")

        raw_path = None
        if reasons:
            raw_path, _ = save_review_pair(best_frame, output_path)
            print(f"[REVIEW] {len(reasons)} flags: {reasons}", flush=True)
        else:
            print("[REVIEW] No warning. RAW kept in RAM for manual review.", flush=True)

        with lock:
            if current_session_id != session_id or not camera_enabled:
                return
            latest_raw_path = raw_path
            latest_review_frame = best_frame.copy()
            latest_review_metadata = {"result_filename": filename}
            latest_review_reasons = reasons
            latest_counts = dict(counts)
            latest_total = total
            latest_detections = total
            latest_result_path = output_path.resolve()
            latest_result_filename = filename
            session_state = "done"
            session_message = f"Hoan tat: {total} linh kien. Da luu anh {filename}"
    except Exception as exc:
        with lock:
            if current_session_id == session_id:
                session_state = "error"
                session_message = f"Loi nhan dien: {exc}"


# ==================================================
# 5. LUONG CAMERA USB
# ==================================================
def camera_worker():
    global camera_enabled, camera_ready, camera_status
    global latest_frame, latest_jpeg
    global session_state, session_message
    global sample_frames, last_sample_time

    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)

    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        with lock:
            camera_enabled = False
            camera_ready = False
            camera_status = "Không mở được camera USB"
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    with lock:
        camera_ready = True
        camera_status = "Đang bật"
        session_message = "Camera đã sẵn sàng"

    try:
        while True:
            with lock:
                if not camera_enabled:
                    break

            ok, frame = cap.read()

            if not ok:
                time.sleep(0.03)
                continue

            now = time.monotonic()
            frame_for_sampling = None
            frames_to_process = None
            process_session_id = None

            with lock:
                latest_frame = frame.copy()

                # Bắt đầu lấy mẫu khi đếm ngược kết thúc.
                if (
                    session_state == "countdown"
                    and now >= countdown_deadline
                ):
                    session_state = "sampling"
                    session_message = f"Đang lấy {SAMPLE_COUNT} frame..."

                    sample_frames = []
                    last_sample_time = 0.0

                # Lấy các frame cách nhau một khoảng thời gian.
                if session_state == "sampling":
                    if (
                        not sample_frames
                        or now - last_sample_time >= SAMPLE_INTERVAL
                    ):
                        frame_for_sampling = frame.copy()
                        sample_frames.append(frame_for_sampling)
                        last_sample_time = now

                    if len(sample_frames) >= SAMPLE_COUNT:
                        frames_to_process = [
                            item.copy() for item in sample_frames
                        ]
                        process_session_id = session_id

                        session_state = "processing"
                        session_message = (
                            "Đang chạy YOLO trên 5 frame và ghép kết quả..."
                        )

            # Mã hóa preview ở ngoài lock để tránh chặn API.
            ok, encoded = cv2.imencode(
                ".jpg",
                frame,
                [cv2.IMWRITE_JPEG_QUALITY, 85]
            )

            if ok:
                with lock:
                    latest_jpeg = encoded.tobytes()

            # YOLO chạy trên thread riêng nên preview vẫn cập nhật.
            if frames_to_process is not None:
                Thread(
                    target=process_count_session,
                    args=(frames_to_process, process_session_id),
                    daemon=True
                ).start()

            time.sleep(0.005)

    finally:
        cap.release()

        with lock:
            camera_enabled = False
            camera_ready = False
            camera_status = "Đã dừng"
            session_state = "idle"
            session_message = "Camera đã dừng"


# ==================================================
# 6. FLASK ROUTES
# ==================================================
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/start", methods=["POST"])
def start_camera():
    global camera_enabled, camera_thread

    with lock:
        if camera_enabled:
            return jsonify(
                ok=True,
                message="Camera đang bật"
            )

        camera_enabled = True
        camera_thread = Thread(
            target=camera_worker,
            daemon=True
        )
        camera_thread.start()

    return jsonify(
        ok=True,
        message="Đang khởi động camera..."
    )


@app.route("/stop", methods=["POST"])
def stop_camera():
    global camera_enabled, session_id
    global session_state, session_message

    with lock:
        camera_enabled = False
        session_id += 1
        session_state = "idle"
        session_message = "Đã dừng camera"

    return jsonify(
        ok=True,
        message="Đã yêu cầu dừng camera"
    )


@app.route("/start_count", methods=["POST"])
def start_count():
    global session_id, session_state, session_message
    global countdown_deadline, sample_frames
    global latest_counts, latest_total, latest_detections
    global latest_result_path, latest_result_filename
    global latest_raw_path, latest_review_frame, latest_review_metadata, latest_review_reasons

    with lock:
        if not camera_enabled or not camera_ready:
            return jsonify(
                ok=False,
                message="Hãy bật camera và đợi camera sẵn sàng."
            ), 400

        if session_state in (
            "countdown", "sampling", "processing"
        ):
            return jsonify(
                ok=False,
                message="Phiên đếm hiện tại chưa kết thúc."
            ), 400

        session_id += 1
        session_state = "countdown"
        session_message = "Chuẩn bị đếm ngược 3 giây..."

        countdown_deadline = (
            time.monotonic() + COUNTDOWN_SECONDS
        )

        sample_frames = []

        # Mỗi phiên có kết quả riêng, không cộng dồn phiên trước.
        latest_counts = {}
        latest_total = 0
        latest_detections = 0
        latest_result_path = None
        latest_result_filename = None
        latest_raw_path = None
        latest_review_frame = None
        latest_review_metadata = None
        latest_review_reasons = []

    return jsonify(
        ok=True,
        message="Bắt đầu đếm ngược 3 giây"
    )


def generate_frames():
    while True:
        with lock:
            frame = latest_jpeg
            active = camera_enabled

        if frame is not None and active:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + frame + b"\r\n"
            )
        else:
            time.sleep(0.05)


@app.route("/video_feed")
def video_feed():
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/counts")
def get_counts():
    with lock:
        countdown_remaining = 0

        if session_state == "countdown":
            countdown_remaining = max(
                0,
                math.ceil(
                    countdown_deadline - time.monotonic()
                )
            )

        return jsonify(
            counts=latest_counts,
            total=latest_total,
            detections=latest_detections,
            camera_status=camera_status,
            camera_ready=camera_ready,
            session_state=session_state,
            session_message=session_message,
            countdown_remaining=countdown_remaining,
            result_filename=latest_result_filename,
            raw_filename=latest_raw_path.name if latest_raw_path else None,
            review_required=bool(latest_review_reasons),
            review_reasons=latest_review_reasons,
        )


@app.route("/latest_result")
def latest_result():
    with lock:
        path = latest_result_path

    if path is None or not Path(path).is_file():
        return jsonify(
            ok=False,
            message="Chưa có ảnh kết quả"
        ), 404

    return send_file(path, mimetype="image/jpeg")


@app.route("/save", methods=["POST"])
def save_image():
    with lock:
        frame = (
            latest_frame.copy()
            if latest_frame is not None else None
        )

    if frame is None:
        return jsonify(
            ok=False,
            message="Chưa có hình ảnh để lưu"
        ), 400

    filename = datetime.now().strftime(
        "manual_%Y%m%d_%H%M%S_%f.jpg"
    )
    path = SAVE_DIR / filename

    if not cv2.imwrite(str(path), frame):
        return jsonify(
            ok=False,
            message="Không lưu được ảnh"
        ), 500

    return jsonify(
        ok=True,
        message="Đã lưu ảnh camera hiện tại",
        filename=filename
    )


@app.route("/save_review", methods=["POST"])
def save_review():
    """Danh dau thu cong phien gan nhat; khong tao ban sao neu da review."""
    global latest_raw_path, latest_review_reasons
    with lock:
        raw_frame = latest_review_frame.copy() if latest_review_frame is not None else None
        result_path = latest_result_path
        existing_raw = latest_raw_path
    if raw_frame is None or result_path is None or not result_path.is_file():
        return jsonify(ok=False, message="Chua co phien dem de review"), 404
    if existing_raw is not None and existing_raw.is_file():
        return jsonify(ok=True, message="Phien nay da duoc luu review",
                       raw_filename=existing_raw.name)
    raw_path, annotated_path = save_review_pair(raw_frame, result_path)
    with lock:
        if latest_result_path == result_path:
            latest_raw_path = raw_path
            if "manual_review" not in latest_review_reasons:
                latest_review_reasons = latest_review_reasons + ["manual_review"]
    return jsonify(ok=True, message="Da luu 2 anh vao review",
                   raw_filename=raw_path.name,
                   annotated_filename=annotated_path.name)


@app.route("/latest_raw")
def latest_raw():
    """View last session RAW, even when it is not saved to disk."""
    with lock:
        frame = latest_review_frame.copy() if latest_review_frame is not None else None
    if frame is None:
        return jsonify(ok=False, message="Chua co RAW"), 404
    ok, encoded = cv2.imencode(".png", frame)
    if not ok:
        return jsonify(ok=False, message="Khong ma hoa duoc RAW"), 500
    return Response(encoded.tobytes(), mimetype="image/png")


if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False,
        threaded=True
    )
