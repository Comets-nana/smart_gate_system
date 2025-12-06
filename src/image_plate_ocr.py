# =====================================
# 🚗 차량 번호판 인식 전체 파이프라인 코드
# - YOLO로 차량과 번호판 탐지
# - 투시 보정 및 OCR 전처리
# - CLOVA OCR로 텍스트 인식
# - Firebase / Supabase 연동
# =====================================

import os, re, io, json, time, uuid, csv       # 파일, 문자열, 시간, JSON, CSV 등 기본 라이브러리
import numpy as np                             # 배열 계산용 라이브러리
import cv2                                     # OpenCV: 이미지 처리용 라이브러리
import requests                                # HTTP 요청을 보내기 위한 라이브러리 (CLOVA OCR 호출용)
from ultralytics import YOLO                   # YOLO 모델 (객체 탐지용)
from PIL import Image                          # PIL: 이미지를 파이썬 객체로 다루기 위해 사용
import pyperclip                               # 클립보드 복사용
from config_keys import CLOVA_OCR_URL, CLOVA_SECRET_KEY  # CLOVA OCR API 키 설정 가져오기

# 👉 추가: 장치 관련 라이브러리
import platform
import torch


# =====================================
# 🔍 장치(Device) 자동 감지 (Mac / Windows / Linux 공통 지원)
# =====================================

def get_device():
    """
    현재 환경에서 사용할 PyTorch 장치를 자동 감지한다.
    - Apple Silicon(M1/M2/M3) + MPS 사용 가능 → 'mps'
    - NVIDIA GPU 사용 가능 → 'cuda'
    - 그 외 → 'cpu'
    """
    # 🍎 Apple Silicon (M1/M2/M3)
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        if torch.backends.mps.is_available():
            print("🍎 Apple Silicon(M1/M2/M3) → MPS 가속 사용")
            return "mps"
        else:
            print("⚠️ Apple Silicon 감지됨, 하지만 MPS 사용 불가능 → CPU로 진행")
            return "cpu"

    # 🚀 NVIDIA GPU
    if torch.cuda.is_available():
        print(f"🚀 NVIDIA CUDA GPU 사용: {torch.cuda.get_device_name(0)}")
        return "cuda"

    # 🧱 CPU
    print("⚠️ GPU 미감지 → CPU 사용")
    return "cpu"

DEVICE = get_device()


# =====================================
# ⚙️ 경로, 모델, 저장 관련 설정
# =====================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # 현재 파일 기준
SAVE_DIR = os.path.join(BASE_DIR, "..", "plate_results")

os.makedirs(SAVE_DIR, exist_ok=True)  # 폴더가 없으면 새로 만들기

# 로그 CSV 파일 경로 설정
LOG_CSV = os.path.join(SAVE_DIR, "plate_log.csv")
# 로그 파일이 없으면 첫 줄(헤더) 생성
if not os.path.exists(LOG_CSV):
    with open(LOG_CSV, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["Timestamp", "Plate_Index", "Car_Box", "Plate_Box", "OCR_Text"])

# 테스트용 이미지 경로 설정
IMAGE_PATH = os.path.join(BASE_DIR, "..", "assets", "images", "test_car.jpg")
IMAGE_PATH = os.path.normpath(IMAGE_PATH)

# YOLO 모델 경로 설정
MODEL_COCO = os.path.normpath(os.path.join(BASE_DIR, "..", "models", "yolov8n.pt"))
MODEL_LP  = os.path.normpath(os.path.join(BASE_DIR, "..", "models", "LP-detection.pt"))

# YOLO 모델 불러오기
COCO_MODEL = YOLO(MODEL_COCO).to(DEVICE)
PLATE_MODEL = YOLO(MODEL_LP).to(DEVICE)


# =====================================
# 🧰 유틸 함수 모음
# =====================================

def sanitize_filename(name: str, fallback="unknown"):
    if not name:
        name = fallback
    name = re.sub(r'[\\/:*?"<>|]+', '', name)
    name = re.sub(r'[\s\[\]\(\)]+', '_', name)
    name = re.sub(r'_+', '_', name).strip('_')
    return name or fallback

def imwrite_unicode_safe(path, img):
    img = np.ascontiguousarray(img)
    ext = os.path.splitext(path)[1] or ".jpg"
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        return False
    try:
        buf.tofile(path)
        return os.path.exists(path)
    except Exception:
        return False

def copy_image_to_clipboard_bmp(gray_or_bgr):
    if len(gray_or_bgr.shape) == 2:
        img = gray_or_bgr
    else:
        img = cv2.cvtColor(gray_or_bgr, cv2.COLOR_BGR2GRAY)
    pil = Image.fromarray(img)
    output = io.BytesIO()
    pil.save(output, format="BMP")
    data = output.getvalue()[14:]
    pyperclip.copy("")
    pyperclip.copy(data)

def draw_box(img, xyxy, color=(0,255,0), label=None):
    x1, y1, x2, y2 = map(int, xyxy)
    cv2.rectangle(img, (x1,y1), (x2,y2), color, 2)
    if label:
        cv2.putText(img, label, (x1, max(0, y1-8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

def perspective_correct(crop):
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3,3), 0)
    edged = cv2.Canny(gray, 100, 200)
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    warped = crop.copy()
    applied = False
    angle_print = 0.0
    ratio_print = 0.0

    for c in contours:
        area = cv2.contourArea(c)
        if area < crop.shape[0]*crop.shape[1]*0.2:
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02*peri, True)
        if len(approx) == 4:
            pts = approx.reshape(4,2).astype(np.float32)
            rect = np.zeros((4,2), dtype="float32")

            s = pts.sum(axis=1)
            rect[0] = pts[np.argmin(s)]
            rect[2] = pts[np.argmax(s)]
            diff = np.diff(pts, axis=1)
            rect[1] = pts[np.argmin(diff)]
            rect[3] = pts[np.argmax(diff)]

            tl, tr, br, bl = rect
            widthA = np.linalg.norm(br - bl)
            widthB = np.linalg.norm(tr - tl)
            heightA = np.linalg.norm(tr - br)
            heightB = np.linalg.norm(tl - bl)
            maxW = int(max(widthA, widthB))
            maxH = int(max(heightA, heightB))

            top_angle = np.degrees(np.arctan2(tr[1]-tl[1], tr[0]-tl[0]))
            bottom_angle = np.degrees(np.arctan2(br[1]-bl[1], br[0]-bl[0]))
            mean_angle = (top_angle + bottom_angle)/2.0

            aspect_ratio_diff = abs(widthA - widthB) / max(widthA, widthB)
            angle_print = float(mean_angle)
            ratio_print = float(aspect_ratio_diff)

            if abs(mean_angle) < 10 and aspect_ratio_diff < 0.3:
                applied = False
            else:
                dst = np.array([[0,0],[maxW-1,0],[maxW-1,maxH-1],[0,maxH-1]], dtype="float32")
                M = cv2.getPerspectiveTransform(rect, dst)
                warped = cv2.warpPerspective(crop, M, (maxW, maxH))
                applied = True
            break

    if applied:
        print(f"🟨 투시변환 적용 (기울기 {angle_print:.1f}°, 비율차 {ratio_print:.2f})")
    else:
        print("🟩 정면 번호판 감지 → 투시변환 생략")
    return warped, applied

def preprocess_for_ocr(warped_bgr):
    img = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    gamma = 1.2
    img = ((img / 255.0) ** (1.0 / gamma)) * 255.0
    img = np.clip(img, 0, 255).astype(np.uint8)
    img = cv2.resize(img, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    img = clahe.apply(img)
    sharpen_kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
    img = cv2.filter2D(img, -1, sharpen_kernel)
    img = cv2.bilateralFilter(img, 7, 75, 75)
    edge = cv2.Canny(img, 80, 200)
    binary_inv = cv2.adaptiveThreshold(img, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 19, 7)
    k = np.ones((2, 2), np.uint8)
    proc = cv2.morphologyEx(binary_inv, cv2.MORPH_CLOSE, k, iterations=1)
    proc = cv2.morphologyEx(proc, cv2.MORPH_OPEN, k, iterations=1)
    combined = cv2.bitwise_or(proc, edge)
    return combined

def call_clova_ocr(image_np_uint8, filename="plate.jpg"):
    if image_np_uint8.ndim == 2:
        img_bgr = cv2.cvtColor(image_np_uint8, cv2.COLOR_GRAY2BGR)
    else:
        img_bgr = image_np_uint8
    ok, buf = cv2.imencode(".jpg", img_bgr)
    if not ok:
        return ""
    req_json = {
        "images": [{"format":"jpg","name":"plate"}],
        "requestId": str(uuid.uuid4()),
        "version": "V2",
        "timestamp": int(round(time.time()*1000))
    }
    payload = {'message': (None, json.dumps(req_json), 'application/json')}
    files = [('file', (filename, buf.tobytes(), 'application/octet-stream'))]
    headers = {'X-OCR-SECRET': CLOVA_SECRET_KEY}
    print("📤 CLOVA OCR 요청 중...")
    resp = requests.post(CLOVA_OCR_URL, headers=headers, files=files, data=payload, timeout=30)
    print("HTTP", resp.status_code)

    if resp.status_code != 200:
        print(resp.text)
        return ""
    try:
        result = resp.json()
    except:
        return ""

    texts = []
    for img in result.get("images", []):
        for field in img.get("fields", []):
            t = field.get("inferText", "")
            if t:
                texts.append(t)
    return "".join(texts)

# =====================================
# 🔧 추가 기능: 자동 리사이즈 이미지 창
# =====================================

def show_resizable_window(window_name, img):
    """창 크기가 바뀔 때마다 이미지도 자동 리사이즈해서 보여주는 Viewer"""
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    while True:
        try:
            x, y, win_w, win_h = cv2.getWindowImageRect(window_name)
        except:
            win_w, win_h = img.shape[1], img.shape[0]

        # 🔥 추가: 창 크기가 아직 0 또는 음수라면 이미지 크기로 대체
        if win_w <= 1 or win_h <= 1:
            win_w, win_h = img.shape[1], img.shape[0]

        # 🔥 추가: 최소 안전 크기 확보
        win_w = max(2, win_w)
        win_h = max(2, win_h)

        resized = cv2.resize(img, (win_w, win_h), interpolation=cv2.INTER_AREA)
        cv2.imshow(window_name, resized)

        key = cv2.waitKey(30)
        if key == 13:  # Enter
            break


# =====================================
# 🚀 실제 파이프라인 실행
# =====================================
frame = cv2.imread(IMAGE_PATH)
if frame is None:
    raise FileNotFoundError("이미지를 불러올 수 없습니다.")

# 1️⃣ 차량 탐지
car_res = COCO_MODEL(frame, verbose=False)
car_boxes = []
for r in car_res:
    for b in r.boxes:
        cls = int(b.cls[0])
        if cls in (2, 5, 7):
            car_boxes.append(b.xyxy[0].tolist())

# 2️⃣ 번호판 탐지
plate_res = PLATE_MODEL(frame, verbose=False)
plate_boxes_raw = []
for r in plate_res:
    for b in r.boxes:
        plate_boxes_raw.append(b.xyxy[0].tolist())

def is_inside(inner, outer, margin=0):
    ix1, iy1, ix2, iy2 = inner
    ox1, oy1, ox2, oy2 = outer
    return (
        ix1 >= ox1 - margin and
        iy1 >= oy1 - margin and
        ix2 <= ox2 + margin and
        iy2 <= oy2 + margin
    )

plate_boxes = []
for pbox in plate_boxes_raw:
    for cbox in car_boxes:
        if is_inside(pbox, cbox, margin=10):
            plate_boxes.append(pbox)
            break

print(f"🚗 차량 내 번호판 수: {len(plate_boxes)} / 전체 번호판: {len(plate_boxes_raw)}")

# 시각화를 위한 복사본
vis = frame.copy()
for box in car_boxes:
    draw_box(vis, box, (255, 165, 0), "CAR")
for i, box in enumerate(plate_boxes, start=1):
    draw_box(vis, box, (0,255,0), f"PLATE {i}")

# === 추가된 부분: 전체 박스 시각화 창 ===
show_resizable_window("① Car & Plate Detection", vis)


# 3️⃣ 각 번호판별 처리
plate_idx = 0
for pbox in plate_boxes:
    plate_idx += 1

    x1, y1, x2, y2 = map(int, pbox)
    crop = frame[y1:y2, x1:x2].copy()
    if crop.size == 0:
        print(f"⚠️ Plate {plate_idx}: empty crop (skipped)")
        continue

    warped, applied_warp = perspective_correct(crop)

    ocr_input = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    ocr_input = cv2.normalize(ocr_input, None, 0, 255, cv2.NORM_MINMAX)
    ocr_input = cv2.resize(ocr_input, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
    ocr_input = cv2.bilateralFilter(ocr_input, 7, 75, 75)

    # === 추가된 부분: 보정 이미지 / OCR 전처리 이미지 창 ===
    show_resizable_window("② Perspective Correction Result", warped)
    show_resizable_window("③ OCR Preprocess Result", ocr_input)

    print("\n[Enter] 키 입력됨 → OCR 요청 진행 중...")

    # CLOVA OCR 실행
    ocr_text = call_clova_ocr(ocr_input, filename=f"plate_{plate_idx}.jpg")

    print(f"📸 Plate {plate_idx}: {ocr_text if ocr_text else '(인식 실패)'}")

    # === OCR 결과 처리 ===
    if ocr_text:
        from firebase_utils import upload_to_firebase
        from db_utils import insert_gate_entry, update_gate_exit, is_plate_existing

        plate_url = upload_to_firebase(warped, ocr_text, SAVE_DIR)

        if is_plate_existing(ocr_text):
            update_gate_exit(ocr_text)
            print(f"🚗 [EXIT] {ocr_text} 출차 완료")
        else:
            insert_gate_entry(plate_url, ocr_text)
            print(f"🚗 [ENTRY] {ocr_text} 입차 기록 추가")


cv2.destroyAllWindows()
print("✅ 파이프라인 완료")
