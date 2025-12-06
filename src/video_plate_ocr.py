# =====================================
# 🚗 차량 번호판 인식 GUI 시스템 (영상/웹캠 지원)
# - YOLO로 차량 탐지 (ROI 적용)
# - 차량 영역 캡처 후 번호판 탐지 및 보정
# - CLOVA OCR로 텍스트 인식
# - Firebase / Supabase 연동
# =====================================

import os, re, io, json, time, uuid, csv  # 파일, 문자열, 시간, JSON, CSV 등 기본 라이브러리
import numpy as np  # 배열 계산용 라이브러리
import cv2  # OpenCV: 이미지 처리용 라이브러리
import requests  # HTTP 요청을 보내기 위한 라이브러리 (CLOVA OCR 호출용)
from ultralytics import YOLO  # YOLO 모델 (객체 탐지용)
from PIL import Image, ImageTk  # PIL: 이미지를 파이썬 객체로 다루기 위해 사용
import pyperclip  # 클립보드 복사용
from config_keys import CLOVA_OCR_URL, CLOVA_SECRET_KEY  # CLOVA OCR API 키 설정 가져오기

# 👉 추가: 장치 및 GUI 관련 라이브러리
import platform
import torch
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
from datetime import datetime, timedelta  # 날짜 계산용 추가


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
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        if torch.backends.mps.is_available():
            print("🍎 Apple Silicon(M1/M2/M3) → MPS 가속 사용")
            return "mps"
        else:
            return "cpu"
    if torch.cuda.is_available():
        print(f"🚀 NVIDIA CUDA GPU 사용: {torch.cuda.get_device_name(0)}")
        return "cuda"
    return "cpu"


DEVICE = get_device()

# =====================================
# ⚙️ 경로, 모델, 저장 관련 설정
# =====================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR = os.path.join(BASE_DIR, "..", "plate_results")
os.makedirs(SAVE_DIR, exist_ok=True)

# YOLO 모델 경로 설정
MODEL_COCO = os.path.normpath(os.path.join(BASE_DIR, "..", "models", "yolov8n.pt"))
MODEL_LP = os.path.normpath(os.path.join(BASE_DIR, "..", "models", "LP-detection.pt"))

# YOLO 모델 불러오기 (전역 로드)
print("⏳ 모델 로딩 중...")
COCO_MODEL = YOLO(MODEL_COCO).to(DEVICE)
PLATE_MODEL = YOLO(MODEL_LP).to(DEVICE)
print("✅ 모델 로드 완료")


# =====================================
# 🧰 유틸 함수 모음 (기존 로직 유지)
# =====================================

def sanitize_filename(name: str, fallback="unknown"):
    if not name: name = fallback
    name = re.sub(r'[\\/:*?"<>|]+', '', name)
    name = re.sub(r'[\s\[\]\(\)]+', '_', name)
    name = re.sub(r'_+', '_', name).strip('_')
    return name or fallback


def perspective_correct(crop):
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    edged = cv2.Canny(gray, 100, 200)
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    warped = crop.copy()
    applied = False

    # 디버깅용 변수
    angle_print = 0.0
    ratio_print = 0.0

    for c in contours:
        area = cv2.contourArea(c)
        if area < crop.shape[0] * crop.shape[1] * 0.2:
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            pts = approx.reshape(4, 2).astype(np.float32)
            rect = np.zeros((4, 2), dtype="float32")

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

            top_angle = np.degrees(np.arctan2(tr[1] - tl[1], tr[0] - tl[0]))
            bottom_angle = np.degrees(np.arctan2(br[1] - bl[1], br[0] - bl[0]))
            mean_angle = (top_angle + bottom_angle) / 2.0

            aspect_ratio_diff = abs(widthA - widthB) / max(widthA, widthB)
            angle_print = float(mean_angle)
            ratio_print = float(aspect_ratio_diff)

            if abs(mean_angle) < 10 and aspect_ratio_diff < 0.3:
                applied = False
            else:
                dst = np.array([[0, 0], [maxW - 1, 0], [maxW - 1, maxH - 1], [0, maxH - 1]], dtype="float32")
                M = cv2.getPerspectiveTransform(rect, dst)
                warped = cv2.warpPerspective(crop, M, (maxW, maxH))
                applied = True
            break

    return warped, applied


def call_clova_ocr(image_np_uint8, filename="plate.jpg"):
    if image_np_uint8.ndim == 2:
        img_bgr = cv2.cvtColor(image_np_uint8, cv2.COLOR_GRAY2BGR)
    else:
        img_bgr = image_np_uint8
    ok, buf = cv2.imencode(".jpg", img_bgr)
    if not ok: return ""

    req_json = {
        "images": [{"format": "jpg", "name": "plate"}],
        "requestId": str(uuid.uuid4()),
        "version": "V2",
        "timestamp": int(round(time.time() * 1000))
    }
    payload = {'message': (None, json.dumps(req_json), 'application/json')}
    files = [('file', (filename, buf.tobytes(), 'application/octet-stream'))]
    headers = {'X-OCR-SECRET': CLOVA_SECRET_KEY}

    try:
        resp = requests.post(CLOVA_OCR_URL, headers=headers, files=files, data=payload, timeout=10)
        if resp.status_code != 200:
            return ""
        result = resp.json()
        texts = []
        for img in result.get("images", []):
            for field in img.get("fields", []):
                t = field.get("inferText", "")
                if t: texts.append(t)
        return "".join(texts)
    except Exception as e:
        print(f"OCR Error: {e}")
        return ""


# =====================================
# 🚦 핵심 로직: 캡처된 차량 이미지 처리 (싱글샷 로직 재사용)
# =====================================
def process_captured_car_image(car_img):
    """
    영상에서 잘라낸 차량 이미지(car_img)를 받아
    번호판 탐지 -> 투시 변환 -> OCR -> DB 처리를 수행한다.
    """
    if car_img.size == 0: return None

    # 1. 번호판 탐지 (LP-detection)
    plate_res = PLATE_MODEL(car_img, verbose=False)
    plate_boxes = []
    for r in plate_res:
        for b in r.boxes:
            plate_boxes.append(b.xyxy[0].tolist())

    if not plate_boxes:
        return None

    # 가장 확신도가 높거나 큰 번호판 하나만 처리 (일단 첫 번째 것)
    pbox = plate_boxes[0]
    px1, py1, px2, py2 = map(int, pbox)

    # 번호판 Crop
    plate_crop = car_img[py1:py2, px1:px2].copy()
    if plate_crop.size == 0: return None

    # 2. 투시 보정
    warped, applied = perspective_correct(plate_crop)

    # 3. OCR 전처리
    ocr_input = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    ocr_input = cv2.normalize(ocr_input, None, 0, 255, cv2.NORM_MINMAX)
    ocr_input = cv2.resize(ocr_input, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
    ocr_input = cv2.bilateralFilter(ocr_input, 7, 75, 75)

    # 4. CLOVA OCR 호출
    ocr_text = call_clova_ocr(ocr_input, filename=f"captured_{uuid.uuid4()}.jpg")

    # 5. DB / Firebase 연동
    if ocr_text:
        print(f"✅ 인식 성공: {ocr_text}")
        try:
            # 사용자의 환경에 해당 파일이 있다고 가정
            from firebase_utils import upload_to_firebase
            from db_utils import insert_gate_entry, update_gate_exit, is_plate_existing

            # DB 로직
            plate_url = upload_to_firebase(warped, ocr_text, SAVE_DIR)

            if is_plate_existing(ocr_text):
                update_gate_exit(ocr_text)
                return f"[출차] {ocr_text}"
            else:
                insert_gate_entry(plate_url, ocr_text)
                return f"[입차] {ocr_text}"
        except ImportError:
            print("⚠️ DB 관련 모듈(firebase_utils, db_utils)을 찾을 수 없어 DB 저장은 건너뜁니다.")
            return f"[Test] {ocr_text}"
        except Exception as e:
            print(f"❌ DB 처리 오류: {e}")
            return f"[Error] {ocr_text}"

    return None


# =====================================
# 🎥 영상 처리 루프 (웹캠/파일 공통)
# =====================================
def run_video_analysis(source):
    """
    source: 0 (웹캠) 또는 '파일경로.mp4'
    """
    cap = cv2.VideoCapture(source)

    # 👉 [ZOOM ADDITION] 웹캠 여부 확인 및 해상도 설정
    is_webcam = isinstance(source, int)
    if is_webcam:
        # 고화질로 켜야 확대해도 덜 깨집니다 (1920x1080)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    if not cap.isOpened():
        print("❌ 영상을 열 수 없습니다.")
        return

    # 쿨타임 관리 (중복 인식 방지)
    last_process_time = 0
    COOLDOWN = 5.0  # 5초 대기

    window_name = "Parking System - Monitor"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    # 👉 [ZOOM ADDITION] 웹캠일 경우 트랙바 생성
    if is_webcam:
        cv2.createTrackbar('Zoom', window_name, 10, 100, lambda x: None)

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # 영상 끝

        # 👉 [ZOOM ADDITION] 웹캠일 경우 줌 로직 적용
        if is_webcam:
            h, w = frame.shape[:2]
            # 트랙바 값 가져오기
            zoom_val = cv2.getTrackbarPos('Zoom', window_name)
            if zoom_val < 10: zoom_val = 10  # 최소값 보정

            scale = zoom_val / 10.0  # 줌 배율 계산

            # 크롭 및 리사이즈 (디지털 줌)
            new_w, new_h = int(w / scale), int(h / scale)
            center_x, center_y = w // 2, h // 2
            x1 = max(0, center_x - new_w // 2)
            y1 = max(0, center_y - new_h // 2)
            x2 = min(w, center_x + new_w // 2)
            y2 = min(h, center_y + new_h // 2)

            cropped = frame[y1:y2, x1:x2]
            if cropped.size != 0:
                frame = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)

            # 줌된 프레임(frame)이 아래 로직으로 전달되어 분석됩니다.

        # 화면 중앙 ROI 설정 (전체 화면의 50% 크기)
        h, w = frame.shape[:2]
        roi_w, roi_h = int(w * 0.5), int(h * 0.5)
        roi_x = int((w - roi_w) / 2)
        roi_y = int((h - roi_h) / 2)

        # 시각화용 복사본
        vis = frame.copy()

        # 빨간색 사각형 그리기 (ROI)
        cv2.rectangle(vis, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (0, 0, 255), 3)
        cv2.putText(vis, "Target Area", (roi_x, roi_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # 현재 시간
        curr_time = time.time()

        # 쿨타임 중일 때 표시
        if curr_time - last_process_time < COOLDOWN:
            remain = COOLDOWN - (curr_time - last_process_time)
            cv2.putText(vis, f"Cooldown: {remain:.1f}s", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        # 1. 차량 탐지 (YOLOv8n) - 전체 프레임에서 수행
        car_results = COCO_MODEL(frame, verbose=False)

        target_car_img = None
        target_box = None

        for r in car_results:
            for b in r.boxes:
                cls = int(b.cls[0])
                # COCO 클래스: 2=car, 5=bus, 7=truck
                if cls in (2, 5, 7):
                    xyxy = b.xyxy[0].tolist()
                    x1, y1, x2, y2 = map(int, xyxy)

                    # 중심점 계산
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                    # 중심점이 ROI 안에 있는지 확인
                    if roi_x < cx < roi_x + roi_w and roi_y < cy < roi_y + roi_h:
                        # 인식된 차량 박스 그리기 (초록색)
                        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)

                        # 처리 가능한 상태라면 캡처 대상 선정
                        if curr_time - last_process_time >= COOLDOWN:
                            target_car_img = frame[y1:y2, x1:x2].copy()  # 캡처!
                            target_box = (x1, y1, x2, y2)
                            break  # 한 번에 한 대만 처리

        # 캡처된 차량이 있으면 분석 시작
        if target_car_img is not None:
            # UI에 'Processing...' 표시
            cv2.putText(vis, "Analyzing...", (target_box[0], target_box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0, 255, 0), 2)
            cv2.imshow(window_name, vis)
            cv2.waitKey(1)  # 화면 갱신

            print("\n📸 ROI 내 차량 감지! 분석 시작...")
            result_msg = process_captured_car_image(target_car_img)

            if result_msg:
                # 성공 시 쿨타임 리셋
                last_process_time = time.time()
                # 결과 화면에 잠시 띄우기 (옵션)
                print(f"👉 결과: {result_msg}")
            else:
                print("❌ 번호판 인식 실패")

        cv2.imshow(window_name, vis)

        # 'q' 키를 누르면 종료
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# =====================================
# 🖥️ GUI 구현 (Tkinter) - 체크박스 및 일괄 삭제 추가됨
# =====================================

class ParkingGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("주차 관제 시스템 Control Panel")
        self.root.geometry("500x400")

        # 👉 상세 팝업 창 상태 관리를 위한 변수
        self.det_win = None
        self.det_widgets = {}  # 팝업 창 내부의 위젯 참조 저장

        # 👉 이미지 줌 관련 상태 변수
        self.current_original_pil = None
        self.current_tk_image = None
        self.zoom_var = tk.DoubleVar(value=1.0)

        # 스타일 설정
        style = ttk.Style()
        style.configure("TButton", font=("맑은 고딕", 12), padding=10)
        style.configure("TLabel", font=("맑은 고딕", 14, "bold"))
        style.configure("Treeview", font=("맑은 고딕", 10), rowheight=25)

        # 메인 프레임
        main_frame = ttk.Frame(root, padding=20)
        main_frame.pack(expand=True, fill="both")

        # 제목
        title_lbl = ttk.Label(main_frame, text="🚗 차량 번호판 인식 시스템", anchor="center")
        title_lbl.pack(pady=20)

        # 버튼 1: 입차/출차 등록 (실시간 분석)
        btn_register = ttk.Button(main_frame, text="입차/출차 등록 (영상 분석)", command=self.open_register_window)
        btn_register.pack(fill="x", pady=10)

        # 버튼 2: 기록 보기
        btn_view = ttk.Button(main_frame, text="입차/출차 기록 보기", command=self.open_records_window)
        btn_view.pack(fill="x", pady=10)

        # 종료 버튼
        btn_exit = ttk.Button(main_frame, text="종료", command=root.quit)
        btn_exit.pack(fill="x", pady=20)

    def open_register_window(self):
        # ... (기존과 동일) ...
        top = tk.Toplevel(self.root)
        top.title("영상 입력 소스 선택")
        top.geometry("300x200")

        ttk.Label(top, text="분석할 소스를 선택하세요", font=("맑은 고딕", 11)).pack(pady=20)

        def start_webcam():
            top.destroy()
            run_video_analysis(0)

        def start_file():
            file_path = filedialog.askopenfilename(title="영상 파일 선택",
                                                   filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv")])
            if file_path:
                top.destroy()
                run_video_analysis(file_path)

        ttk.Button(top, text="📹 실시간 웹캠", command=start_webcam).pack(fill="x", padx=20, pady=5)
        ttk.Button(top, text="📂 영상 파일 열기", command=start_file).pack(fill="x", padx=20, pady=5)

    def open_records_window(self):
        # DB 기록 조회 창
        rec_win = tk.Toplevel(self.root)
        rec_win.title("입출차 기록 조회")
        rec_win.geometry("1000x600")  # 너비 약간 확장

        # === 상단 필터 영역 ===
        filter_frame = ttk.LabelFrame(rec_win, text="검색 및 관리", padding=15)
        filter_frame.pack(fill="x", padx=10, pady=5)

        # 1. 날짜 범위 선택
        today_str = datetime.now().strftime("%Y-%m-%d")

        ttk.Label(filter_frame, text="시작일:").grid(row=0, column=0, padx=5, sticky="e")
        entry_start = ttk.Entry(filter_frame, width=12)
        entry_start.insert(0, today_str)
        entry_start.grid(row=0, column=1, padx=5)

        ttk.Label(filter_frame, text="~ 종료일:").grid(row=0, column=2, padx=5, sticky="e")
        entry_end = ttk.Entry(filter_frame, width=12)
        entry_end.insert(0, today_str)
        entry_end.grid(row=0, column=3, padx=5)

        # 2. 입출차 유형 선택
        ttk.Label(filter_frame, text="구분:").grid(row=0, column=4, padx=5, sticky="e")
        type_var = tk.StringVar(value="전체")
        cmb_type = ttk.Combobox(filter_frame, textvariable=type_var, values=["전체", "입차", "출차"], state="readonly",
                                width=6)
        cmb_type.grid(row=0, column=5, padx=5)

        # 3. 조회 버튼
        btn_refresh = ttk.Button(filter_frame, text="🔍 조회", command=lambda: load_data())
        btn_refresh.grid(row=0, column=6, padx=10)

        # 4. 👉 [추가] 일괄 선택/삭제 버튼 영역
        ttk.Separator(filter_frame, orient="vertical").grid(row=0, column=7, sticky="ns", padx=10)

        self.is_all_checked = False  # 전체 선택 상태 토글용 변수
        btn_select_all = ttk.Button(filter_frame, text="✅ 전체선택", command=lambda: self.toggle_select_all(tree))
        btn_select_all.grid(row=0, column=8, padx=2)

        btn_delete_sel = ttk.Button(filter_frame, text="🗑️ 선택삭제",
                                    command=lambda: self.delete_selected_items(tree, load_data))
        btn_delete_sel.grid(row=0, column=9, padx=2)

        # === 결과 테이블 영역 ===
        # 👉 [수정] 맨 앞에 'check' 컬럼 추가
        columns = ("check", "id", "type", "plate", "entry", "exit", "duration", "fee", "image_url")
        tree = ttk.Treeview(rec_win, columns=columns, show="headings")

        # 헤더 설정
        tree.heading("check", text="선택")  # 체크박스 헤더
        tree.heading("id", text="ID")
        tree.heading("type", text="구분")
        tree.heading("plate", text="차량번호")
        tree.heading("entry", text="입차시간")
        tree.heading("exit", text="출차시간")
        tree.heading("duration", text="주차시간")
        tree.heading("fee", text="요금(원)")
        tree.heading("image_url", text="이미지 URL")

        # 너비 및 정렬 설정
        tree.column("check", width=50, anchor="center")  # 체크박스 칸은 좁게
        tree.column("id", width=40, anchor="center")
        tree.column("type", width=60, anchor="center")
        tree.column("plate", width=100, anchor="center")
        tree.column("entry", width=150, anchor="center")
        tree.column("exit", width=150, anchor="center")
        tree.column("duration", width=80, anchor="center")
        tree.column("fee", width=80, anchor="e")
        tree.column("image_url", width=300, anchor="w")

        # 스크롤바
        scrollbar = ttk.Scrollbar(rec_win, orient="vertical", command=tree.yview)
        tree.configure(yscroll=scrollbar.set)

        tree.pack(side="left", fill="both", expand=True, padx=10, pady=10)
        scrollbar.pack(side="right", fill="y", pady=10)

        # 👉 [추가] 싱글 클릭 이벤트 (체크박스 토글용)
        tree.bind("<ButtonRelease-1>", lambda event: self.on_tree_click(event, tree))

        # 더블 클릭 이벤트 (상세 보기용)
        tree.bind("<Double-1>", lambda event: self.on_item_double_click(event, tree, load_data))

        # 데이터 로드 함수
        def load_data():
            # 기존 목록 초기화
            for row in tree.get_children():
                tree.delete(row)
            self.is_all_checked = False  # 초기화

            s_date = entry_start.get().strip()
            e_date = entry_end.get().strip()
            sel_type = type_var.get()
            type_map = {"전체": "All", "입차": "Entry", "출차": "Exit"}
            db_type = type_map.get(sel_type, "All")

            try:
                from db_utils import search_records
                print(f"🔎 조회 요청: {s_date} ~ {e_date} ({db_type})")
                records = search_records(s_date, e_date, db_type)

                if not records:
                    messagebox.showinfo("알림", "조건에 맞는 데이터가 없습니다.")
                    return

                for row_data in records:
                    # row_data는 (id, type, plate, ...) 순서
                    # 👉 맨 앞에 체크박스 기본값 "☐" 추가
                    final_values = ("☐",) + row_data
                    tree.insert("", "end", values=final_values)

            except Exception as e:
                # DB 연결 실패 시 테스트 데이터 (형식 맞춤)
                print(f"⚠️ DB 조회 실패({e}), 테스트 데이터를 표시합니다.")
                dummy_data = [
                    ("1", "Entry", "12가3456", f"{s_date} 09:00:00", "-", "-", "0", "test.jpg"),
                    ("2", "Exit", "99하9999", f"{s_date} 10:00:00", f"{s_date} 11:00:00", "1h", "2000", "test2.jpg"),
                ]
                for d in dummy_data:
                    tree.insert("", "end", values=("☐",) + d)

        load_data()

    # 👉 [추가] 트리뷰 클릭 시 체크박스 토글 처리
    def on_tree_click(self, event, tree):
        region = tree.identify_region(event.x, event.y)
        if region == "cell":
            column = tree.identify_column(event.x)
            # 첫 번째 컬럼('#1')이 'check' 컬럼임
            if column == "#1":
                item_id = tree.identify_row(event.y)
                if not item_id: return

                # 현재 값 가져오기
                current_values = tree.item(item_id, "values")
                current_check = current_values[0]

                # 토글 (☐ <-> ☑)
                new_check = "☑" if current_check == "☐" else "☐"

                # 값 업데이트 (튜플이라 새로 생성해야 함)
                new_values = (new_check,) + current_values[1:]
                tree.item(item_id, values=new_values)

    # 👉 [추가] 전체 선택 / 해제 토글
    def toggle_select_all(self, tree):
        self.is_all_checked = not self.is_all_checked
        target_mark = "☑" if self.is_all_checked else "☐"

        for item_id in tree.get_children():
            current_values = tree.item(item_id, "values")
            new_values = (target_mark,) + current_values[1:]
            tree.item(item_id, values=new_values)

    # 👉 [추가] 선택된 항목 일괄 삭제
    def delete_selected_items(self, tree, refresh_callback):
        # 1. 체크된 항목의 ID 수집
        selected_ids = []
        for item_id in tree.get_children():
            values = tree.item(item_id, "values")
            # values[0]은 체크박스, values[1]은 ID
            if values[0] == "☑":
                selected_ids.append(values[1])

        if not selected_ids:
            messagebox.showwarning("알림", "삭제할 항목을 선택해주세요.")
            return

        # 2. 확인 메시지
        count = len(selected_ids)
        if not messagebox.askyesno("삭제 확인", f"선택한 {count}개의 데이터를 정말 삭제하시겠습니까?\n복구할 수 없습니다."):
            return

        # 3. DB 삭제 반복 실행
        try:
            from db_utils import delete_record_by_id
            success_count = 0

            for rec_id in selected_ids:
                try:
                    delete_record_by_id(rec_id)
                    success_count += 1
                except Exception as e:
                    print(f"삭제 실패 ID {rec_id}: {e}")

            messagebox.showinfo("완료", f"{success_count}개의 데이터가 삭제되었습니다.")
            refresh_callback()  # 목록 새로고침

        except ImportError:
            messagebox.showerror("오류", "db_utils.py를 찾을 수 없습니다.")
        except Exception as e:
            messagebox.showerror("오류", f"삭제 처리 중 오류 발생: {e}")

    # 👉 리스트 더블 클릭 (상세창) - 체크박스 클릭과 구분
    def on_item_double_click(self, event, tree, refresh_callback):
        # 클릭된 컬럼 확인
        column = tree.identify_column(event.x)
        if column == "#1":  # 체크박스 컬럼 더블클릭은 무시
            return

        selected_item = tree.selection()
        if not selected_item:
            return

        # 값 가져오기 (체크박스 포함되어 있으므로 인덱스 주의)
        # values: (check, id, type, plate, entry, exit, duration, fee, url)
        all_values = tree.item(selected_item, "values")

        # 상세창에는 체크박스 제외하고 전달 (id부터 끝까지)
        real_data = all_values[1:]
        self.show_detail_popup(real_data, refresh_callback)

    # 👉 상세 정보 팝업창 (기존 로직 유지)
    def show_detail_popup(self, data, refresh_callback):
        # data Unpacking (체크박스 제외된 8개 데이터)
        r_id, r_type, r_plate, r_entry, r_exit, r_dur, r_fee, r_url = data

        if self.det_win is not None and self.det_win.winfo_exists():
            self.det_win.lift()
            is_new_window = False
        else:
            is_new_window = True
            self.det_win = tk.Toplevel(self.root)
            self.det_win.title("상세 정보 및 수정")
            self.det_win.geometry("700x500")

        if is_new_window:
            self.det_widgets = {}
            paned = ttk.PanedWindow(self.det_win, orient="horizontal")
            paned.pack(fill="both", expand=True, padx=10, pady=10)

            # 좌측 프레임 (이미지+줌)
            frame_left = ttk.LabelFrame(paned, text="차량 이미지")
            paned.add(frame_left, weight=2)

            canvas_frame = ttk.Frame(frame_left)
            canvas_frame.pack(fill="both", expand=True, padx=5, pady=5)
            h_scroll = ttk.Scrollbar(canvas_frame, orient="horizontal")
            v_scroll = ttk.Scrollbar(canvas_frame, orient="vertical")
            canvas = tk.Canvas(canvas_frame, bg='#e1e1e1',
                               xscrollcommand=h_scroll.set, yscrollcommand=v_scroll.set)
            h_scroll.config(command=canvas.xview)
            v_scroll.config(command=canvas.yview)
            h_scroll.pack(side="bottom", fill="x")
            v_scroll.pack(side="right", fill="y")
            canvas.pack(side="left", fill="both", expand=True)
            self.det_widgets['canvas'] = canvas

            zoom_control = ttk.Frame(frame_left)
            zoom_control.pack(fill="x", padx=10, pady=5)
            ttk.Label(zoom_control, text="-").pack(side="left")
            ttk.Scale(zoom_control, from_=0.1, to=3.0, variable=self.zoom_var,
                      command=self.update_image_zoom).pack(side="left", fill="x", expand=True)
            ttk.Label(zoom_control, text="+").pack(side="left")

            # 우측 프레임 (정보)
            frame_info = ttk.LabelFrame(paned, text="단속 정보")
            paned.add(frame_info, weight=1)

            grid_frame = ttk.Frame(frame_info, padding=10)
            grid_frame.pack(fill="both", expand=True)

            def create_field(row, txt, key, ro=True):
                ttk.Label(grid_frame, text=txt).grid(row=row, column=0, sticky="e", pady=5)
                ent = ttk.Entry(grid_frame, state="readonly" if ro else "normal")
                ent.grid(row=row, column=1, sticky="ew", padx=5)
                self.det_widgets[key] = ent

            create_field(0, "ID:", 'id')
            create_field(1, "구분:", 'type')
            create_field(2, "차량번호:", 'plate', ro=False)
            create_field(3, "입차시간:", 'entry')
            create_field(4, "출차시간:", 'exit')
            create_field(5, "주차시간:", 'duration')
            create_field(6, "요금:", 'fee')

            btn_frame = ttk.Frame(frame_info)
            btn_frame.pack(fill="x", pady=10)
            ttk.Button(btn_frame, text="수정", command=lambda: self.update_record(refresh_callback)).pack(side="left",
                                                                                                        expand=True,
                                                                                                        fill="x",
                                                                                                        padx=2)
            ttk.Button(btn_frame, text="삭제", command=lambda: self.delete_record(refresh_callback)).pack(side="left",
                                                                                                        expand=True,
                                                                                                        fill="x",
                                                                                                        padx=2)

        # 데이터 바인딩
        w = self.det_widgets
        for k, v in zip(['id', 'type', 'plate', 'entry', 'exit', 'duration', 'fee'],
                        [r_id, r_type, r_plate, r_entry, r_exit, r_dur, r_fee]):
            w[k].configure(state='normal')
            w[k].delete(0, 'end')
            w[k].insert(0, str(v))
            if k != 'plate': w[k].configure(state='readonly')

        self.load_image_to_canvas(r_url)

    # (이하 load_image_to_canvas, update_image_zoom, update_record, delete_record 함수들은 기존 코드와 동일하므로 생략하지 않고 필요시 기존 코드를 그대로 사용)
    # 편의를 위해 이전에 작성된 함수들을 그대로 클래스 내부에 포함시켜 두어야 합니다.

    def load_image_to_canvas(self, url):
        # ... (이전 코드와 동일) ...
        canvas = self.det_widgets.get('canvas')
        if not canvas: return
        canvas.delete("all")
        self.current_original_pil = None
        try:
            image_data = None
            if url.startswith("http"):
                resp = requests.get(url, timeout=5)
                if resp.status_code == 200: image_data = resp.content
            elif os.path.exists(url):
                with open(url, "rb") as f:
                    image_data = f.read()

            if image_data:
                self.current_original_pil = Image.open(io.BytesIO(image_data))
                self.zoom_var.set(1.0)
                self.update_image_zoom()
            else:
                canvas.create_text(150, 150, text="이미지 없음")
        except Exception as e:
            print(f"이미지 로드 에러: {e}")

    def update_image_zoom(self, event=None):
        # ... (이전 코드와 동일) ...
        if not self.current_original_pil: return
        canvas = self.det_widgets.get('canvas')
        scale = self.zoom_var.get()
        w, h = self.current_original_pil.size
        nw, nh = int(w * scale), int(h * scale)
        if nw <= 0 or nh <= 0: return
        resized = self.current_original_pil.resize((nw, nh), Image.LANCZOS)
        self.current_tk_image = ImageTk.PhotoImage(resized)
        canvas.delete("all")
        canvas.create_image(0, 0, anchor="nw", image=self.current_tk_image)
        canvas.config(scrollregion=canvas.bbox("all"))

    def update_record(self, refresh_callback):
        # ... (이전 코드와 동일) ...
        rec_id = self.det_widgets['id'].get()
        new_plate = self.det_widgets['plate'].get()
        if not new_plate: return
        if messagebox.askyesno("수정", f"'{new_plate}'로 수정하시겠습니까?"):
            try:
                from db_utils import update_plate_number
                update_plate_number(rec_id, new_plate)
                messagebox.showinfo("완료", "수정되었습니다.")
                refresh_callback()
            except Exception as e:
                messagebox.showerror("에러", str(e))

    def delete_record(self, refresh_callback):
        # ... (이전 코드와 동일) ...
        rec_id = self.det_widgets['id'].get()
        if messagebox.askyesno("삭제", "삭제하시겠습니까?"):
            try:
                from db_utils import delete_record_by_id
                delete_record_by_id(rec_id)
                messagebox.showinfo("완료", "삭제되었습니다.")
                self.det_win.destroy()
                self.det_win = None
                refresh_callback()
            except Exception as e:
                messagebox.showerror("에러", str(e))


# =====================================
# 🚀 메인 실행부
# =====================================
if __name__ == "__main__":
    try:
        root = tk.Tk()
        app = ParkingGUI(root)
        root.mainloop()
    except KeyboardInterrupt:
        print("프로그램 종료")