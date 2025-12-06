# =====================================
# 🔥 Firebase Storage 업로드 유틸
# =====================================

import os               # 파일 경로와 디렉터리 작업을 위한 모듈
import time             # 현재 시간을 얻기 위한 모듈 (파일 이름에 사용)
import cv2              # OpenCV: 이미지 저장 및 처리용
import firebase_admin    # Firebase SDK (파이썬용 관리자 모듈)
from firebase_admin import credentials, storage  # 인증과 Storage 모듈 불러오기
from config_keys import FIREBASE_KEY_PATH, FIREBASE_BUCKET  # 🔑 Firebase 키와 버킷 이름 설정값 가져오기


# =========================
# ⚙️ Firebase 초기화
# =========================

# Firebase 앱이 이미 초기화되어 있는지 확인
# (firebase_admin은 중복 초기화를 허용하지 않음)
if not firebase_admin._apps:
    # Firebase 비공개 키(JSON 파일)를 불러와 인증 객체 생성
    cred = credentials.Certificate(FIREBASE_KEY_PATH)
    # Firebase 앱 초기화: 어떤 스토리지 버킷(FIREBASE_BUCKET)을 사용할지 지정
    firebase_admin.initialize_app(cred, {"storageBucket": FIREBASE_BUCKET})

# Firebase의 Storage(파일 저장소) 객체를 불러옴
bucket = storage.bucket()


# =========================
# 📤 Firebase 업로드 함수
# =========================
def upload_to_firebase(image, car_id, save_dir):
    """
    번호판 이미지를 Firebase Storage에 업로드하고, 업로드된 이미지의 공개 URL을 반환하는 함수
    - image: OpenCV 이미지 객체 (numpy.ndarray 형태)
    - car_id: 차량 고유 식별자 (예: 번호판 문자)
    - save_dir: 임시로 이미지를 저장할 로컬 경로 (예: SAVE_DIR)
    """

    # 업로드할 파일 이름 생성 (plates/ 폴더 안에 "차량ID_현재시간.jpg" 형태로 저장)
    filename = f"plates/{car_id}_{int(time.time())}.jpg"

    # save_dir 폴더 내부에 임시 저장 경로 설정
    temp_path = os.path.join(save_dir, os.path.basename(filename))

    # ======================
    # 💾 1️⃣ 이미지 임시 저장
    # ======================
    # Firebase에 업로드하려면 먼저 로컬 파일로 저장해야 함
    cv2.imwrite(temp_path, image)  # OpenCV로 이미지 파일 저장

    # ======================
    # ☁️ 2️⃣ Firebase 업로드
    # ======================
    # Firebase Storage 내의 "plates/..." 경로에 파일을 업로드하기 위한 Blob 객체 생성
    blob = bucket.blob(filename)

    # 방금 저장한 로컬 이미지 파일을 Firebase에 업로드
    blob.upload_from_filename(temp_path)

    # ======================
    # 🌍 3️⃣ 업로드된 파일을 공개 URL로 전환
    # ======================
    # 기본적으로 Firebase에 올린 파일은 비공개이므로,
    # 누구나 접근 가능한 공개 URL로 변경
    blob.make_public()

    # ======================
    # 🧹 4️⃣ 로컬 임시 파일 삭제
    # ======================
    # Firebase에 올렸으니, 로컬 임시 파일은 삭제해 공간 절약
    os.remove(temp_path)

    # ======================
    # 🔗 5️⃣ 최종적으로 공개된 이미지 URL 반환
    # ======================
    return blob.public_url
