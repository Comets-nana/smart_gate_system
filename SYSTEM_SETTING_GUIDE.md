# 🖥️ System Setting Guide  
Smart Gate System – License Plate OCR Pipeline  
(환경 세팅 전용 문서)

---

## 🧠 실험 환경 정보

| 항목 | 내용 |
|------|------|
| **Python 버전** | 3.11 |
| **GPU** | NVIDIA GeForce RTX 3080 |
| **CUDA 버전** | 12.6 |
| **OS** | Windows 10 |

---

## ⚙️ 1. 환경 세팅 방법

### ✅ 자동 세팅 (권장)

> **PyCharm, VSCode 등 IDE에서 Python Interpreter를 Python 3.11로만 설정해두면  
> conda 여부와 상관없이 자동 세팅이 가능합니다.**

1. **편집기(IDE)에서 Python Interpreter를 Python 3.11로 선택**
   - PyCharm: *Settings → Project → Python Interpreter*
   - VSCode: 좌측 하단 Python 버전 클릭 → **Python 3.11 선택**

2. `setup/setup_env.py` 파일을 열고 **▶(Run)** 버튼을 눌러 실행  
   - 명령어 입력 필요 없음  
   - CUDA 버전에 맞는 PyTorch 및 필요한 모든 패키지가 자동 설치됨  

---

### 🧩 수동 세팅 (선택)

1. Python 3.11 설치  
2. 가상환경 생성 및 활성화
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```

3. 패키지 설치
   ```bash
   pip install -r setup/requirements.txt
   ```

---

## 🧰 2. 폴더 구조

```
parking_system/
│
├── assets/                      
│   ├── images/                  
│   │   ├── test_car.jpg
│   │   ├── test_car_1.jpg
│   │   └── test_car_2.jpg
│   └── videos/                  
│       ├── video1.mp4
│       └── video2.mp4
│
├── models/                      
│   ├── LP-detection.pt
│   └── yolov8n.pt
│
├── plate_results/               
│   └── plate_log.csv
│
├── setup/                       
│   ├── requirements.txt
│   ├── setup_env.py
│   └── smart-gate-ocr-firebase-adminsdk-xxxx.json
│
└── src/                         
    ├── config_keys.py           
    ├── db_utils.py              
    ├── firebase_utils.py        
    ├── image_plate_ocr.py       
    └── video_plate_ocr.py       
```

---

## 📈 참고 및 권장 환경

- **Python**: 3.11  
- **PyTorch**: 자동 설치 (CUDA 12.6 대응 버전)  
- **GPU**: NVIDIA RTX 3080  
- **모델**
  - 차량 검출: `yolov8n.pt`
  - 번호판 검출: `LP-detection.pt`

---

📌 **작성자:** 김나현  
📅 **Last Updated:** 2025-11-13
