# 🚗 Smart Gate System

YOLO 기반 차량 번호판 인식 및 Firebase 업로드 시스템  
디지털영상처리 실습 과제를 위해 제작된 프로젝트입니다.

---

## :two_men_holding_hands: 팀원 소개

| [김나현(팀장)](https://github.com/Comets-nana)                                                      | 백가영                                                                      | 심가현                                                                 | [정이레](https://github.com/ile777)                                                                 |
| -------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------- |
| <img src="https://avatars.githubusercontent.com/u/133929111?v=4" alt="김나현" width="100" height="100"/> | <img src="https://cdn-icons-png.flaticon.com/512/12225/12225773.png" alt="백가영" width="100" height="100"/> | <img src="https://cdn-icons-png.flaticon.com/512/12225/12225773.png" alt="심가현" width="100" height="100"/> | <img src="https://avatars.githubusercontent.com/u/210948992?v=4" alt="정이레" width="100" height="100"/> |

---

## :calendar: 개발 기간  
### **2025.09.09 ~ 2025.12.08**

---

## 🧰 1. 주요 기능

| 모듈 | 기능 |
|------|---------------------------------------------|
| **image_plate_ocr.py** | 단일 차량 이미지에서 번호판 검출 및 OCR 수행 |
| **video_plate_ocr.py** | 영상 스트림 내 프레임별 차량 번호판 검출 및 텍스트 추출 |
| **firebase_utils.py** | 인식된 번호판 이미지를 Firebase Storage에 업로드 |
| **db_utils.py** | 인식 결과를 CSV 로그로 저장 (plate_log.csv) |
| **config_keys.py** | CLOVA OCR, Firebase Key, SUPABASE Key 설정 관리 |

---

## ▶️ 2. 실행 방법

### 단일 이미지 OCR 실행
```bash
python src/image_plate_ocr.py
```

### 비디오 OCR 실행
```bash
python src/video_plate_ocr.py
```

---

## 🔗 참고 자료

- YOLOv8: https://github.com/ultralytics/ultralytics  
- CLOVA OCR: https://www.ncloud.com/product/aiService/ocr  
- Firebase Console: https://console.firebase.google.com  
- Supabase Dashboard: https://supabase.com/dashboard  
- License Plate Model: https://huggingface.co/MKgoud/License-Plate-Recognizer/tree/main  

---

📌 **작성자:** 김나현  
📅 **Last Updated:** 2025-11-13
