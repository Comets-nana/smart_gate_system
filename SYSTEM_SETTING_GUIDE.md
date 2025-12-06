# 🖥️ System Setting Guide  
Parking System – License Plate OCR Pipeline  
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

> IDE에서 Python Interpreter를 Python 3.11로 선택 →  
> `setup/setup_env.py` 실행하면 CUDA 버전 자동 감지 + 패키지 자동 설치

---

### 🧩 수동 세팅

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r setup/requirements.txt
```

---

## 🧰 2. 폴더 구조

```
parking_system/
├── assets/
├── models/
├── plate_results/
├── setup/
└── src/
```

---

## 🧱 3. Supabase DB 접속 방법

### CMD 접속

```bash
psql "host=aws-1-ap-northeast-2.pooler.supabase.com       port=5432       dbname=postgres       user={ 사용자명 }       password={ 비밀번호 }       sslmode=require"
```

#### 확인 명령
```sql
\dt
SELECT * FROM gate_log LIMIT 10;
```

---

### HeidiSQL 설정

| 항목 | 값 |
|------|------|
| 호스트 | aws-1-ap-northeast-2.pooler.supabase.com |
| 포트 | 5432 |
| 사용자명 | { 사용자명 } |
| 비밀번호 | { 비밀번호 } |
| DB | postgres |
| SSL | 사용(검증 안 함) |

---

### PyCharm Database 설정

| 항목 | 값 |
|------|------|
| Host | aws-1-ap-northeast-2.pooler.supabase.com |
| Port | 5432 |
| Database | postgres |
| SSL | Require |

#### 예시 쿼리
```sql
SELECT * FROM gate_log ORDER BY entry_time DESC LIMIT 10;
```

---

## 🗄️ 4. 테이블 예시: `gate_log`

| 컬럼명 | 타입 | 설명 |
|--------|------|------|
| id | int8 | 기본 키 |
| gate_type | varchar | 입차/출차 |
| plate_imgurl | text | Firebase Storage URL |
| plate_number | text | OCR 결과 |
| entry_time | timestamp | 입차 |
| exit_time | timestamp | 출차 |
| created_at | timestamp | 자동 생성 |
| fee | int4 | 요금 |
| duration_text | text | 체류 시간 |

---

📌 **작성자:** 김나현  
📅 **Last Updated:** 2025-11-13
