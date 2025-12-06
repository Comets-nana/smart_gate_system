# =====================================
# 🚗 Supabase 데이터베이스 연동 유틸 (수정됨)
# - gate_log 테이블 구조에 맞춘 로직
# =====================================

from supabase import create_client
from datetime import datetime
from config_keys import SUPABASE_URL, SUPABASE_KEY

# Supabase 클라이언트 생성
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)


# =========================
# 📌 입차 기록 추가 함수
# =========================
def insert_gate_entry(plate_url: str, plate_number: str):
    """
    차량이 입차할 때 gate_log 테이블에 데이터 추가
    - 사진 속 컬럼명 'plate_imgurl' 사용
    """
    data = {
        "gate_type": "ENTRY",  # 입장 상태
        "plate_imgurl": plate_url,  # Firebase 이미지 URL
        "plate_number": plate_number,  # 차량 번호
        "entry_time": datetime.now().isoformat(),  # 현재 시간
        "exit_time": None,  # 출차 시간은 비워둠
        "duration_text": "00:00:00",  # 초기값
        "fee": 0  # 초기값
    }

    # Supabase Insert
    response = supabase.table("gate_log").insert(data).execute()

    if response.data:
        print(f"✅ [DB] ENTRY inserted: {plate_number}")
    else:
        print(f"⚠️ [DB] ENTRY insert failed: {response}")


# =========================
# 📌 출차 기록 업데이트 함수
# =========================
def update_gate_exit(plate_number: str):
    """
    출차 시 기존 입차 기록을 찾아 출차 시간, 요금, 주차 시간을 업데이트
    """
    # 1. 현재 출차하지 않은(exit_time is null) 해당 차량 조회
    res = supabase.table("gate_log").select("id, entry_time") \
        .eq("plate_number", plate_number).is_("exit_time", None).execute()

    if not res.data:
        print(f"⚠️ no active entry for {plate_number}")
        return

    # 2. 시간 및 요금 계산
    entry_data = res.data[0]
    entry_time_str = entry_data["entry_time"]
    entry_time = datetime.fromisoformat(entry_time_str)  # 입차 시간
    exit_time = datetime.now()  # 출차 시간 (현재)

    # 주차 시간 계산 (초 단위)
    delta = exit_time - entry_time
    total_sec = int(delta.total_seconds())

    # 텍스트 포맷 (HH:MM:SS)
    hours, mins, secs = total_sec // 3600, (total_sec % 3600) // 60, total_sec % 60
    duration_text = f"{hours:02}:{mins:02}:{secs:02}"

    # 요금 계산 (예: 1분당 1000원)
    fee = max(1, total_sec // 60) * 1000

    # 3. 업데이트 데이터 구성
    update_data = {
        "gate_type": "EXIT",  # 상태 변경
        "exit_time": exit_time.isoformat(),  # 출차 시간 기록
        "duration_text": duration_text,  # 주차 소요 시간
        "fee": fee  # 주차 요금
    }

    # 4. Supabase Update (해당 ID의 레코드만 업데이트)
    supabase.table("gate_log").update(update_data).eq("id", entry_data["id"]).execute()

    print(f"✅ [DB] EXIT updated: {plate_number} | ⏱ {duration_text} | 💰 {fee}원")


# =========================
# 📌 입차 여부 확인 함수
# =========================
def is_plate_existing(plate_number: str) -> bool:
    """
    현재 주차장에 해당 차량이 있는지 확인 (출차 안 한 차량)
    """
    res = supabase.table("gate_log").select("id") \
        .eq("plate_number", plate_number).is_("exit_time", None).execute()

    return len(res.data) > 0


# ==========================================
# 🟢 [추가됨] 차량 번호 수동 수정 (GUI 연동)
# ==========================================
def update_plate_number(record_id, new_plate):
    """
    GUI 상세 창에서 호출: 특정 ID의 차량 번호를 수정
    """
    try:
        # Supabase Update
        response = supabase.table("gate_log").update({"plate_number": new_plate}).eq("id", record_id).execute()

        # 성공 여부 확인 (response.data가 비어있지 않으면 성공)
        if response.data:
            print(f"✅ [DB] ID {record_id} updated to {new_plate}")
            return True
        else:
            raise Exception("업데이트된 데이터가 없습니다.")

    except Exception as e:
        print(f"❌ [DB] Update Error: {e}")
        raise e


# ==========================================
# 🔴 [추가됨] 기록 삭제 (GUI 연동)
# ==========================================
def delete_record_by_id(record_id):
    """
    GUI 상세 창에서 호출: 특정 ID의 기록을 삭제
    """
    try:
        # Supabase Delete
        response = supabase.table("gate_log").delete().eq("id", record_id).execute()

        # 성공 여부 확인
        if response.data:
            print(f"✅ [DB] ID {record_id} deleted successfully")
            return True
        else:
            # ID가 없어서 삭제가 안 된 경우 등
            raise Exception("삭제할 데이터를 찾지 못했습니다.")

    except Exception as e:
        print(f"❌ [DB] Delete Error: {e}")
        raise e


# =========================
# 🔍 GUI 조회용 함수 (수정됨: 모든 컬럼 반환)
# =========================
def search_records(start_date: str, end_date: str, type_filter: str = "All"):
    """
    GUI에서 호출하는 조회 함수
    - 모든 컬럼(ID, 구분, 번호, 입차, 출차, 시간, 요금, 이미지) 반환
    """
    records = []

    # Supabase 검색을 위한 날짜 포맷
    s_ts = f"{start_date}T00:00:00"
    e_ts = f"{end_date}T23:59:59"

    try:
        # Supabase 쿼리 빌더
        query = supabase.table("gate_log").select("*") \
            .gte("entry_time", s_ts) \
            .lte("entry_time", e_ts)

        # 필터 조건 적용
        if type_filter == "Entry":
            query = query.eq("gate_type", "ENTRY")
        elif type_filter == "Exit":
            query = query.eq("gate_type", "EXIT")

        # 실행 (최신순)
        response = query.order("entry_time", desc=True).execute()

        # 데이터 가공
        for row in response.data:
            # 1. 기본 데이터 가져오기
            r_id = str(row.get("id", ""))
            g_type = row.get("gate_type", "")
            plate = row.get("plate_number", "")

            # 2. 시간 포맷팅 (T 제거)
            entry_t = row.get("entry_time", "")
            entry_t = entry_t.replace("T", " ")[:19] if entry_t else ""

            exit_t = row.get("exit_time")
            if exit_t:
                exit_t = exit_t.replace("T", " ")[:19]
            else:
                exit_t = "-"  # 출차 안했으면 하이픈

            # 3. 기타 정보
            duration = row.get("duration_text", "00:00:00")
            fee = row.get("fee", 0)
            img_url = row.get("plate_imgurl", "")

            # 4. 리스트에 추가 (GUI 컬럼 순서대로)
            # 순서: ID, 구분, 차량번호, 입차시간, 출차시간, 주차시간, 요금, 이미지URL
            records.append((r_id, g_type, plate, entry_t, exit_t, duration, fee, img_url))

    except Exception as e:
        print(f"❌ [DB] Search Error: {e}")
        return []

    return records