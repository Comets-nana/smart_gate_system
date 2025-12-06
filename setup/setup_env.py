import os
import subprocess
import sys
import platform

# =============== 설정 ===============
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REQUIREMENTS_FILE = os.path.join(BASE_DIR, "requirements.txt")
PYTHON = sys.executable  # 현재 파이썬 실행 파일 경로
TARGET_PY_VERSION = "3.11"
# ====================================


def run(cmd):
    """명령 실행 함수"""
    print(f"\n[실행 중] {cmd}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"❌ 오류 발생: {cmd}")
        sys.exit(1)


def check_python_version():
    """Python 버전 확인 및 설치 필요 시 자동 설치"""
    major, minor = sys.version_info[:2]
    if f"{major}.{minor}" == TARGET_PY_VERSION:
        print(f"✅ Python {TARGET_PY_VERSION} 버전이 이미 설치되어 있습니다.")
        return

    print(f"⚠️ 현재 Python 버전은 {major}.{minor}, {TARGET_PY_VERSION} 버전이 필요합니다.")
    install_python_311()


def install_python_311():
    """OS별 Python 3.11 자동 설치"""
    system = platform.system().lower()

    print(f"🔧 Python {TARGET_PY_VERSION} 설치를 시작합니다 ({system})...")

    if "windows" in system:
        installer_url = "https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe"
        installer_path = os.path.join(os.getcwd(), "python-3.11-installer.exe")
        run(f"curl -L {installer_url} -o {installer_path}")
        run(f"{installer_path} /quiet InstallAllUsers=1 PrependPath=1 Include_test=0")
        print("✅ Python 3.11 설치 완료. 다시 실행해 주세요.")
        sys.exit(0)

    elif "linux" in system:
        run("sudo apt update -y")
        run("sudo apt install -y software-properties-common")
        run("sudo add-apt-repository -y ppa:deadsnakes/ppa")
        run("sudo apt install -y python3.11 python3.11-venv python3.11-dev")
        print("✅ Python 3.11 설치 완료.")

    elif "darwin" in system:  # macOS
        run("brew update")
        run("brew install python@3.11")
        print("✅ Python 3.11 설치 완료.")

    else:
        print("⚠️ 알 수 없는 OS입니다. 수동으로 Python 3.11을 설치해주세요.")
        sys.exit(1)


def detect_gpu():
    """GPU 감지 (NVIDIA / Apple Silicon 구분)"""
    import platform

    # Apple Silicon (M1/M2/M3)
    if platform.machine() == "arm64" and platform.system() == "Darwin":
        print("🍎 Apple Silicon GPU(M1/M2/M3) 감지됨 → MPS 지원 가능")
        return "apple_silicon"

    # ✅ Intel 기반 macOS (x86_64) 처리: CUDA 미지원, CPU 전용으로 진행
    if platform.machine() == "x86_64" and platform.system() == "Darwin":
        print("🍎 Intel 기반 macOS 감지됨 → CUDA 미지원, CPU 전용 PyTorch로 진행")
        return None

    # NVIDIA GPU (Torch CUDA)
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            print(f"✅ GPU 감지됨: {gpu_name}")
            return gpu_name
    except Exception:
        pass

    # NVIDIA SMI 감지
    try:
        out = subprocess.check_output("nvidia-smi --query-gpu=name --format=csv,noheader", shell=True)
        gpu_name = out.decode().strip().split("\n")[0]
        print(f"✅ GPU 감지됨: {gpu_name}")
        return gpu_name
    except Exception:
        print("⚠️ GPU 감지 실패 (CPU 전용 설치로 진행)")
        return None



def install_torch(gpu_name):
    """GPU 이름 기반 Torch + CUDA/MPS 자동 설치"""

    print("\n🔧 기존 torch / torchvision / torchaudio 제거 중...")
    run(f"{PYTHON} -m pip uninstall -y torch torchvision torchaudio")

    # ⭐ Apple Silicon (M1/M2/M3)
    if gpu_name == "apple_silicon":
        print("🍎 Apple Silicon(M1/M2/M3) → MPS 백엔드 PyTorch 설치")
        run(f"{PYTHON} -m pip install torch torchvision torchaudio")
        print("✅ Apple Silicon 용 PyTorch 설치 완료 (MPS 사용 가능)")
        return

    # ==========================
    # 기존 GPU 매핑 (NVIDIA)
    # ==========================
    cuda_version = "cpu"

    if gpu_name:
        name = gpu_name.lower()

        # RTX 50 시리즈 (향후 5090/5080/5070/5060/5050 등) → CUDA 12.4
        if any(x in name for x in ["rtx 50", "5090", "5080", "5070", "5060", "5050"]):
            cuda_version = "cu124"

        # 40시리즈 → CUDA 12.4
        elif any(x in name for x in ["rtx 40", "4090", "4080", "4070", "4060", "4050"]):
            cuda_version = "cu124"

        # 30시리즈 → CUDA 11.8
        elif any(x in name for x in ["rtx 30", "3090", "3080", "3070", "3060", "3050"]):
            cuda_version = "cu118"

        # 데이터센터 GPU
        elif any(x in name for x in ["a100", "h100"]):
            cuda_version = "cu121"
        elif any(x in name for x in ["v100", "t4"]):
            cuda_version = "cu118"

        # RTX 20 시리즈 → CUDA 11.8
        elif any(x in name for x in ["rtx 20", "2080", "2070", "2060", "2050"]):
            cuda_version = "cu118"

        # GTX / RTX 10, 16 시리즈 → CUDA 11.8
        elif any(x in name for x in [
            "gtx 10", "gtx 16",
            "1080", "1070", "1060", "1050",
            "1660", "1650"
        ]):
            cuda_version = "cu118"

        # 일반 A 시리즈 (A10/A30/A40 등, A100/H100 제외) → CUDA 11.8
        elif ((" a" in name) or name.startswith("a")) and not any(x in name for x in ["a100", "h100"]):
            cuda_version = "cu118"

    print(f"🔧 선택된 CUDA 버전: {cuda_version}")

    # CPU 전용
    if cuda_version == "cpu":
        print("⚠️ GPU 미감지 → CPU 전용 PyTorch 설치")
        run(f"{PYTHON} -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu")
    else:
        print(f"🚀 GPU 버전 PyTorch 설치 중... ({cuda_version})")
        run(f"{PYTHON} -m pip install torch torchvision torchaudio --force-reinstall --index-url https://download.pytorch.org/whl/{cuda_version}")

    print("✅ Torch 설치 완료")



def main():
    print("🚀 환경 자동 세팅 시작")

    # 1️⃣ Python 버전 확인 및 설치
    check_python_version()

    # 2️⃣ pip 업그레이드
    run(f"{PYTHON} -m pip install --upgrade pip")

    # 3️⃣ GPU 감지 및 Torch 설치
    gpu_name = detect_gpu()
    install_torch(gpu_name)

    # 4️⃣ requirements.txt 설치
    if os.path.exists(REQUIREMENTS_FILE):
        print(f"\n📦 requirements.txt 설치 중 ({REQUIREMENTS_FILE})")
        run(f"{PYTHON} -m pip install -r {REQUIREMENTS_FILE}")
    else:
        print("⚠️ requirements.txt 파일이 없습니다. 기본 패키지 설치를 생략합니다.")

    print("\n✅ 모든 설치 완료!")


if __name__ == "__main__":
    main()
