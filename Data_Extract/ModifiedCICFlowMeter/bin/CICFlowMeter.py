import os
import sys
import subprocess
import shutil
from pathlib import Path
from typing import List


# --- Cấu hình Cố định ---
APP_NAME = "CICFlowMeter"
MAIN_CLASS = "cic.cs.unb.ca.ifm.App"

DEFAULT_JVM_OPTS_TEMPLATE = "-Djava.library.path={}/lib/native"

MEMORY_OPTS = ["-Xmx10g", "-Xms512m"] 

def get_app_home() -> Path:
    script_path = Path(os.path.realpath(sys.argv[0]))
    app_home = script_path.parent.parent.resolve()
    
    if not (app_home / "lib").is_dir():
        print(f"Lỗi: Không tìm thấy thư mục 'lib' trong đường dẫn đã xác định: {app_home}")
        sys.exit(1)
        
    return app_home

APP_HOME = get_app_home()
print(f"[{APP_NAME}] APP_HOME đã được xác định: {APP_HOME}")


def build_classpath(app_home: Path) -> str:
    """Quét thư mục lib và tạo chuỗi CLASSPATH."""
    lib_dir = app_home / "lib"
    jar_files = [str(p.resolve()) for p in lib_dir.glob("*.jar")]
    
    classpath_string = os.pathsep.join(jar_files)
    
    if not classpath_string:
        print("Lỗi: Không tìm thấy file JAR nào trong thư mục lib. Kiểm tra cài đặt.")
        sys.exit(1)
        
    return classpath_string

CLASSPATH = build_classpath(APP_HOME)


def find_java_executable() -> str:
    """Tìm lệnh Java để chạy JVM."""
    if os.name.lower() == "linux":
        return "java"
    
    java_home = os.environ.get('JAVA_HOME')
    if java_home:
        javacmd = Path(java_home) / "bin" / "java"
        if javacmd.is_file() and os.access(javacmd, os.X_OK):
            return str(javacmd)
        
        javacmd_aix = Path(java_home) / "jre" / "sh" / "java"
        if javacmd_aix.is_file() and os.access(javacmd_aix, os.X_OK):
            return str(javacmd_aix)
        
        print(f"Lỗi: JAVA_HOME đã được đặt ({java_home}) nhưng không tìm thấy lệnh 'java' hợp lệ.")
        sys.exit(1)
    javacmd = shutil.which("java")
    if javacmd:
        return javacmd
    
        print("Lỗi: JAVA_HOME không được đặt và lệnh 'java' không được tìm thấy trong PATH.")
    print("Vui lòng cài đặt Java và thiết lập biến môi trường JAVA_HOME hoặc PATH.")
    sys.exit(1)

JAVACMD = find_java_executable()
print(f"[{APP_NAME}] JAVACMD đã được xác định: {JAVACMD}")

APP_ARGS: List[str] = sys.argv[1:]

JAVA_OPTS = os.environ.get('JAVA_OPTS', '').split()     #"--enable-native-access=ALL-UNNAMED", os.environ.get('JAVA_OPTS', '').split()
CIC_FLOW_METER_OPTS = os.environ.get('CIC_FLOW_METER_OPTS', '').split()

default_jvm_opt = DEFAULT_JVM_OPTS_TEMPLATE.format(APP_HOME.as_posix())
print("delf jcvm lb native :", default_jvm_opt)

cmd = [
    JAVACMD,
    *MEMORY_OPTS,                           # [-Xmx10g, -Xms512m]
    default_jvm_opt,                        # [-Djava.library.path=...]
    JAVA_OPTS,
    #*JAVA_OPTS,                             # Các tùy chọn JVM từ ENV
    *CIC_FLOW_METER_OPTS,                   # Các tùy chọn FlowMeter từ ENV
    "-classpath", CLASSPATH,                # CLASSPATH đã xây dựng
    MAIN_CLASS,                             # Tên lớp chính của Java
    *APP_ARGS                               # Các tham số người dùng truyền vào
]

cmd = [arg for arg in cmd if arg]

# --- 5. Thực thi lệnh Java ---
try:
    print(f"\n[{APP_NAME}] Đang thực thi lệnh:")
    # In 1p nếu nhiều lib
    # print(" ".join(FINAL_COMMAND[:3]) + " ... " + " ".join(FINAL_COMMAND[-3:]))
    
    result = subprocess.run(
        cmd,
        check=True,
        text=True,
    )
    
    print(f"\n[{APP_NAME}] Ứng dụng Java đã kết thúc thành công. Mã thoát: {result.returncode}")

except subprocess.CalledProcessError as e:
    print(f"\n[{APP_NAME}] LỖI: Lệnh Java thất bại với mã thoát {e.returncode}.")
    sys.exit(e.returncode)
    
except FileNotFoundError as e:
    print(f"\n[{APP_NAME}] LỖI Hệ thống: Không thể tìm thấy hoặc thực thi file: {e}")
    sys.exit(1)

sys.exit(0)