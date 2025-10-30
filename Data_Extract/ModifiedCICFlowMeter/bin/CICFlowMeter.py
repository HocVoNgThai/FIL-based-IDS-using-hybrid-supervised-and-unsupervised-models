import os
import subprocess

APP_HOME = r"C:\Users\hoang\Code\FIL-based-IDS-using-hybrid-supervised-and-unsupervised-models\Data_Extract\ModifiedCICFlowMeter"
JAVA_EXE = r"C:\Program Files\Java\jdk-25\bin\java.exe"
LIB_DIR = os.path.join(APP_HOME, "lib")


classpath = ";".join([
    os.path.join(LIB_DIR, jar)
    for jar in os.listdir(LIB_DIR)
    if jar.endswith(".jar")
])


native_lib_path = os.path.join(LIB_DIR, "native")
jvm_opts = f'-Djava.library.path="{native_lib_path}"'
main_class = "cic.cs.unb.ca.ifm.App"
# cmd_args = ["-i", r"C:\pcaps\example.pcap", "-c", "flows"]

cmd = [
    JAVA_EXE,
    "-Xmx8g -Xms512m",
    jvm_opts,
    "-classpath", classpath,
    main_class
] 
# + cmd_args


# print(">> Running command:\n", " ".join(cmd))

try:
    result = subprocess.run(cmd, capture_output=True, text=True)

except Exception as e:
    print("[i] Information: ", e)
    
# print("=== STDOUT ===")
# print(result.stdout)
# print("=== STDERR ===")
# print(result.stderr)
