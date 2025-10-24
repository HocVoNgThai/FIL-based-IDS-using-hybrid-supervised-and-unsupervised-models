import os
import sys
import subprocess
import shutil
from pathlib import Path

def main():
    script_dir = Path(__file__).resolve().parent
    print(script_dir)
    app_home = script_dir.parent
    lib_dir = app_home / "lib"
    native_dir = lib_dir / "native"

    java_home = os.environ.get("JAVA_HOME")
    if java_home:
        java_exe = Path(java_home) / "bin" / "java.exe"
    else:
        java_exe = shutil.which("java")

    if not java_exe or not Path(java_exe).exists():
        print("❌ ERROR: JAVA_HOME not set or java not found in PATH")
        sys.exit(1)

    default_jvm_opts = f'-Djava.library.path={native_dir}'
    jars = [str(p) for p in lib_dir.glob("*.jar")]
    classpath = ";".join(jars)

    main_class = "cic.cs.unb.ca.ifm.Cmd"
    cmd_args = sys.argv[1:]

    # cmd
    cmd = [
        str(java_exe),
        default_jvm_opts,
        "-classpath", classpath,
        main_class
    ] + cmd_args

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Java process exited with error code {e.returncode}")
        sys.exit(e.returncode)

if __name__ == "__main__":
    main()
