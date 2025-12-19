import sys
from pathlib import Path

def appendPath():
    
    current_dir = Path.cwd()
    folders = [item.resolve() for item in current_dir.iterdir() if item.is_dir()]
    
    for fol in folders:
        print(f"Adding folder \"{fol}\" to syspath")
        sys.path.append(str(fol))
