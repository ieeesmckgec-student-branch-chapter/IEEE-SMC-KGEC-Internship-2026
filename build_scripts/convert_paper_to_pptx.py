import os
import sys

# Redirect execution to build_scripts/convert_paper_to_pptx.py
script_path = os.path.join("build_scripts", "convert_paper_to_pptx.py")
if os.path.exists(script_path):
    with open(script_path, "r", encoding="utf-8") as f:
        code = f.read()
    exec(code)
else:
    print("Error: build_scripts/convert_paper_to_pptx.py not found.")
