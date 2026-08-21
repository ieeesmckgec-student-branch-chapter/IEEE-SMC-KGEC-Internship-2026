import os
import shutil

base_dir = r"c:\Users\Welcome\TS_SCLF"

# 1. Check for D: drive backup
d_drive = r"D:\IEEE_SMC_KGEC_Backup"
backup_dir = d_drive if os.path.exists("D:\\") else os.path.join(base_dir, "backup_d_drive")

try:
    os.makedirs(backup_dir, exist_ok=True)
    # Copy final PDFs and Presentation to D: drive backup
    final_sub_dir = os.path.join(base_dir, "final_submission")
    if os.path.exists(final_sub_dir):
        for f in os.listdir(final_sub_dir):
            src_f = os.path.join(final_sub_dir, f)
            dst_f = os.path.join(backup_dir, f)
            shutil.copy2(src_f, dst_f)
            print(f"Backed up to D:\\ drive ({backup_dir}): {f}")
except Exception as e:
    print(f"Notice during D:\\ backup: {e}")

# 2. Setup Official GitHub Structure
# Recommended structure:
# TS_SCLF/
# ├── README.md
# ├── requirements.txt
# ├── src/
# └── docs/

src_dir = os.path.join(base_dir, "src")
docs_dir = os.path.join(base_dir, "docs")
os.makedirs(src_dir, exist_ok=True)
os.makedirs(docs_dir, exist_ok=True)

# Move source_code -> src/
old_src = os.path.join(base_dir, "source_code")
if os.path.exists(old_src):
    for item in os.listdir(old_src):
        s = os.path.join(old_src, item)
        d = os.path.join(src_dir, item)
        if not os.path.exists(d):
            shutil.move(s, d)
            print(f"Moved to src/: {item}")
    try:
        shutil.rmtree(old_src)
    except Exception:
        pass

# Move build_scripts -> src/scripts
old_build = os.path.join(base_dir, "build_scripts")
if os.path.exists(old_build):
    dst_scripts = os.path.join(src_dir, "build_scripts")
    os.makedirs(dst_scripts, exist_ok=True)
    for item in os.listdir(old_build):
        s = os.path.join(old_build, item)
        d = os.path.join(dst_scripts, item)
        if not os.path.exists(d):
            shutil.move(s, d)
            print(f"Moved to src/build_scripts/: {item}")
    try:
        shutil.rmtree(old_build)
    except Exception:
        pass

# Move figures -> docs/figures
old_figs = os.path.join(base_dir, "figures")
if os.path.exists(old_figs):
    dst_figs = os.path.join(docs_dir, "figures")
    os.makedirs(dst_figs, exist_ok=True)
    for item in os.listdir(old_figs):
        s = os.path.join(old_figs, item)
        d = os.path.join(dst_figs, item)
        if not os.path.exists(d):
            shutil.move(s, d)
    try:
        shutil.rmtree(old_figs)
        print("Moved figures/ into docs/figures/")
    except Exception:
        pass

# Move source_paper -> docs/source_paper
old_paper = os.path.join(base_dir, "source_paper")
if os.path.exists(old_paper):
    dst_paper = os.path.join(docs_dir, "source_paper")
    os.makedirs(dst_paper, exist_ok=True)
    for item in os.listdir(old_paper):
        s = os.path.join(old_paper, item)
        d = os.path.join(dst_paper, item)
        if not os.path.exists(d):
            shutil.move(s, d)
    try:
        shutil.rmtree(old_paper)
        print("Moved source_paper/ into docs/source_paper/")
    except Exception:
        pass

# Move templates_and_admin -> docs/templates
old_tpl = os.path.join(base_dir, "templates_and_admin")
if os.path.exists(old_tpl):
    dst_tpl = os.path.join(docs_dir, "templates")
    os.makedirs(dst_tpl, exist_ok=True)
    for item in os.listdir(old_tpl):
        s = os.path.join(old_tpl, item)
        d = os.path.join(dst_tpl, item)
        if not os.path.exists(d):
            shutil.move(s, d)
    try:
        shutil.rmtree(old_tpl)
        print("Moved templates_and_admin/ into docs/templates/")
    except Exception:
        pass

# Move other_resources -> docs/other_resources
old_other = os.path.join(base_dir, "other_resources")
if os.path.exists(old_other):
    dst_other = os.path.join(docs_dir, "other_resources")
    os.makedirs(dst_other, exist_ok=True)
    for item in os.listdir(old_other):
        s = os.path.join(old_other, item)
        d = os.path.join(dst_other, item)
        if not os.path.exists(d):
            shutil.move(s, d)
    try:
        shutil.rmtree(old_other)
    except Exception:
        pass

print("\n=== Official GitHub Recommended Structure Created! ===")
print("Folder format:")
print("TS_SCLF/")
print("├── README.md")
print("├── requirements.txt")
print("├── src/")
print("└── docs/")
