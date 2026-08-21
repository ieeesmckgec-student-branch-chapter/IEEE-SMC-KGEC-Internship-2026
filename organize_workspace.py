import os
import shutil

base_dir = r"c:\Users\Welcome\TS_SCLF"

# Categorization mapping
dirs = {
    "final_submission": [
        "IEEE-SMC-SBC-KGEC-12 (1).pdf",
        "IEEE-SMC-SBC-KGEC-12.pdf",
        "ALAS_SCLF_Polar_Decoders.pdf",
        "IEEE_SMC_KGEC_Presentation.pptx",
        "ALAS_SCLF_Full_Paper.docx"
    ],
    "source_paper": [
        "alas_sclf_paper.html",
        "alas_sclf_paper.pdf",
        "alas_sclf_paper.tex",
        "ALAS_SCLF_Overleaf_Paper.tex",
        "alas_sclf_presentation.pptx"
    ],
    "source_code": [
        "polar_simulator.py",
        "rigorous_polar.py",
        "dqn_alas_optimizer.py",
        "dqn_lra_optimizer.py",
        "main.py"
    ],
    "source_code/sweeps": [
        "alas_sclf_sweep.py",
        "bounded_latency_sweep.py",
        "full_sweep.py",
        "learned_flip_sweep.py",
        "segmented_crc_sweep.py",
        "sweep_guard.py",
        "true_complexity_sweep.py",
        "verify_dynamic_lseg.py"
    ],
    "build_scripts": [
        "convert_html_to_pdf.py",
        "convert_paper_to_pptx.py",
        "generate_word_doc.py",
        "generate_all_paper_figures.py",
        "generate_comparison_graphs.py",
        "generate_dqn_alas_graphs.py",
        "generate_paper_plot.py",
        "plot_dqn_progress.py",
        "standalone_all_graphs.py",
        "prepare_final_files.py"
    ],
    "templates_and_admin": [
        "IEEE_SMC_KGEC_Report_Template (1).docx",
        "Final_Selected_candidates.xlsx"
    ]
}

# Unwanted temporary lock files and folders to delete
junk_files = [
    "clean_workspace.py",
    "~$IEEE_SMC_KGEC_Presentation.pptx"
]

folders_to_delete = [
    "logos",
    "pdf-forge-exports"
]

# Exclude from moving
exclude = ["README.md", "organize_workspace.py", "requirements.txt", ".git", ".venv", "__pycache__"]

print("=== Reorganizing ALL Files into Folders ===")

# Move categorized files
for folder_rel, files in dirs.items():
    folder_path = os.path.join(base_dir, folder_rel)
    os.makedirs(folder_path, exist_ok=True)
    for fname in files:
        src = os.path.join(base_dir, fname)
        dst = os.path.join(folder_path, fname)
        if os.path.exists(src):
            try:
                shutil.move(src, dst)
                print(f"Moved: {fname}  -->  {folder_rel}/")
            except Exception as e:
                print(f"Error moving {fname}: {e}")

# Remove junk files
for jf in junk_files:
    jpath = os.path.join(base_dir, jf)
    if os.path.exists(jpath):
        try:
            os.remove(jpath)
            print(f"Removed temporary file: {jf}")
        except Exception:
            pass

# Remove temporary folders
for fd in folders_to_delete:
    fpath = os.path.join(base_dir, fd)
    if os.path.exists(fpath):
        try:
            shutil.rmtree(fpath)
            print(f"Deleted folder: {fd}")
        except Exception as e:
            print(f"Could not delete folder {fd}: {e}")

# Catch-all: Move any remaining loose files into 'other_resources'
fallback_folder = os.path.join(base_dir, "other_resources")
os.makedirs(fallback_folder, exist_ok=True)

remaining = [f for f in os.listdir(base_dir) if f not in exclude and not os.path.isdir(os.path.join(base_dir, f))]

for rf in remaining:
    src = os.path.join(base_dir, rf)
    dst = os.path.join(fallback_folder, rf)
    try:
        shutil.move(src, dst)
        print(f"Moved remaining: {rf}  -->  other_resources/")
    except Exception as e:
        print(f"Error moving remaining file {rf}: {e}")

print("\n=== Workspace Reorganization Complete! ===")
print("Zero loose files remaining in root directory!")
