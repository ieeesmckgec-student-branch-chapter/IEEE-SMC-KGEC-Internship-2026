import os
import shutil

final_dir = r"c:\Users\Welcome\TS_SCLF\final_submission"
src_pdf = os.path.join(final_dir, "IEEE-SMC-SBC-KGEC-12 (1).pdf")

if not os.path.exists(src_pdf):
    # Fallback search
    for f in os.listdir(final_dir):
        if f.endswith(".pdf"):
            src_pdf = os.path.join(final_dir, f)
            break

if os.path.exists(src_pdf):
    # 1. Clean Group Name PDF
    dst_group = os.path.join(final_dir, "IEEE-SMC-SBC-KGEC-12.pdf")
    shutil.copyfile(src_pdf, dst_group)
    print(f"Created: {dst_group}")
    
    # 2. Project Name PDF (as mentioned in latest WhatsApp notice)
    dst_proj = os.path.join(final_dir, "ALAS_SCLF_Polar_Decoders.pdf")
    shutil.copyfile(src_pdf, dst_proj)
    print(f"Created: {dst_proj}")

print("Final submission files ready.")
