import pypdf

def extract_cv_text():
    pdf_path = r"c:\Users\Welcome\TS_SCLF\Deep_Shekhar_Halder_Resume (1).pdf"
    out_path = r"c:\Users\Welcome\TS_SCLF\cv_text.txt"
    reader = pypdf.PdfReader(pdf_path)
    with open(out_path, "w", encoding="utf-8") as f:
        for i, page in enumerate(reader.pages):
            f.write(f"\n--- PAGE {i+1} ---\n")
            f.write(page.extract_text() or "")
    print(f"Extracted CV to {out_path}")

if __name__ == "__main__":
    extract_cv_text()
