from xhtml2pdf import pisa
import os

def convert_html_to_pdf(html_path, pdf_path):
    with open(html_path, "r", encoding="utf-8") as html_file:
        html_content = html_file.read()
        
    # Open PDF file for writing binary
    with open(pdf_path, "wb") as pdf_file:
        pisa_status = pisa.CreatePDF(
            src=html_content,
            dest=pdf_file
        )
    return pisa_status.err

if __name__ == "__main__":
    html_path = "alas_sclf_paper.html"
    pdf_path = "alas_sclf_paper.pdf"
    err = convert_html_to_pdf(html_path, pdf_path)
    if not err:
        print(f"PDF generated successfully: {os.path.abspath(pdf_path)}")
    else:
        print(f"Error occurred: {err}")
