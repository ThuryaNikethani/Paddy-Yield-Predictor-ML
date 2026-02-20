"""Extract data from Maha and 1.3-1.6 PDFs"""
import pdfplumber
import os

pdf_files = [
    '2024_2025Maha_Metric.pdf',
    '2024_2025Maha_Imperial.pdf',
    '1.3.pdf',
    '1.4.pdf',
    '1.5.pdf',
    '1.6.pdf'
]

for pdf_file in pdf_files:
    if os.path.exists(pdf_file):
        print("=" * 70)
        print(f"FILE: {pdf_file}")
        print("=" * 70)
        try:
            with pdfplumber.open(pdf_file) as pdf:
                for page in pdf.pages:
                    print(f"\n--- Page {page.page_number} ---")
                    text = page.extract_text()
                    if text:
                        print(text)
                    tables = page.extract_tables()
                    if tables:
                        for i, table in enumerate(tables):
                            print(f"\n--- Table {i+1} on Page {page.page_number} ---")
                            for row in table:
                                print(row)
        except Exception as e:
            print(f"Error: {e}")
    else:
        print(f"File not found: {pdf_file}")
