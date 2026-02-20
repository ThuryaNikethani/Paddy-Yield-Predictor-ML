import pdfplumber
import os

pdfs = [f for f in os.listdir('.') if f.endswith('.pdf')]
print(f'Found {len(pdfs)} PDFs:')
for p in pdfs:
    print(f'  - {p}')

sep = "=" * 70

for pdf_file in pdfs:
    print(f'\n{sep}')
    print(f'FILE: {pdf_file}')
    print(f'{sep}')
    try:
        with pdfplumber.open(pdf_file) as pdf:
            for i, page in enumerate(pdf.pages[:3]):
                text = page.extract_text()
                if text:
                    print(f'\n--- Page {i+1} ---')
                    print(text[:3000])
                tables = page.extract_tables()
                if tables:
                    print(f'\n--- Tables on Page {i+1} ---')
                    for ti, table in enumerate(tables[:2]):
                        print(f'Table {ti+1}:')
                        for row in table[:20]:
                            print(row)
    except Exception as e:
        print(f'Error reading {pdf_file}: {e}')
