import fitz
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

pdfs = ["Gaps.pdf", "ABF16 (2).pdf", "s00211-023-01368-6.pdf"]

for pdf_name in pdfs:
    print(f"\n{'='*80}")
    print(f"FILE: {pdf_name}")
    print(f"{'='*80}")
    try:
        doc = fitz.open(pdf_name)
        for i, page in enumerate(doc):
            print(f"\n--- PAGE {i+1} ---")
            text = page.get_text()
            print(text)
        doc.close()
    except Exception as e:
        print(f"ERROR reading {pdf_name}: {e}")
