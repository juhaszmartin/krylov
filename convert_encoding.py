with open('extracted_pdfs.txt', 'r', encoding='utf-16') as f:
    text = f.read()
with open('extracted_pdfs_utf8.txt', 'w', encoding='utf-8') as g:
    g.write(text)
print("Done, length:", len(text))
