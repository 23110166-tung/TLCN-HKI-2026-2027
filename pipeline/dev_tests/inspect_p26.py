import docx, sys, os
sys.stdout.reconfigure(encoding='utf-8')

doc = docx.Document(r'sample/Don_xin_chuyenhedaotao.docx')
for i, p in enumerate(doc.paragraphs):
    if i in [26, 27, 28, 29, 30, 31]:
        print(f"P{i} raw: {repr(p.text)}")
