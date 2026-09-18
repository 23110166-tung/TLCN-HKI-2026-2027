# -*- coding: utf-8 -*-
"""
High-Performance PDF Converter for DOCX files
Keeps Word COM open during batch conversion for fast and reliable PDF export.
"""

import os
import sys
import subprocess
import shutil
import time

class PDFConverter:
    def __init__(self):
        self.word_app = None
        self._init_word()
        
    def _init_word(self):
        try:
            import win32com.client
            import pythoncom
            pythoncom.CoInitialize()
            self.word_app = win32com.client.DispatchEx("Word.Application")
            self.word_app.Visible = False
            self.word_app.DisplayAlerts = False
        except Exception as e:
            print(f"[Notice] Could not initialize Word COM: {e}. Will use CLI fallback.")
            self.word_app = None

    def convert(self, docx_path, pdf_path):
        docx_path = os.path.abspath(docx_path)
        pdf_path = os.path.abspath(pdf_path)
        os.makedirs(os.path.dirname(pdf_path), exist_ok=True)
        
        # 1. Use Word COM if initialized
        if self.word_app:
            try:
                doc = self.word_app.Documents.Open(docx_path, ReadOnly=True, ConfirmConversions=False)
                # 17 = wdFormatPDF
                doc.SaveAs2(pdf_path, FileFormat=17)
                doc.Close(False)
                if os.path.exists(pdf_path) and os.path.getsize(pdf_path) > 0:
                    return True
            except Exception as e:
                print(f"[Warning] Word COM failed for {os.path.basename(docx_path)}: {e}")

        # 2. Try LibreOffice
        soffice_cmd = shutil.which("libreoffice") or shutil.which("soffice")
        if soffice_cmd:
            try:
                out_dir = os.path.dirname(pdf_path)
                cmd = [soffice_cmd, "--headless", "--convert-to", "pdf", docx_path, "--outdir", out_dir]
                subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60)
                expected_pdf = os.path.join(out_dir, os.path.splitext(os.path.basename(docx_path))[0] + ".pdf")
                if os.path.exists(expected_pdf):
                    if expected_pdf != pdf_path:
                        shutil.move(expected_pdf, pdf_path)
                    return True
            except Exception as e:
                print(f"[Warning] LibreOffice failed: {e}")

        # 3. Try docx2pdf
        try:
            from docx2pdf import convert
            convert(docx_path, pdf_path)
            if os.path.exists(pdf_path) and os.path.getsize(pdf_path) > 0:
                return True
        except Exception as e:
            print(f"[Error] docx2pdf failed: {e}")

        raise RuntimeError(f"Could not convert {docx_path} to PDF.")

    def close(self):
        if self.word_app:
            try:
                self.word_app.Quit()
            except:
                pass
            self.word_app = None

def convert_docx_to_pdf(docx_path, pdf_path):
    converter = PDFConverter()
    try:
        res = converter.convert(docx_path, pdf_path)
        return res
    finally:
        converter.close()
