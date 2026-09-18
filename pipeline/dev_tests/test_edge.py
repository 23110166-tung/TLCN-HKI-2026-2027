import subprocess
import os

edge_path = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
html_path = os.path.abspath("test_page.html")
with open(html_path, "w", encoding="utf-8") as f:
    f.write("<!DOCTYPE html><html><body style='background:#f0f2f5;font-family:Arial;'><h1>TRƯỜNG ĐẠI HỌC CÔNG NGHỆ KỸ THUẬT TP.HCM</h1></body></html>")

img_out = os.path.abspath("test_out.png")
cmd = [
    edge_path,
    "--headless=new",
    "--disable-gpu",
    "--hide-scrollbars",
    "--window-size=1080,800",
    f"--screenshot={img_out}",
    html_path
]
res = subprocess.run(cmd, capture_output=True, text=True)
print("Return code:", res.returncode)
print("Output exists:", os.path.exists(img_out))
if os.path.exists(img_out):
    print("Size:", os.path.getsize(img_out))
