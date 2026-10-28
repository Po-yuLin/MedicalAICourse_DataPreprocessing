import sys, nbformat, time, os, shutil
from nbclient import NotebookClient
src, dst = sys.argv[1], sys.argv[2]
shutil.rmtree(os.path.expanduser("~/AI_preprocessing_class"), ignore_errors=True)
nb = nbformat.read(src, as_version=4)
t = time.time()
c = NotebookClient(nb, timeout=900, kernel_name="python3", allow_errors=True)
c.execute()
print("total", round(time.time() - t), "s")
nbformat.write(nb, dst)
bad = 0
for i, cell in enumerate(nb.cells):
    if cell.cell_type != "code": continue
    for o in cell.get("outputs", []):
        if o.output_type == "error":
            bad += 1; print(f"--- ERROR in cell {i}: {o.ename}: {o.evalue[:300]}"); print(cell.source[:120].replace("\n"," | "))
print("errors:", bad)
