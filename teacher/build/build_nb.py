import json, base64, zlib, re, sys
import nbformat as nbf
import pandas as pd
import os
R = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")) + "/"
ns = {}
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "cells_main.py"), encoding="utf-8").read(), ns)
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "cells_main2.py"), encoding="utf-8").read(), ns)
CELLS = ns["CELLS"]

def b64(path):
    return base64.b64encode(zlib.compress(open(path, "rb").read(), 9)).decode()
emb = {n: b64(R + "data/" + n) for n in ["heart_dirty_small.csv", "heart_dirty_full.csv"]}
emb_src = "{\n" + ",\n".join(f'    "{k}":\n        "{v}"' for k, v in emb.items()) + "\n}"

core = {k: open(os.path.join(os.path.dirname(os.path.abspath(__file__)), f"core_{k}.py"), encoding="utf-8").read() for k in ["clean", "ml", "prep"]}
# core_clean 需要 pd/np（已在 0-1 載入）；core_ml 自帶 torch 載入
def fill(src):
    return (src.replace("__EMBEDDED__", emb_src).replace("__CORE_CLEAN__", core["clean"])
               .replace("__CORE_ML__", core["ml"]).replace("__CORE_PREP__", core["prep"]))

# 老師版附上完整答案
key = pd.read_csv(R + "teacher/answer_key_small.csv")
names = {"missing": "缺值", "impossible": "不可能值", "unit": "單位錯誤", "text_in_number": "數字欄位混入文字",
         "category_spelling": "類別寫法", "duplicate": "重複列"}
key["error_type"] = key["error_type"].map(names)
tbl = "| patient_id | 欄位 | 類型 | 髒值 | 原本的值 |\n|---|---|---|---|---|\n" + "\n".join(
    f"| {r.patient_id} | {r.column} | {r.error_type} | `{r.dirty_value}` | `{r.clean_value}` |" for r in key.fillna("").itertuples())
ANSWER = "**🎓 老師備註：小資料集（40 位）的注入錯誤完整答案**（另外 P088 的 thal、P167 的 ca 是資料集本來就缺）\n\n" + tbl

def build(teacher, path):
    nb = nbf.v4.new_notebook()
    nb.metadata = {"colab": {"provenance": [], "toc_visible": True}, "kernelspec": {"name": "python3", "display_name": "Python 3"},
                   "language_info": {"name": "python"}}
    for typ, src, fl in CELLS:
        if fl["teacher"] and not teacher:
            continue
        src = fill(src)
        if typ == "md":
            nb.cells.append(nbf.v4.new_markdown_cell(src))
        else:
            c = nbf.v4.new_code_cell(src)
            if fl.get("form"):
                c.metadata["cellView"] = "form"
            nb.cells.append(c)
        if teacher and "完整答案：`teacher/answer_key_small.csv`" in src:
            nb.cells.append(nbf.v4.new_markdown_cell(ANSWER))
    nbf.write(nb, path)
    print(path, len(nb.cells), "cells")

build(False, R + "notebooks/AI_preprocessing_student.ipynb")
build(True, R + "notebooks/AI_preprocessing_teacher.ipynb")
