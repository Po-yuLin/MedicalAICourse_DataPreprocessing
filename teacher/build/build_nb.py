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

core = {k: open(os.path.join(os.path.dirname(os.path.abspath(__file__)), f"core_{k}.py"), encoding="utf-8").read() for k in ["clean", "view", "ml", "prep"]}
# core_clean 需要 pd/np（已在 0-1 載入）；core_ml 自帶 torch 載入
# 檢查器用的「標準答案」與「合理範圍」
import numpy as np
dirty_small = pd.read_csv(R + "data/heart_dirty_small.csv", dtype=str, keep_default_na=False)
clean_all = pd.read_csv(R + "teacher/heart_clean_original.csv", dtype=str, keep_default_na=False)
truth_small = clean_all[clean_all.patient_id.isin(dirty_small.patient_id)].reset_index(drop=True)
truth_b64 = base64.b64encode(zlib.compress(truth_small.to_csv(index=False).encode("utf-8"), 9)).decode()
cf = pd.read_csv(R + "teacher/heart_clean_original.csv")
bands = {c: [round(float(np.percentile(cf[c].dropna(), 1)), 1), round(float(np.percentile(cf[c].dropna(), 99)), 1)]
         for c in ["age", "resting_bp", "chol", "max_hr", "oldpeak", "ca"]}

# 「大家來找碴」要秀的病人（涵蓋各種錯誤類型），以及 sex／disease 轉換前後對照要秀的病人
DEMO_IDS = ["P005", "P110", "P021", "P111", "P135", "P145", "P058", "P079", "P189", "P247", "P128"]
CONV_IDS = ["P005", "P030", "P110", "P167", "P135", "P247", "P132", "P109", "P044", "P036"]
for _i in DEMO_IDS + CONV_IDS:
    assert _i in set(dirty_small.patient_id), f"{_i} 不在小資料集裡，請改 build_nb.py 的 DEMO_IDS/CONV_IDS"

def fill(src):
    return (src.replace("__EMBEDDED__", emb_src).replace("__TRUTH__", repr(truth_b64)).replace("__BANDS__", repr(bands))
               .replace("__DEMO_IDS__", repr(DEMO_IDS)).replace("__CONV_IDS__", repr(CONV_IDS))
               .replace("__CORE_CLEAN__", core["clean"]).replace("__CORE_VIEW__", core["view"])
               .replace("__CORE_ML__", core["ml"]).replace("__CORE_PREP__", core["prep"]))

# markdown 表格 → 大字 HTML 表格（Colab 的 markdown 無法調整字體大小，所以用 HTML 表格）
MD_TABLE_PX = 22
def _inline(t):
    t = t.strip()
    t = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t)
    t = re.sub(r"`([^`]+)`", r"<code>\1</code>", t)
    return t
def md_tables_to_html(src):
    lines, out, i = src.split("\n"), [], 0
    while i < len(lines):
        if lines[i].lstrip().startswith("|") and i + 1 < len(lines) and re.match(r"^\s*\|[\s\-|:]+\|\s*$", lines[i + 1]):
            head = [c for c in lines[i].strip().strip("|").split("|")]
            i += 2
            rows = []
            while i < len(lines) and lines[i].lstrip().startswith("|"):
                rows.append(lines[i].strip().strip("|").split("|")); i += 1
            th = "".join(f'<th style="border:1px solid #888;padding:6px 12px;background:#e8eaf6;color:#000;text-align:left">{_inline(c)}</th>' for c in head)
            body = "".join("<tr>" + "".join(f'<td style="border:1px solid #888;padding:6px 12px;background:#fff;color:#000;text-align:left">{_inline(c)}</td>' for c in r) + "</tr>" for r in rows)
            out += ["", f'<table style="font-size:{MD_TABLE_PX}px;border-collapse:collapse;margin:8px 0"><tr>{th}</tr>{body}</table>', ""]
        else:
            out.append(lines[i]); i += 1
    return "\n".join(out)

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
            nb.cells.append(nbf.v4.new_markdown_cell(md_tables_to_html(src)))
        else:
            c = nbf.v4.new_code_cell(src)
            if fl.get("form"):
                c.metadata["cellView"] = "form"
            nb.cells.append(c)
        if teacher and "完整答案：`teacher/answer_key_small.csv`" in src:
            nb.cells.append(nbf.v4.new_markdown_cell(md_tables_to_html(ANSWER)))
    nbf.write(nb, path)
    print(path, len(nb.cells), "cells")

build(False, R + "notebooks/AI_preprocessing_student.ipynb")
build(True, R + "notebooks/AI_preprocessing_teacher.ipynb")
