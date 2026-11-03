import re
# ===== 資料字典與規則（檢查器與自動清理共用）=====
NUM_COLS = ["age", "resting_bp", "chol", "max_hr", "oldpeak", "ca"]          # 連續數值／計數
BIN_COLS = ["fbs", "exang"]                                                  # 0/1
CODE_COLS = ["rest_ecg", "slope"]                                            # 用數字編碼的類別
CAT_COLS = {"sex": ["male", "female"],
            "chest_pain": ["typical", "nontypical", "nonanginal", "asymptomatic"],
            "thal": ["normal", "fixed", "reversible"],
            "disease": ["yes", "no"]}
RANGES = {"age": (18, 100), "resting_bp": (70, 250), "chol": (100, 600),
          "max_hr": (60, 220), "oldpeak": (0, 8), "ca": (0, 3),
          "fbs": (0, 1), "exang": (0, 1), "rest_ecg": (0, 2), "slope": (1, 3)}
ALL_NUMERIC = NUM_COLS + BIN_COLS + CODE_COLS
MISSING_TOKENS = {"", "?", "na", "n/a", "nan", "null", "none", "unknown", "-", "--"}


def _is_missing(x):
    return pd.isna(x) or str(x).strip().lower() in MISSING_TOKENS


def _num(v):
    try:
        return float(str(v).strip())
    except ValueError:
        return None


CODE_ALLOWED = {"fbs": {0, 1}, "exang": {0, 1}, "rest_ecg": {0, 1, 2}, "slope": {1, 2, 3}, "ca": {0, 1, 2, 3}}
ISSUE_ORDER = ["缺值", "數字欄位混入文字", "不可能的數值", "單位疑似錯誤", "類別寫法不一致", "重複列",
               "不該改的被改了", "這格有固定答案，但填的不對", "填的值不合理", "誤刪正常的病人", "多出不認識的病人"]


def _cell_kind(col, dirty_val, truth_val):
    """這一格原本是：clean（本來就對）/ fixed（有固定答案）/ fixed_unit（單位換算）/ imputable（缺值或不可能值，沒有唯一答案）"""
    tv = "" if pd.isna(truth_val) else str(truth_val).strip()
    if tv == "":
        return "imputable"
    dv = "" if pd.isna(dirty_val) else str(dirty_val)
    if col in ALL_NUMERIC:
        x = _num(dv)
        if x is not None and abs(x - float(tv)) < 1e-9:
            return "clean"
        m = re.search(r"-?\d+\.?\d*", dv)
        if x is None and m and not _is_missing(dv) and abs(float(m.group()) - float(tv)) < 1e-9:
            return "fixed"
        if col == "chol" and x is not None and 0 < x < 30 and abs(x * 38.67 - float(tv)) <= 2:
            return "fixed_unit"
        return "imputable"
    if dv == tv:
        return "clean"
    return "imputable" if _is_missing(dv) else "fixed"


def _accept(col, v, kind, truth_val, bands):
    """學生填的值可不可以接受？有固定答案的要一模一樣；沒有唯一答案的，只要『合理』。"""
    if kind in ("clean", "fixed", "fixed_unit"):
        if col in ALL_NUMERIC:
            x = _num(v)
            return x is not None and abs(x - float(truth_val)) <= (2 if kind == "fixed_unit" else 1e-9)
        return str(v) == str(truth_val)
    if col in ALL_NUMERIC:
        x = _num(v)
        if x is None:
            return False
        if col in CODE_ALLOWED:
            return x == int(x) and int(x) in CODE_ALLOWED[col]
        lo, hi = bands[col]
        return lo <= x <= hi
    return str(v) in CAT_COLS[col]


def find_issues(df, dirty=None, truth=None, bands=None):
    """找出表格裡所有問題。只給 df：只做「格式與範圍」檢查。
    再給 dirty（原始髒資料）、truth（答案）、bands（合理範圍）：還會檢查「不該改的被改了」「有固定答案但填錯」等。"""
    d = df.reset_index(drop=True)
    out = []
    def add(row, pid, col, typ, detail, expected=""):
        out.append(dict(row=row, patient_id=pid, column=col, type=typ, detail=detail, expected=expected))
    need = ["patient_id"] + ALL_NUMERIC + list(CAT_COLS)
    lost = [c for c in need if c not in d.columns]
    if lost:
        return None, lost
    for i, row in d.iterrows():
        pid = str(row["patient_id"]).strip()
        for col in ALL_NUMERIC:
            v = row[col]
            if _is_missing(v):
                add(i, pid, col, "缺值", "空白或 ?、NA、unknown…")
                continue
            x = _num(v)
            if x is None:
                add(i, pid, col, "數字欄位混入文字", f"「{v}」不是純數字")
                continue
            lo, hi = RANGES[col]
            if col == "chol" and 0 < x < 30:
                add(i, pid, col, "單位疑似錯誤", f"{x:g} 看起來是 mmol/L，要的是 mg/dL")
            elif not (lo <= x <= hi):
                add(i, pid, col, "不可能的數值", f"{x:g} 超出合理範圍 {lo}–{hi}")
        for col, allowed in CAT_COLS.items():
            v = row[col]
            if _is_missing(v):
                add(i, pid, col, "缺值", "空白或 NA…")
            elif str(v) not in allowed:
                add(i, pid, col, "類別寫法不一致", f"「{v}」不在 {allowed} 之中")
    for i in d.index[d.duplicated("patient_id", keep="first")]:
        add(i, str(d.at[i, "patient_id"]).strip(), "(整列)", "重複列", "同一位病人出現不只一次")

    deleted_info = None
    if truth is not None and dirty is not None:
        dirty_ix = dirty.assign(patient_id=dirty["patient_id"].str.strip()).drop_duplicates("patient_id").set_index("patient_id")
        truth_ix = truth.assign(patient_id=truth["patient_id"].str.strip()).set_index("patient_id")
        flagged = {(o["row"], o["column"]) for o in out}
        cols = ALL_NUMERIC + list(CAT_COLS)
        for i, row in d.iterrows():
            pid = str(row["patient_id"]).strip()
            if pid not in truth_ix.index:
                add(i, pid, "(整列)", "多出不認識的病人", "這個 patient_id 不在原始資料裡")
                continue
            for col in cols:
                if (i, col) in flagged:
                    continue
                tv, v = truth_ix.at[pid, col], row[col]
                kind = _cell_kind(col, dirty_ix.at[pid, col], tv)
                if _accept(col, v, kind, tv, bands):
                    continue
                if kind == "clean":
                    add(i, pid, col, "不該改的被改了", f"這格原本就是正常的，你把它改成了「{v}」", f"{tv}")
                elif kind in ("fixed", "fixed_unit"):
                    add(i, pid, col, "這格有固定答案，但填的不對", f"你填的是「{v}」", f"{tv}")
                else:
                    rng = f"{bands[col][0]:g}–{bands[col][1]:g}" if col in bands else "合法的類別"
                    add(i, pid, col, "填的值不合理", f"你填的是「{v}」，不在合理的範圍內", f"任何落在 {rng} 的值")
        present = set(d["patient_id"].astype(str).str.strip())
        gone = [p for p in truth_ix.index if p not in present]
        n_bad = 0
        for pid in gone:
            had_problem = any(_cell_kind(c, dirty_ix.at[pid, c], truth_ix.at[pid, c]) != "clean" for c in cols)
            if had_problem:
                n_bad += 1
            else:
                add(-1, pid, "(整列)", "誤刪正常的病人", "這位病人的資料原本完全正常，不該刪")
        deleted_info = (len(gone), n_bad)
    res = pd.DataFrame(out, columns=["row", "patient_id", "column", "type", "detail", "expected"])
    return res, deleted_info


def check_cleaned(df, hint=1, dirty=None, truth=None, bands=None):
    """檢查清理後的表格。hint=1 只報數量；2 加上欄位；3 再加上 patient_id；4 連答案都告訴你。"""
    res, info = find_issues(df, dirty, truth, bands)
    if res is None:
        print(f"❌ 找不到欄位 {info}。請不要更改標題列（第 1 列）的欄位名稱。")
        return False
    if truth is not None and info and info[0]:
        print(f"（你刪掉了 {info[0]} 位病人，其中 {info[1]} 位原本就有問題。）")
    if res.empty:
        print(f"✅ 全部通過！共 {len(df)} 列，沒有發現問題。")
        return True
    print(f"❌ 還有 {len(res)} 個問題：")
    for kind in [k for k in ISSUE_ORDER if k in set(res["type"])]:
        g = res[res["type"] == kind]
        line = f"  • {kind}：{len(g)} 個"
        if hint >= 2:
            line += f"　欄位：{', '.join(sorted(g['column'].unique()))}"
        print(line)
        if hint >= 3:
            for _, r in g.iterrows():
                extra = f"　→ 答案：{r['expected']}" if (hint >= 4 and r["expected"]) else ""
                print(f"      - {r['patient_id']} 的 {r['column']}：{r['detail']}{extra}")
    return False


def auto_clean(raw, fill=None):
    """用程式一次清完：同樣的規則，對 300 列跟 300 萬列都一樣快。
    fill：補缺值用的數字（中位數/眾數）。None＝由這批資料自己算；推論時要傳入「訓練時算好的」fill。"""
    d = raw.copy().astype(str)
    d = d.apply(lambda s: s.str.strip())
    d = d.replace({t: np.nan for t in ["", "?", "NA", "N/A", "na", "n/a", "null", "unknown", "-", "--"]})
    d = d.drop_duplicates(subset="patient_id", keep="first")           # 重複列
    for c in ALL_NUMERIC:                                              # 取出數字（去掉 mmHg、bpm…）
        d[c] = pd.to_numeric(d[c].str.extract(r"(-?\d+\.?\d*)")[0], errors="coerce")
    d.loc[d["chol"] < 30, "chol"] *= 38.67                             # mmol/L → mg/dL
    for c, (lo, hi) in RANGES.items():                                 # 不可能的值 → 視為缺值
        d.loc[(d[c] < lo) | (d[c] > hi), c] = np.nan
    fix = {                                                            # 類別寫法統一
        "sex": {"m": "male", "male": "male", "f": "female", "female": "female"},
        "chest_pain": {"typical": "typical", "typical angina": "typical",
                       "nontypical": "nontypical", "non-typical": "nontypical", "atypical": "nontypical",
                       "nonanginal": "nonanginal", "non-anginal": "nonanginal", "non anginal": "nonanginal",
                       "asymptomatic": "asymptomatic"},
        "thal": {"normal": "normal", "fixed": "fixed", "fixed defect": "fixed",
                 "reversible": "reversible", "reversable": "reversible", "reversible defect": "reversible"},
        "disease": {"yes": "yes", "y": "yes", "1": "yes", "no": "no", "n": "no", "0": "no"},
    }
    cats = [c for c in CAT_COLS if c in d.columns]
    for c in cats:
        d[c] = d[c].str.lower().map(fix[c])
    if fill is None:                                                   # 補缺值：數字用中位數、類別用眾數
        fill = {c: d[c].median() for c in ALL_NUMERIC}
        fill.update({c: d[c].mode()[0] for c in cats})
    for c in ALL_NUMERIC + cats:
        d[c] = d[c].fillna(fill[c])
    d["chol"] = d["chol"].round(0)
    d[ALL_NUMERIC] = d[ALL_NUMERIC].astype(float)
    d = d.reset_index(drop=True)
    d.attrs["fill"] = fill
    return d
