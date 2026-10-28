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


def check_cleaned(df, hint=1):
    """檢查清理後的表格。hint=1 只報數量；2 加上欄位；3 再加上 patient_id。"""
    issues = []   # (類型, 欄位, patient_id, 說明)
    d = df.copy()
    need = ["patient_id"] + ALL_NUMERIC + list(CAT_COLS)
    lost = [c for c in need if c not in d.columns]
    if lost:
        print(f"❌ 找不到欄位 {lost}。請不要更改標題列（第 1 列）的欄位名稱。")
        return False
    for i, row in d.iterrows():
        pid = row.get("patient_id", f"第{i+2}列")
        for col in ALL_NUMERIC:
            v = row[col]
            if _is_missing(v):
                issues.append(("缺值", col, pid, "空白或 ?、NA、unknown…"))
                continue
            try:
                x = float(str(v).strip())
            except ValueError:
                issues.append(("數字欄位混入文字", col, pid, f"「{v}」不是純數字"))
                continue
            lo, hi = RANGES[col]
            if col == "chol" and 0 < x < 30:
                issues.append(("單位疑似錯誤", col, pid, f"{x} 看起來是 mmol/L，要的是 mg/dL"))
            elif not (lo <= x <= hi):
                issues.append(("不可能的數值", col, pid, f"{x} 超出合理範圍 {lo}–{hi}"))
        for col, allowed in CAT_COLS.items():
            v = row[col]
            if _is_missing(v):
                issues.append(("缺值", col, pid, "空白或 NA…"))
            elif str(v) not in allowed:
                issues.append(("類別寫法不一致", col, pid, f"「{v}」不在 {allowed} 之中"))
    dup = d[d.duplicated("patient_id", keep="first")]
    for pid in dup["patient_id"]:
        issues.append(("重複列", "(整列)", pid, "同一位病人出現不只一次"))

    if not issues:
        print(f"✅ 全部通過！共 {len(d)} 列，沒有發現問題。")
        return True
    rep = pd.DataFrame(issues, columns=["類型", "欄位", "patient_id", "說明"])
    print(f"❌ 還有 {len(rep)} 個問題：")
    for kind, g in rep.groupby("類型"):
        print(f"  • {kind}：{len(g)} 個", end="")
        if hint >= 2:
            print(f"   欄位：{', '.join(sorted(g['欄位'].unique()))}", end="")
        print()
        if hint >= 3:
            for _, r in g.iterrows():
                print(f"      - {r['patient_id']} 的 {r['欄位']}：{r['說明']}")
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
