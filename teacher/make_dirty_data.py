"""
產生課堂用的「髒資料」。只需執行一次，輸出會放在 data/ 並 commit 到 GitHub。
來源：UCI Heart Disease (Cleveland)，ISLR 整理版，CC BY 4.0
    Detrano et al. (1989) Am J Cardiol 64:304-310
用法：python make_dirty_data.py
"""
import numpy as np, pandas as pd, os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "data")
SEED = 2026

# ---------- 1. 乾淨基準資料 ----------
raw = pd.read_csv(os.path.join(HERE, "heart_islr_source.csv"), index_col=0)
clean = pd.DataFrame({
    "patient_id": [f"P{i:03d}" for i in raw.index],
    "age": raw.Age,
    "sex": raw.Sex.map({1: "male", 0: "female"}),
    "chest_pain": raw.ChestPain,
    "resting_bp": raw.RestBP,
    "chol": raw.Chol,
    "fbs": raw.Fbs,
    "rest_ecg": raw.RestECG,
    "max_hr": raw.MaxHR,
    "exang": raw.ExAng,
    "oldpeak": raw.Oldpeak,
    "slope": raw.Slope,
    "ca": raw.Ca.astype("Int64"),
    "thal": raw.Thal.replace({"reversable": "reversible"}),
    "disease": raw.AHD.str.lower(),
}).reset_index(drop=True)
clean.to_csv(os.path.join(HERE, "heart_clean_original.csv"), index=False)

# ---------- 2. 錯誤注入 ----------
VARIANTS = {
    "sex": {"male": ["M", "Male", "m", " male", "MALE"],
            "female": ["F", "Female", "f", "female ", "FEMALE"]},
    "chest_pain": {"typical": ["Typical", "typical angina", "TYPICAL"],
                   "nontypical": ["non-typical", "Nontypical", "atypical"],
                   "nonanginal": ["non-anginal", "Nonanginal", "non anginal"],
                   "asymptomatic": ["Asymptomatic", "ASYMPTOMATIC", "asymptomatic "]},
    "thal": {"normal": ["Normal", "NORMAL"],
             "fixed": ["Fixed", "fixed defect"],
             "reversible": ["reversable", "Reversible", "reversible defect"]},
    "disease": {"yes": ["Yes", "YES", "Y", "1", "y"],
                "no": ["No", "NO", "N", "0", "n"]},
}
TEXT_FMT = {"resting_bp": "{} mmHg", "chol": "{} mg/dL", "max_hr": "{}bpm", "age": "{}y"}
MISSING_TOKENS = ["", "?", "NA", "unknown", "N/A", "null"]


def to_str(v):
    if pd.isna(v):
        return ""
    if isinstance(v, (float, np.floating)) and float(v).is_integer():
        return str(int(v))
    return str(v)


def inject(sub, plan, rng, protected_ids=()):
    """sub: 乾淨資料子集。plan: 要注入的錯誤。回傳 (髒資料 DataFrame(全字串), 錯誤紀錄)
    protected_ids: 這些病人的 chol / oldpeak 不注入錯誤（它們是「看起來怪但是真的」的極端值）"""
    truth = sub.reset_index(drop=True).map(to_str)
    d = truth.copy().astype(object)
    n = len(d)
    used = set()
    log = []

    def pick(col, k=1):
        cands = [i for i in range(n) if (i, col) not in used and truth.at[i, col] != ""
                 and not (col in ("chol", "oldpeak") and truth.at[i, "patient_id"] in protected_ids)]
        k = min(k, len(cands))
        return [int(x) for x in rng.choice(cands, size=k, replace=False)] if k else []

    def put(i, col, val, kind):
        used.add((i, col))
        log.append(dict(patient_id=d.at[i, "patient_id"], column=col, error_type=kind,
                        clean_value=truth.at[i, col], dirty_value=val))
        d.at[i, col] = val

    for col, tok in plan.get("missing", []):
        if col is None:
            col = str(rng.choice(["resting_bp", "chol", "max_hr", "oldpeak", "ca", "thal", "age"]))
        if tok is None:
            tok = str(rng.choice(MISSING_TOKENS))
        for i in pick(col):
            put(i, col, tok, "missing")
    for col in plan.get("impossible", []):
        if col is None:
            col = str(rng.choice(["age", "resting_bp", "chol", "max_hr"]))
        for i in pick(col):
            v = float(truth.at[i, col])
            bad = {"age": str(rng.choice(["250", "520", "-34"])), "resting_bp": "0", "chol": "0",
                   "max_hr": str(int(v) * 10)}[col]
            put(i, col, bad, "impossible")
    for i in pick("chol", plan.get("unit", 0)):
        put(i, "chol", f"{float(truth.at[i, 'chol']) / 38.67:.2f}", "unit")
    for col in plan.get("text", []):
        if col is None:
            col = str(rng.choice(list(TEXT_FMT)))
        for i in pick(col):
            put(i, col, TEXT_FMT[col].format(truth.at[i, col]), "text_in_number")
    for col, k in plan.get("category", {}).items():
        for i in pick(col, k):
            put(i, col, str(rng.choice(VARIANTS[col][truth.at[i, col]])), "category_spelling")
    for i in rng.choice(n, size=plan.get("duplicates", 0), replace=False):
        row = d.iloc[[int(i)]].copy()
        pos = int(rng.integers(0, len(d) + 1))
        d = pd.concat([d.iloc[:pos], row, d.iloc[pos:]], ignore_index=True)
        log.append(dict(patient_id=row.iloc[0]["patient_id"], column="(整列)",
                        error_type="duplicate", clean_value="", dirty_value="duplicate row"))
    return d, pd.DataFrame(log)


# ---------- 3. 小型練習集（40 列）----------
rng = np.random.default_rng(SEED)
real_extreme = [clean.chol.idxmax(), clean.oldpeak.idxmax()]       # 真實極端值：不是錯誤！
native_missing = list(clean[clean.ca.isna() | clean.thal.isna()].index[:2])
must = list(dict.fromkeys(real_extreme + native_missing))
protected = clean.loc[real_extreme, "patient_id"].tolist()
pool = [i for i in clean.index if i not in must]
pos = [i for i in pool if clean.at[i, "disease"] == "yes"]
neg = [i for i in pool if clean.at[i, "disease"] == "no"]
rest = 40 - len(must)
chosen = must + [int(x) for x in rng.choice(pos, rest - rest // 2, replace=False)] + \
         [int(x) for x in rng.choice(neg, rest // 2, replace=False)]
small_clean = clean.loc[sorted(chosen)]
plan_small = dict(
    missing=[("chol", ""), ("ca", "?"), ("thal", "NA"), ("resting_bp", "unknown"), ("oldpeak", "N/A")],
    impossible=["age", "resting_bp", "chol", "max_hr"],
    unit=3,
    text=["resting_bp", "resting_bp", "chol", "max_hr"],
    category=dict(sex=3, chest_pain=3, thal=2, disease=3),
    duplicates=2,
)
small_dirty, small_log = inject(small_clean, plan_small, rng, protected)

# ---------- 4. 完整資料集（303 + 重複列）----------
rng2 = np.random.default_rng(SEED + 1)
plan_full = dict(
    missing=[(None, None)] * 8,
    impossible=[None] * 6,
    unit=10,
    text=[None] * 10,
    category=dict(sex=40, chest_pain=45, thal=25, disease=45),
    duplicates=8,
)
full_dirty, full_log = inject(clean, plan_full, rng2, protected)

small_dirty.to_csv(os.path.join(OUT, "heart_dirty_small.csv"), index=False)
full_dirty.to_csv(os.path.join(OUT, "heart_dirty_full.csv"), index=False)
small_log.to_csv(os.path.join(HERE, "answer_key_small.csv"), index=False)
full_log.to_csv(os.path.join(HERE, "answer_key_full.csv"), index=False)

print("small:", small_dirty.shape, small_log.error_type.value_counts().to_dict())
print("full :", full_dirty.shape, full_log.error_type.value_counts().to_dict())
print("真實極端值（不是錯誤）:", clean.loc[real_extreme, ["patient_id", "chol", "oldpeak"]].values.tolist())
print("原本就缺值的病人:", clean.loc[native_missing, ["patient_id", "ca", "thal"]].values.tolist())
