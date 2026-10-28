# ===== 前處理管線（單元三、四共用）=====
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, train_test_split, GroupShuffleSplit
import matplotlib.pyplot as plt

PREP_NUM = NUM_COLS                       # 要標準化的連續數值
PREP_BIN = ["male"] + BIN_COLS            # 本來就是 0/1
PREP_CAT = ["chest_pain", "thal", "rest_ecg", "slope"]   # 類別


def _rough_scale(X):
    return np.asarray(X, dtype="float64") / np.array([ROUGH[c] for c in PREP_NUM])


def make_prep(encode="onehot", scale="standard"):
    """encode: 'none'（類別欄位只用數字碼，且不用文字欄位）/ 'int'（文字類別轉成整數）/ 'onehot'
    scale: 'none' 不縮放 / 'rough' 除以固定數字 / 'standard'（或 True）標準化"""
    if scale is True:
        scale = "standard"
    if scale is False:
        scale = "none"
    num_tf = {"none": "passthrough", "rough": FunctionTransformer(_rough_scale), "standard": StandardScaler()}[scale]
    steps = [("num", num_tf, PREP_NUM),
             ("bin", "passthrough", PREP_BIN)]
    if encode == "none":
        steps.append(("code", "passthrough", CODE_COLS))
    elif encode == "int":
        steps.append(("code", "passthrough", CODE_COLS))
        steps.append(("ord", OrdinalEncoder(categories=[CATEGORIES["chest_pain"], CATEGORIES["thal"]]),
                      ["chest_pain", "thal"]))
    elif encode == "onehot":
        steps.append(("onehot", OneHotEncoder(categories=[CATEGORIES[c] for c in PREP_CAT], sparse_output=False),
                      PREP_CAT))
    return ColumnTransformer(steps)


RESULTS = []          # 單元三所有實驗的成績，最後一起比較


def cv_eval(df, label, encode="onehot", scale="standard", augment=None, epochs=300, lr=0.003, repeats=3, record=True):
    """5 折交叉驗證 × repeats 次。前處理「只用訓練折」來 fit，再套用到驗證折。"""
    d = add_male(df).reset_index(drop=True)
    y = (d["disease"] == "yes").astype(float).values
    rows = []
    for k, (tr, va) in enumerate(RepeatedStratifiedKFold(n_splits=5, n_repeats=repeats, random_state=0).split(d, y)):
        prep = make_prep(encode, scale)
        Xtr = prep.fit_transform(d.iloc[tr]).astype("float32")
        Xva = prep.transform(d.iloc[va]).astype("float32")
        ytr = y[tr]
        if augment is not None:
            Xtr, ytr = augment(Xtr, ytr, k)
        model, _ = train_mlp(Xtr, ytr, epochs=epochs, lr=lr, seed=k)
        rows.append(evaluate(model, Xva, y[va]))
    r = pd.DataFrame(rows)
    out = dict(label=label, acc=r["acc"].mean(), acc_sd=r["acc"].std(), auc=r["auc"].mean(), auc_sd=r["auc"].std())
    if record:
        RESULTS.append(out)
    print(f"{label:<30} accuracy {out['acc']:.3f} ± {out['acc_sd']:.3f}   AUC {out['auc']:.3f} ± {out['auc_sd']:.3f}")
    return out


def noise_augment(copies=3, sd=0.1):
    """表格資料的擴增：把訓練資料複製幾份，連續數值加上一點雜訊。只能用在訓練集！"""
    def f(X, y, seed):
        r = np.random.default_rng(seed)
        mask = np.r_[np.ones(len(PREP_NUM)), np.zeros(X.shape[1] - len(PREP_NUM))].astype("float32")
        Xs, ys = [X], [y]
        for _ in range(copies):
            Xs.append(X + r.normal(0, sd, X.shape).astype("float32") * mask)
            ys.append(y)
        return np.vstack(Xs), np.concatenate(ys)
    return f
