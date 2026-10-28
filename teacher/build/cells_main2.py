# -*- coding: utf-8 -*-
# 接續 cells_main.py（同一個命名空間執行）

# ============================== 單元 2 ==============================
md(r'''
---
## 單元 2｜資料怎麼分配

模型的目的，是在**沒看過的新病人**身上也準。所以我們不能用「它學過的資料」來評分。
把資料分成三份，就像準備考試：

| 資料 | 像是 | 用途 | 可以用幾次 |
|---|---|---|---|
| **訓練集 train** | 課本與考古題 | 模型從這裡學習 | 無限次 |
| **驗證集 validation** | 模擬考 | 比較不同做法、調整設定 | 很多次（但用越多，越會偏向它） |
| **測試集 test** | **正式考試** | 最後給模型一個誠實的成績 | **只有一次** |
''')
code(r'''
#@title 2-1　隨機切成「訓練 80％／測試 20％」，換 5 個隨機種子看看
if "df_clean" not in globals():                 # 保險：沒跑單元 1 也能繼續
    df_clean = auto_clean(df_dirty_full)
X_all, y_all = to_xy(df_clean)

accs = []
for seed in range(5):
    Xtr, Xte, ytr, yte = train_test_split(X_all, y_all, test_size=0.2, random_state=seed)
    model, _ = train_mlp(Xtr, ytr)
    acc = evaluate(model, Xte, yte)["acc"]
    accs.append(acc)
    print(f"random_state={seed}：測試集準確率 {acc:.1%}　（測試集只有 {len(yte)} 位病人）")

plt.figure(figsize=(5, 2.8)); plt.plot(range(5), accs, "o-"); plt.ylim(0.4, 1)
plt.xlabel("random_state (how the data was split)"); plt.ylabel("test accuracy"); plt.tight_layout(); plt.show()
print(f"\n👉 同一個模型、同一份資料，只是『切法』不同，準確率就從 {min(accs):.0%} 跳到 {max(accs):.0%}。")
print("   測試集只有 61 個人，多抽到幾個難判斷的病人，成績就差很多。要小心『一次切分的結果』。")
''')
code(r'''
#@title 2-2　切成 訓練／驗證／測試 三份，並把「測試集」鎖起來
TEST_SIZE, VAL_SIZE, SEED = 0.20, 0.20, 42        # 👉 可以改改看

is_yes = df_clean["disease"] == "yes"
trainval_df, test_df = train_test_split(df_clean, test_size=TEST_SIZE, random_state=SEED, stratify=is_yes)
train_df, val_df = train_test_split(trainval_df, test_size=VAL_SIZE / (1 - TEST_SIZE), random_state=SEED,
                                    stratify=trainval_df["disease"] == "yes")      # stratify：讓每份的『有病比例』差不多

summary = pd.DataFrame({
    "人數": [len(train_df), len(val_df), len(test_df)],
    "有病比例": [(d["disease"] == "yes").mean() for d in (train_df, val_df, test_df)],
}, index=["訓練集 train", "驗證集 validation", "測試集 test"])
display(summary.style.format({"有病比例": "{:.1%}"}))

test_df.to_csv(os.path.join(WORKDIR, "heart_test_locked.csv"), index=False)
del test_df
print("🔒 測試集已存到 Drive，並從記憶體刪除。單元 4 才會打開它，而且只打開一次。")

Xtr, ytr = to_xy(train_df); Xva, yva = to_xy(val_df)
model, _ = train_mlp(Xtr, ytr)
print(f"\n用訓練集學習，在驗證集上的準確率：{evaluate(model, Xva, yva)['acc']:.1%}")
''')
code(r'''
#@title 2-3　K 折交叉驗證：資料太少時，讓每一位病人都輪流當一次「驗證」
K = 5
Xtv, ytv = to_xy(trainval_df)
member = np.zeros((K, len(trainval_df)))                 # 0＝這個人這一折在訓練，1＝在驗證
fold_acc = []
for k, (tr, va) in enumerate(StratifiedKFold(n_splits=K, shuffle=True, random_state=SEED).split(Xtv, ytv)):
    member[k, va] = 1
    model, _ = train_mlp(Xtv[tr], ytv[tr], seed=k)
    fold_acc.append(evaluate(model, Xtv[va], ytv[va])["acc"])
    print(f"第 {k+1} 折：訓練 {len(tr)} 人、驗證 {len(va)} 人 → 驗證準確率 {fold_acc[-1]:.1%}")
print(f"\n平均 {np.mean(fold_acc):.1%} ± {np.std(fold_acc):.1%}（± 是 5 折之間的標準差）")

from matplotlib.colors import ListedColormap
order = np.argsort(member.argmax(axis=0))
plt.figure(figsize=(8, 2.2))
plt.imshow(member[:, order], aspect="auto", cmap=ListedColormap(["#9ecae1", "#de2d26"]), interpolation="nearest")
plt.yticks(range(K), [f"fold {i+1}" for i in range(K)]); plt.xlabel("patients (blue = train, red = validation)")
plt.tight_layout(); plt.show()
print("👉 報告成績時，請報『平均 ± 標準差』，不要只報某一次切分的最好結果。")
''')
md(r'''
### ⚠️ 最容易犯、也最難發現的錯：資料洩漏（data leakage）

想像每位病人做了**好幾次檢查**（或一段 EEG 被切成好幾個片段）。如果你把這些「列」隨機打散去切分，**同一位病人就會同時出現在訓練集和測試集**。
模型不需要真的學會判斷疾病，只要「認得這個人」就能答對。
''')
code(r'''
#@title 2-4　洩漏示範：同一位病人同時出現在訓練與測試
rng = np.random.default_rng(0)

def make_multi_visit(d, visits=5, jitter=0.03):
    """假設每位病人檢查了 5 次，每次的數值有一點點測量誤差。"""
    out = []
    for _ in range(visits):
        e = d.copy()
        for c in ["age", "resting_bp", "chol", "max_hr", "oldpeak"]:
            e[c] = e[c] * rng.normal(1, jitter, len(e))
        out.append(e)
    return pd.concat(out, ignore_index=True)

multi = make_multi_visit(trainval_df)
Xm, ym = to_xy(multi); groups = multi["patient_id"].values
print(f"{len(trainval_df)} 位病人 × 5 次檢查 = {len(multi)} 列")

def fit_eval(tr, te, seed):
    sc = StandardScaler().fit(Xm[tr])        # 標準化：單元 3 會細講，這裡先用，讓模型學得更好
    model, _ = train_mlp(sc.transform(Xm[tr]), ym[tr], lr=0.01, seed=seed, hidden=(64, 32))
    # 這裡刻意用「大」的網路：越大的模型越會『背答案』，洩漏的傷害就越大
    return evaluate(model, sc.transform(Xm[te]), ym[te])["acc"]

acc_row, acc_pat = [], []
for seed in range(4):
    tr, te = train_test_split(np.arange(len(multi)), test_size=0.2, random_state=seed)              # ❌ 以「列」切分
    acc_row.append(fit_eval(tr, te, seed))
    tr, te = next(GroupShuffleSplit(1, test_size=0.2, random_state=seed).split(Xm, ym, groups))     # ✅ 以「病人」切分
    acc_pat.append(fit_eval(tr, te, seed))

print(f"\n❌ 以『列』隨機切分（同一位病人跨訓練/測試）：測試準確率 {np.mean(acc_row):.1%}")
print(f"✅ 以『病人』切分（每位病人只在一邊）　　　：測試準確率 {np.mean(acc_pat):.1%}")
plt.figure(figsize=(4, 2.8)); plt.bar(["split by row\n(leaky)", "split by patient\n(correct)"], [np.mean(acc_row), np.mean(acc_pat)],
                                      color=["#de2d26", "#3182bd"]); plt.ylim(0.5, 1); plt.ylabel("test accuracy"); plt.tight_layout(); plt.show()
print("\n👉 好到不真實的成績，要先懷疑『有沒有洩漏』，再開心。")
''')
md(r'''
**🎓 老師備註（單元 2）**
- 2-1 的準確率會明顯跳動（小網路、只做粗略縮放，準確率大約 70–90%）：這是刻意的，單元 3 會示範更好的縮放。
- 2-4 的差距大約是 99％ vs 78％。可以問：「如果這是一篇論文的結果，你會怎麼發現？」→ 看切分單位是病人還是影像／切片。
- 也可提：EEG 的 epoch、MRI 的切片、同一病人的多次住院，都是同樣的結構。
''', teacher=True)

# ============================== 單元 3 ==============================
md(r'''
---
## 單元 3｜前處理如何影響訓練

接下來每個實驗都用**同一批資料、同一個小網路**，只改「前處理」這一件事。
為了不被運氣影響，每個實驗都用 5 折交叉驗證重複 3 次（共 15 次訓練），報告**平均 ± 標準差**。
（只用訓練＋驗證資料；測試集還鎖著。）

我們用的網路超小：**輸入 → 3 個感知器 → 1 個感知器，總共 4 個**（約 70 個參數）。資料只有 300 筆，網路大了只會背答案。
''')
code(r'''
#@title 3-1　標準化：把每個欄位拉到同一把尺上
cols = ["age", "resting_bp", "chol", "max_hr", "oldpeak", "ca"]
print("縮放之前，各欄位的『大小』差很多：")
display(trainval_df[cols].describe().loc[["mean", "std", "min", "max"]].round(1))
print("膽固醇動輒 200–300，oldpeak 只有 0–6。數值大的欄位會『壓過』數值小的欄位。")
print("（單元 1、2 的 to_xy 其實偷偷做了『粗略版』：每欄除以一個固定的數字，縮到大約 0–1。現在來比較三種做法。）\n")

def loss_curves(scale):
    prep = make_prep("onehot", scale)
    Xtr = prep.fit_transform(add_male(train_df)).astype("float32")       # 只用「訓練集」算出縮放需要的數字
    Xva = prep.transform(add_male(val_df)).astype("float32")             # 驗證集只能「套用」，不能參與計算
    ytr = (train_df["disease"] == "yes").values.astype("float32"); yva = (val_df["disease"] == "yes").values.astype("float32")
    return train_mlp(Xtr, ytr, X_val=Xva, y_val=yva)[1]

curves = {"no scaling": loss_curves("none"), "rough (divide by fixed number)": loss_curves("rough"), "standardized": loss_curves("standard")}
fig, ax = plt.subplots(1, 2, figsize=(10, 3), sharey=True)
for a_, key, title in zip(ax, ["train", "val"], ["training loss", "validation loss"]):
    for name, h in curves.items():
        a_.plot(h[key], label=name)
    a_.set_title(title); a_.set_xlabel("epoch"); a_.legend(fontsize=8)
plt.tight_layout(); plt.show()

RESULTS.clear()
print("五折交叉驗證 × 3 次，三種做法的成績：")
_ = cv_eval(trainval_df, "S0. no scaling", encode="onehot", scale="none")
_ = cv_eval(trainval_df, "S1. rough scaling", encode="onehot", scale="rough")
res_std = cv_eval(trainval_df, "S2. standardize", encode="onehot", scale="standard")
print("\n👉 不縮放：成績明顯較低，而且很不穩定（± 很大）；縮放之後又好又穩。標準化＝減平均、除以標準差，讓每欄平均 0、標準差 1。")
''')
md(r'''
**標準化的重點**：`(數值 − 訓練集的平均) ÷ 訓練集的標準差`。注意這裡的「平均」和「標準差」是**從訓練集算出來**，之後套用到驗證集、測試集、新病人身上。
''')
md(r'''
**🎓 老師備註**：預期 S0 約 70％（± 約 14，很不穩定）、S1 約 82％、S2 約 83％。看 loss 曲線：S0 下降慢又抖。S1 vs S2 的差距不大，可以說明「粗略縮放也能用，標準化是更標準、更穩的版本」。
''', teacher=True)
code(r'''
#@title 3-2　類別資料：整數編碼 vs One-hot（之後的實驗都使用標準化）
print("原始表格：類別是文字")
display(trainval_df[["patient_id", "chest_pain", "thal"]].head(4))

print("\nOne-hot：每個類別變成一欄 0/1")
oh = pd.get_dummies(trainval_df[["chest_pain"]]).astype(int)
display(pd.concat([trainval_df[["patient_id", "chest_pain"]], oh], axis=1).head(4))

print("\n整數編碼會是 typical=0, nontypical=1, nonanginal=2, asymptomatic=3。")
print("模型會以為 asymptomatic(3) 是 typical(0) 的『三倍』、nonanginal 在『中間』，但這個順序是我們隨便排的。")
print("\n各類型實際的有病比例：")
display((trainval_df.groupby("chest_pain")["disease"].apply(lambda s: (s == "yes").mean())
         .round(2).to_frame("有病比例")))

print("\n三種做法的比較：")
_ = cv_eval(trainval_df, "A. numbers only (no text cols)", encode="none")
_ = cv_eval(trainval_df, "B. integer codes", encode="int")
_ = cv_eval(trainval_df, "C. one-hot", encode="onehot")
''')
md(r'''
**怎麼讀這個結果？** 加進胸痛類型與 thal 之後，成績有沒有變好（A → B/C）？One-hot 比整數編碼好多少？看看差距有沒有大過 ± 的範圍。
資料很少時，**小的差異可能只是運氣**，這就是為什麼我們要看平均 ± 標準差。
''')
md(r'''
**🎓 老師備註**：A → B/C 有進步；B vs C 常在雜訊範圍內。可以誠實地說：「在這個小網路與小資料上，one-hot 的好處不大；它真正重要的場景是類別很多（例如 ICD 碼）或線性模型。但**整數編碼會引入不存在的順序，是概念上的錯誤**。」
''', teacher=True)
code(r'''
#@title 3-3　⚠️ 洩漏：「先標準化全部資料、再切分」 vs 「只用訓練集的平均與標準差」
d = add_male(trainval_df).reset_index(drop=True)
y = (d["disease"] == "yes").astype(float).values
X_leaky = make_prep("onehot", True).fit_transform(d).astype("float32")       # ❌ 平均/標準差用了全部資料，包含之後要拿來『考試』的人
rows = []
for k, (tr, va) in enumerate(RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=0).split(d, y)):
    model, _ = train_mlp(X_leaky[tr], y[tr], seed=k)
    rows.append(evaluate(model, X_leaky[va], y[va]))
r = pd.DataFrame(rows)
proper = res_std
print(f"❌ 先標準化再切分：accuracy {r['acc'].mean():.3f}   AUC {r['auc'].mean():.3f}")
print(f"✅ 只用訓練集　　：accuracy {proper['acc']:.3f}   AUC {proper['auc']:.3f}")
print("\n這個資料集上兩者幾乎一樣：因為只是一點點平均值與標準差的資訊『滲』過去。")
print("但原則不能破：任何『從資料算出來的東西』（平均、標準差、補缺值用的中位數、挑特徵）都只能用訓練資料算。")
''')
code(r'''
#@title 3-4　（加碼，時間夠再跑）洩漏有多可怕？用「完全隨機」的資料看看
rng = np.random.default_rng(0)
n = len(trainval_df)
X_noise = rng.normal(size=(n, 300)).astype("float32")      # 300 個純雜訊的「基因」
y_noise = rng.integers(0, 2, n).astype("float32")          # 隨機決定有病或沒病：根本不可能預測

def top_k(X, y, k=10):
    Xc = (X - X.mean(0)) / X.std(0); yc = (y - y.mean()) / y.std()
    return np.argsort(-np.abs((Xc * yc[:, None]).mean(0)))[:k]

leaky, proper_ = [], []
for k, (tr, va) in enumerate(RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=0).split(X_noise, y_noise)):
    sel = top_k(X_noise, y_noise)                      # ❌ 用「全部」資料的答案來挑特徵
    m, _ = train_mlp(X_noise[tr][:, sel], y_noise[tr], seed=k, lr=0.01); leaky.append(evaluate(m, X_noise[va][:, sel], y_noise[va])["acc"])
    sel = top_k(X_noise[tr], y_noise[tr])              # ✅ 只用訓練折挑特徵
    m, _ = train_mlp(X_noise[tr][:, sel], y_noise[tr], seed=k, lr=0.01); proper_.append(evaluate(m, X_noise[va][:, sel], y_noise[va])["acc"])
print(f"❌ 先用全部資料挑特徵：準確率 {np.mean(leaky):.1%}")
print(f"✅ 只用訓練資料挑特徵：準確率 {np.mean(proper_):.1%}　（隨機資料，理論上就是 50％）")
print("\n👉 資料裡完全沒有訊號，洩漏卻讓模型看起來『有預測力』。基因體學、影像組學（特徵多、人數少）的論文最容易中招。")
''')
code(r'''
#@title 3-5　資料擴增：表格資料加雜訊
_ = cv_eval(trainval_df, "E. + noise augmentation (x3)", encode="onehot", scale="standard", augment=noise_augment(3, 0.1))
print("\n👉 在這個小資料集上，擴增幾乎沒有幫助（差異在 ± 範圍內）。擴增不是萬靈丹，一樣要用驗證的方式檢查有沒有用。")
print("   而且：只能擴增『訓練集』。如果驗證集也擴增，同一個病人的變形版本就會跨過兩邊，又是洩漏。")
''')
code(r'''
#@title 3-6　影像資料的擴增（純示範，不訓練）：哪些變形在醫學上合理？
try:
    from skimage.data import shepp_logan_phantom
    from skimage.transform import resize, rotate
    head = resize(shepp_logan_phantom(), (200, 200))               # 標準的「頭部切面」測試影像（模擬，非真實病人）
except Exception:
    yy0, xx0 = np.mgrid[:200, :200]; head = (((xx0 - 100) / 80) ** 2 + ((yy0 - 100) / 95) ** 2 < 1) * 0.6
    rotate = lambda im, a: im
yy, xx = np.mgrid[:200, :200]
img = head.copy(); img[((xx - 60) ** 2 + (yy - 95) ** 2) < 14 ** 2] = 1.0       # 在影像「左側」放一個假的病灶
panels = [("original", img), ("flip left-right", img[:, ::-1]), ("rotate 10 deg", rotate(img, 10)),
          ("brightness x1.3", np.clip(img * 1.3, 0, 1)),
          ("gaussian noise", np.clip(img + np.random.default_rng(0).normal(0, 0.08, img.shape), 0, 1))]
fig, ax = plt.subplots(1, 5, figsize=(13, 3))
for a, (t, im) in zip(ax, panels):
    a.imshow(im, cmap="gray", vmin=0, vmax=1); a.set_title(t); a.axis("off")
plt.tight_layout(); plt.show()
print("左右翻轉之後，病灶跑到另一側。如果你的任務跟『哪一側』有關（例如左/右側中風），這種擴增會直接把標籤弄錯。")
print("旋轉一點點、調亮度、加雜訊，通常合理（模擬病人擺位、不同掃描儀）。什麼擴增合理，要由懂臨床的人決定。")
''')
code(r'''
#@title 3-7　資料不平衡：罕見疾病只占一小部分
pos = trainval_df[trainval_df["disease"] == "yes"]; neg = trainval_df[trainval_df["disease"] == "no"]
n_pos = int(round(0.15 * len(neg) / 0.85))
imb_df = pd.concat([neg, pos.sample(n_pos, random_state=1)]).sample(frac=1, random_state=1).reset_index(drop=True)
yi = (imb_df["disease"] == "yes").astype(float).values
print(f"把有病的人抽樣到約 15％：共 {len(imb_df)} 人，其中有病 {int(yi.sum())} 人（{yi.mean():.0%}）")

def oof_proba(strategy, repeats=3):
    """交叉驗證，把每一位病人『沒被訓練到的那一折』的預測機率收集起來。"""
    P = np.zeros(len(imb_df))
    for rep in range(repeats):
        for j, (tr, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=rep).split(imb_df, yi)):
            prep = make_prep("onehot", True)
            Xtr = prep.fit_transform(add_male(imb_df.iloc[tr])).astype("float32"); Xva = prep.transform(add_male(imb_df.iloc[va])).astype("float32")
            ytr, pw = yi[tr], None
            if strategy == "class weight":                       # 讓「錯過一個病人」的懲罰變重
                pw = (ytr == 0).sum() / (ytr == 1).sum()
            if strategy == "oversample":                         # 把少數類別（有病）重複抽樣補到跟多數一樣多
                idx = np.where(ytr == 1)[0]
                extra = np.random.default_rng(j).choice(idx, (ytr == 0).sum() - len(idx), replace=True)
                Xtr, ytr = np.vstack([Xtr, Xtr[extra]]), np.r_[ytr, ytr[extra]]
            model, _ = train_mlp(Xtr, ytr, seed=j, pos_weight=pw)
            P[va] += predict_proba(model, Xva) / repeats
    return P

rows = [{"strategy": "always predict 'no disease'", "accuracy": 1 - yi.mean(), "recall": 0.0, "precision": float("nan"), "AUC": 0.5}]
for s in ["plain", "class weight", "oversample"]:
    P = oof_proba(s); pred = (P >= 0.5).astype(int)
    rows.append({"strategy": s, "accuracy": accuracy_score(yi, pred), "recall": recall_score(yi, pred),
                 "precision": precision_score(yi, pred, zero_division=0), "AUC": roc_auc_score(yi, P)})
display(pd.DataFrame(rows).set_index("strategy").round(3))
print("👉 『全部猜沒病』的準確率就有 85％，recall（有病的人抓到幾成）卻是 0。罕見疾病不能只看 accuracy。")
print("   class weight、過採樣可以多抓到一些病人（recall 上升），代價通常是 precision 下降（誤報變多）。要選哪個，取決於漏診與誤報哪個代價更大。")
''')
code(r'''
#@title 3-8　總結：每一種前處理，到底值多少？
res = pd.DataFrame(RESULTS)
display(res.round(3))
plt.figure(figsize=(7, 3)); plt.barh(res["label"][::-1], res["acc"][::-1], xerr=res["acc_sd"][::-1], color="#3182bd")
plt.xlim(0.5, 1); plt.xlabel("cross-validated accuracy (mean ± sd)"); plt.tight_layout(); plt.show()
print("標準化的影響最大；加入類別欄位、one-hot 次之；擴增在這裡沒有顯著差異。")
print("前處理沒有『一定有用』的招式，每一招都要用乾淨的驗證方式檢查。")
''')

# ============================== 單元 4 ==============================
md(r'''
---
## 單元 4｜推論：新病人進來了

**訓練（training）** 是在學習；**推論（inference）** 是真的拿來用。
關鍵原則：**推論時對資料做的前處理，必須跟訓練時一模一樣。**
這表示：所有「從訓練資料算出來的東西」（標準化的平均與標準差、one-hot 的欄位順序、補缺值用的中位數）都要**一起存起來**，跟模型成套使用。
''')
code(r'''
#@title 4-1　定案：用訓練＋驗證資料建立前處理、訓練最終模型，一起存起來
import joblib
fill = {**{c: trainval_df[c].median() for c in ALL_NUMERIC},                  # 補缺值用的數字：只用訓練＋驗證資料算
        **{c: trainval_df[c].mode()[0] for c in ["sex", "chest_pain", "thal"]}}

final_prep = make_prep("onehot", True)
X_final = final_prep.fit_transform(add_male(trainval_df)).astype("float32")
y_final = (trainval_df["disease"] == "yes").astype("float32").values
final_model, _ = train_mlp(X_final, y_final, seed=0)

ART = os.path.join(WORKDIR, "artifacts"); os.makedirs(ART, exist_ok=True)
joblib.dump({"prep": final_prep, "fill": fill}, os.path.join(ART, "preprocessor.joblib"))
torch.save(final_model.state_dict(), os.path.join(ART, "model.pt"))
print("✅ 已存到 Drive：", sorted(os.listdir(ART)))
print(f"模型輸入維度：{X_final.shape[1]} 個特徵（原始 {len(FEATURES_V1)} 個數字欄位，經過 one-hot 展開後變多了）")
''')
code(r'''
#@title 4-2　🔓 終於打開測試集（只能這一次！）
test_locked = pd.read_csv(os.path.join(WORKDIR, "heart_test_locked.csv"))
test_locked[ALL_NUMERIC] = test_locked[ALL_NUMERIC].astype(float)
X_test = final_prep.transform(add_male(test_locked)).astype("float32")
y_test = (test_locked["disease"] == "yes").astype("float32").values
res = evaluate(final_model, X_test, y_test)
print(f"測試集（{len(y_test)} 位從沒被看過的病人）：accuracy {res['acc']:.1%}　recall {res['recall']:.1%}　precision {res['precision']:.1%}　AUC {res['auc']:.3f}")
print("\n👉 這才是對『新病人』的誠實成績。如果現在回頭調整模型再重測，測試集就變成另一份驗證集，失去意義。")
''')
code(r'''
#@title 4-3　一位新病人走進診間：原始表單 → 同樣的前處理 → 預測
new_patient = {            # 就像從不同系統匯出來的「原始表單」，有各種不整齊的寫法
    "patient_id": "NEW001", "age": "55", "sex": "Male", "chest_pain": "non-anginal",
    "resting_bp": "135 mmHg", "chol": "6.4", "fbs": "0", "rest_ecg": "0", "max_hr": "140",
    "exang": "0", "oldpeak": "1.2", "slope": "2", "ca": "1", "thal": "reversable",
}
print("原始輸入：", {k: v for k, v in new_patient.items() if k in ["sex", "chest_pain", "resting_bp", "chol", "thal"]}, "…")

def predict_risk(raw, prep=final_prep, model=final_model, fill=fill, tweak=None):
    d = auto_clean(pd.DataFrame([raw]), fill=fill)       # 同一套清理規則（含：mmol/L→mg/dL、寫法統一、補值用『訓練時』的數字）
    if tweak: tweak(d)
    return float(predict_proba(model, prep.transform(add_male(d)).astype("float32"))[0])

risk = predict_risk(new_patient)
cleaned = auto_clean(pd.DataFrame([new_patient]), fill=fill)
print(f"清理後：sex={cleaned.sex[0]}, resting_bp={cleaned.resting_bp[0]:.0f}, chol={cleaned.chol[0]:.0f} mg/dL, thal={cleaned.thal[0]}")
print(f"\n✅ 預測有冠心病的機率：{risk:.1%}")
''')
code(r'''
#@title 4-4　故意做錯：前處理跟訓練時不一致，會發生什麼事？
def no_unit_fix(d): d.loc[0, "chol"] = float(new_patient["chol"])   # 忘了把 mmol/L 換成 mg/dL（直接用 6.4）

# 錯誤 A：跳過標準化（直接把原始數字餵進去）
X_ok = final_prep.transform(add_male(cleaned)).astype("float32")
raw_vals = cleaned[PREP_NUM].values.astype("float32")
X_noscale = X_ok.copy(); X_noscale[:, :len(PREP_NUM)] = raw_vals
risk_noscale = float(predict_proba(final_model, X_noscale)[0])

# 錯誤 B：忘了單位換算
risk_unit = predict_risk(new_patient, tweak=no_unit_fix)

# 錯誤 C：在新資料上「重新」算平均與標準差（只有一個人，標準差是 0）
sc_wrong = StandardScaler().fit(cleaned[PREP_NUM])
X_resc = X_ok.copy(); X_resc[:, :len(PREP_NUM)] = sc_wrong.transform(cleaned[PREP_NUM])
risk_resc = float(predict_proba(final_model, X_resc)[0])

table = pd.DataFrame({"做法": ["✅ 正確：套用訓練時存下來的前處理", "❌ A. 跳過標準化", "❌ B. 忘了膽固醇單位換算", "❌ C. 在新資料上重新標準化"],
                      "預測機率": [risk, risk_noscale, risk_unit, risk_resc]})
display(table.style.format({"預測機率": "{:.1%}"}).hide(axis="index"))

# 錯誤 D：現場用 get_dummies 做 one-hot（新病人只有一位，欄位數、順序都跟訓練時不一樣）
try:
    X_bad = pd.get_dummies(cleaned[["chest_pain", "thal"]]).astype("float32").values
    print(f"\n❌ D. 現場 get_dummies：只得到 {X_bad.shape[1]} 欄；模型需要 {X_final.shape[1]} 欄。")
    predict_proba(final_model, X_bad)
except Exception as e:
    print("   模型直接報錯：", str(e)[:90], "…")
print("\n👉 最危險的是 A、B、C：程式不會報錯，只是預測默默變得不可靠；A 甚至會讓『高風險病人』看起來像『低風險』。")
''')
md(r'''
### 收尾：把今天帶走

**資料不只是數字，它還有來源與隱私**
- **去識別化**：病歷文字裡的姓名、日期、病歷號；DICOM 影像標頭裡的個資；腦部 MRI 重建後有可能認出人臉。研究前要經 IRB，並依個資法處理。
- **偏誤**：資料若只來自某一家醫院、某一族群，模型在別的地方可能就不準。跨院、跨儀器的外部驗證很重要。
- **標籤品質**：不同醫師對同一份 EEG 判讀不一致時，AI 學到的是「誰的標準」？

**閱讀 AI 論文時的檢查清單**
1. 資料從哪裡來？有多少醫院、多少病人？
2. 切分是以「病人」還是以「影像／切片」為單位？
3. 前處理（標準化、補值、挑特徵）是在切分**之前**還是**之後**做的？
4. 有沒有外部驗證（不同醫院、不同儀器）？
5. 標籤是誰標的？有幾位？一致性如何？
6. 評估指標適合這個疾病的盛行率嗎？（別只看 accuracy）

🎉 恭喜完成！你的 Google 雲端硬碟資料夾 `AI_preprocessing_class` 裡，有今天產生的所有檔案。
''')
