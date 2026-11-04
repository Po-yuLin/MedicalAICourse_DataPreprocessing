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
md(r'''
### 什麼是「隨機種子」（random seed）？

電腦的「隨機」其實是**算出來的**，就像**洗撲克牌**：「怎麼起手洗牌」就是**種子**。

| 種子 | 結果 |
|---|---|
| **同一個種子** | 每次洗出一模一樣的牌序 → 切出來的訓練／測試資料也一模一樣（別人可以**重現**你的結果） |
| **換一個種子** | 洗出另一種牌序 → 切出來的人不同，成績也跟著不同 |

程式裡，切分資料的種子叫 `random_state`，訓練模型的叫 `seed`。
''')
code(r'''
#@title 2-1　隨機切成「訓練 80％／測試 20％」，換 5 個隨機種子看看
if "df_clean" not in globals():                 # 保險：沒跑單元 1 也能繼續
    df_clean = auto_clean(df_dirty_full)
X_all, y_all = to_xy(df_clean)

TEST_SIZE = 0.2        # 👈 試試看：改成 0.5、0.9 再重新執行，成績會怎麼變？（test_size＝拿多少比例當測試集）
accs = []
for seed in range(5):
    Xtr, Xte, ytr, yte = train_test_split(X_all, y_all, test_size=TEST_SIZE, random_state=seed)
    model, _ = train_mlp(Xtr, ytr)
    acc = evaluate(model, Xte, yte)["acc"]
    accs.append(acc)
    print(f"random_state={seed}：測試集準確率 {acc:.1%}　（測試集只有 {len(yte)} 位病人）")

plt.figure(figsize=(7.5, 4.2)); plt.plot(range(5), accs, "o-"); plt.ylim(0.4, 1)
plt.xlabel("random_state (how the data was split)"); plt.ylabel("test accuracy"); plt.tight_layout(); plt.show()
print(f"\n👉 同一個模型、同一份資料，只是『切法』不同，準確率就從 {min(accs):.0%} 跳到 {max(accs):.0%}。")
print(f"   測試集只有 {len(yte)} 個人，多抽到幾個難判斷的病人，成績就差很多。要小心『一次切分的結果』。")
print("\n👉 請試試看：把上面的 TEST_SIZE 改成 0.5、0.9，再重新執行。測試集變大、訓練集變小，成績會有什麼不同？為什麼？")
''')
md(r'''
**🎓 提醒（2-1 互動）**
- 叫出右側的 **Gemini**，請它「逐行解釋這段程式」，帶學生看懂 `train_test_split`、`test_size`、`random_state`。
- 示範把 `TEST_SIZE` 改成 **0.9**：訓練只剩約 30 人，模型學得少，成績下降。您實測約 70%；我這邊（5 個 seed）約 74–79%，差異來自模型初始化等細節，**以您現場看到的為準**。
- 順便讓學生把 `range(5)` 的種子換掉，看「同一份資料、不同切法」的波動。
''', teacher=True)
md(r'''
### stratify（分層）是什麼？

**stratify**＝切分時，讓**每一份**資料裡「有病／沒病」的比例，都跟全體差不多。
資料少、或疾病罕見時，如果單純隨機切，**運氣不好**就可能出現「測試集裡幾乎沒有病人」。下面的 `stratify=` 就是在做這件事。
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
summary["有病比例"] = summary["有病比例"].map("{:.1%}".format)
show(summary, index=True)

test_df.to_csv(os.path.join(WORKDIR, "heart_test_locked.csv"), index=False)
del test_df
print("🔒 測試集已存到 Drive，並從記憶體刪除。單元 4 才會打開它，而且只打開一次。")

Xtr, ytr = to_xy(train_df); Xva, yva = to_xy(val_df)
model, _ = train_mlp(Xtr, ytr)
print(f"\n用訓練集學習，在驗證集上的準確率：{evaluate(model, Xva, yva)['acc']:.1%}")
''')
code(r'''
#@title 2-3　如果「不 stratify」會怎樣？（故意把切分弄得不平均）
# ① 隨機切 20 次，看測試集裡「有病的比例」會晃多大
rates = {}
for use_strat in (False, True):
    r = []
    for seed in range(20):
        _, te = train_test_split(df_clean, test_size=0.2, random_state=seed, stratify=(is_yes if use_strat else None))
        r.append((te["disease"] == "yes").mean())
    rates[use_strat] = r
show(pd.DataFrame({"切法": ["不 stratify", "stratify"],
                   "測試集有病比例：最低": [f"{min(rates[False]):.1%}", f"{min(rates[True]):.1%}"],
                   "最高": [f"{max(rates[False]):.1%}", f"{max(rates[True]):.1%}"]}))
print(f"全體的有病比例是 {is_yes.mean():.1%}。不 stratify：每次切出來的測試集，有病比例都不一樣。")

# ② 更極端：先依 disease 排序（沒病的在前、有病的在後），再「不洗牌」直接切前 80%／後 20%
srt = df_clean.sort_values("disease").reset_index(drop=True)
tr_bad, te_bad = train_test_split(srt, test_size=0.2, shuffle=False)
tr_ok, te_ok = train_test_split(srt, test_size=0.2, random_state=0, stratify=srt["disease"] == "yes")
rows = []
for name, tr_, te_ in [("❌ 排序後直接切（不洗牌、不 stratify）", tr_bad, te_bad), ("✅ 洗牌並 stratify", tr_ok, te_ok)]:
    Xa, ya = to_xy(tr_); Xb, yb = to_xy(te_)
    model, _ = train_mlp(Xa, ya)
    rows.append({"切法": name, "訓練集有病比例": f"{(tr_['disease'] == 'yes').mean():.0%}",
                 "測試集有病比例": f"{(te_['disease'] == 'yes').mean():.0%}", "測試準確率": f"{evaluate(model, Xb, yb)['acc']:.1%}"})
show(pd.DataFrame(rows))
print("👉 左邊的測試集『全部都是病人』，訓練集的病人卻只有三成：模型沒學到該學的分布，這個準確率也無法反映它在真實世界的表現。")
''')
code(r'''
#@title 2-4　K 折交叉驗證：資料太少時，讓每一位病人都輪流當一次「驗證」
K = 5
Xtv, ytv = to_xy(trainval_df)
folds = list(StratifiedKFold(n_splits=K, shuffle=True, random_state=SEED).split(Xtv, ytv))

# ① 先看「誰在哪一次當驗證」（還沒開始訓練）
member = np.zeros((K, len(trainval_df)))                 # 0＝這次拿去訓練，1＝這次拿去驗證
for k, (tr, va) in enumerate(folds):
    member[k, va] = 1
from matplotlib.colors import ListedColormap
order = np.argsort(member.argmax(axis=0))
plt.figure(figsize=(12, 3.8))
plt.imshow(member[:, order], aspect="auto", cmap=ListedColormap(["#9ecae1", "#de2d26"]), interpolation="nearest")
plt.yticks(range(K), [f"round {i+1}" for i in range(K)]); plt.xlabel(f"{len(trainval_df)} patients (blue = train, red = validation)")
plt.tight_layout(); plt.show()
print("每一橫列是一次訓練：紅色＝這次拿去『驗證』的病人，藍色＝這次拿去『訓練』的病人。")
print("每位病人剛好當一次驗證、四次訓練；5 次的驗證成績平均起來，比只切一次可靠。")

# ② 再依序訓練、評分
fold_acc = []
for k, (tr, va) in enumerate(folds):
    model, _ = train_mlp(Xtv[tr], ytv[tr], seed=k)
    fold_acc.append(evaluate(model, Xtv[va], ytv[va])["acc"])
    print(f"第 {k+1} 次：訓練 {len(tr)} 人、驗證 {len(va)} 人 → 驗證準確率 {fold_acc[-1]:.1%}")
print(f"\n平均 {np.mean(fold_acc):.1%} ± {np.std(fold_acc):.1%}（± 是 5 次之間的標準差）")
print("👉 報告成績時，請報『平均 ± 標準差』，不要只報某一次切分的最好結果。")
''')
md(r'''
### 其他「輪流當驗證」的方式

| 方法 | 做法 | 訓練幾次 | 適合 |
|---|---|---|---|
| **K 折（K-fold）** | 切成 K 份，輪流 1 份當驗證 | K 次 | 最常用，K 通常取 5 或 10 |
| **Leave-one-out（LOO，留一法）** | 每次只留 **1 個樣本**當驗證，其餘全部訓練 | N 次（N＝樣本數） | 資料**非常少**時；但很慢，而且每次驗證只有 1 筆，成績較不穩定 |
| **Leave-one-patient-out（LOPO，留一病人法）** | 每次留 **1 位病人**（的**所有**列）當驗證 | N 次（N＝病人數） | **一位病人有多筆資料**（多次檢查、多張切片）時，必須用這種「以病人分組」的切法，否則會洩漏（見下一節） |
''')
fig('splits', 'K 折、LOO、LOPO 的差別')
code(r'''
#@title 2-5　Leave-one-out 要訓練幾次？
from sklearn.model_selection import LeaveOneOut
loo = LeaveOneOut()
n_fits = loo.get_n_splits(Xtv)
t0 = time.time(); train_mlp(Xtv, ytv); one_fit = time.time() - t0
print(f"{len(Xtv)} 位病人 → Leave-one-out 要訓練 {n_fits} 次（K 折只要 5 次）。")
print(f"訓練一次約 {one_fit:.2f} 秒 → 全部約 {n_fits * one_fit / 60:.1f} 分鐘；而且每次的驗證集只有 1 個人。")
tr0, va0 = next(loo.split(Xtv))
print(f"第 1 次：訓練 {len(tr0)} 人、驗證 {len(va0)} 人（就是 trainval_df 的第 {va0[0] + 1} 位病人）。")

RUN_LOO = False        # 👈 想真的跑一遍（約 1 分鐘）就改成 True
if RUN_LOO:
    hits = []
    for k, (tr, va) in enumerate(loo.split(Xtv)):
        model, _ = train_mlp(Xtv[tr], ytv[tr], seed=k)
        hits.append(float(((predict_proba(model, Xtv[va]) >= 0.5) == ytv[va]).all()))
    print(f"Leave-one-out 準確率：{np.mean(hits):.1%}")
''')
md(r'''
### ⚠️ 資料洩漏（data leakage）：模型「偷看了答案」

模型在訓練時，**不小心拿到了「測試時不該知道的資訊」**，成績就會好到不真實。最常見的三種：
''')
fig('leak_patient', '洩漏①：病人層級')
fig('leak_preproc', '洩漏②：前處理')
fig('leak_feature', '洩漏③：特徵')
md(r'''
下面先動手做第 ① 種（病人層級）；第 ② 種在單元 3 的標準化之後做；第 ③ 種請記得：**特徵一定要是「預測當下」就拿得到的資訊**。
''')
md(r'''
### ① 病人層級洩漏：動手做做看

想像每位病人做了**好幾次檢查**（或一段 EEG 被切成好幾個片段、一份 MRI 被切成好幾張切片）。資料會長這樣（數值僅為示意）：

| patient_id | 第幾次檢查 | resting_bp | chol | max_hr | disease |
|---|---|---|---|---|---|
| P001 | 1 | 145 | 233 | 150 | no |
| P001 | 2 | 148 | 229 | 153 | no |
| P001 | 3 | 143 | 236 | 148 | no |
| P002 | 1 | 160 | 286 | 108 | yes |
| P002 | 2 | 157 | 291 | 110 | yes |
| … | … | … | … | … | … |

同一位病人的好幾列，數值**非常接近**，答案（`disease`）也**一模一樣**。

- **如果把這些「列」全部打散再隨機切分**：P001 的第 1、3 次檢查可能進了訓練集，第 2 次進了測試集。模型不需要真的學會判斷疾病，只要「認得 P001 長這樣」就能答對測試題。
- **正確做法**：先決定「哪些**病人**」當訓練、哪些當測試，再把每位病人的**所有檢查**一起帶走。

### 這個實驗怎麼做
1. 把我們的 242 位病人，每人複製成 5 列，每列的數值加上約 3% 的隨機誤差（模擬 5 次檢查）→ 共 1210 列。
2. 用兩種切法，各把資料分成 80% 訓練、20% 測試：
   - ❌ **A：以「列」切分**——把 1210 列打散後隨機切
   - ✅ **B：以「病人」切分**——先切病人，再帶走他們全部的列
3. 兩種切法用**同一個模型、同樣的訓練方式**（刻意用比較大的網路：越大的模型越會「背答案」，洩漏的傷害就越大）。
4. 比較**測試集**準確率，各做 4 次取平均。
''')
code(r'''
#@title 2-6　先看資料長怎樣，以及兩種切法把「同一位病人」的檢查分到哪裡
rng = np.random.default_rng(0)

def make_multi_visit(d, visits=5, jitter=0.03):
    # 假設每位病人檢查了 5 次，每次的數值有一點點測量誤差
    out = []
    for v in range(visits):
        e = d.copy(); e["visit"] = v + 1
        for c in ["age", "resting_bp", "chol", "max_hr", "oldpeak"]:
            e[c] = (e[c] * rng.normal(1, jitter, len(e))).round(1)
        out.append(e)
    return pd.concat(out, ignore_index=True)

multi = make_multi_visit(trainval_df)
Xm, ym = to_xy(multi); groups = multi["patient_id"].values
print(f"{len(trainval_df)} 位病人 × 5 次檢查 = {len(multi)} 列。其中兩位病人的資料：")
two = trainval_df["patient_id"].iloc[:2].tolist()
show(multi[multi["patient_id"].isin(two)].sort_values(["patient_id", "visit"])
     [["patient_id", "visit", "age", "resting_bp", "chol", "max_hr", "disease"]])

# 兩種切法（random_state=0）會把每一列分到「訓練」還是「測試」？
tr_a, te_a = train_test_split(np.arange(len(multi)), test_size=0.2, random_state=0)
tr_b, te_b = next(GroupShuffleSplit(1, test_size=0.2, random_state=0).split(Xm, ym, groups))
side = pd.DataFrame({"patient_id": groups, "visit": multi["visit"], "A": "train", "B": "train"})
side.loc[te_a, "A"] = "test"; side.loc[te_b, "B"] = "test"
pa = side.pivot(index="patient_id", columns="visit", values="A")
pb = side.pivot(index="patient_id", columns="visit", values="B")
split_ids = pa[(pa == "test").any(axis=1) & (pa == "train").any(axis=1)].head(5).index
pa.columns = pb.columns = [f"第{v}次" for v in pa.columns]
print("\n❌ A：以『列』切分——同一位病人的 5 次檢查，被拆到訓練與測試兩邊：")
show(pa.loc[split_ids], index=True)
print("✅ B：以『病人』切分——同樣這幾位病人，5 次檢查全部在同一邊：")
show(pb.loc[split_ids], index=True)
''')
code(r'''
#@title 2-7　Leave-one-patient-out：每次把「一位病人的全部檢查」留下來驗證
from sklearn.model_selection import LeaveOneGroupOut
logo = LeaveOneGroupOut()
n_splits = logo.get_n_splits(groups=groups)
tr, te = next(logo.split(Xm, ym, groups))
held = multi.iloc[te][["patient_id", "visit"]]
print(f"{len(trainval_df)} 位病人（共 {len(multi)} 列）→ Leave-one-patient-out 要訓練 {n_splits} 次。")
print("第 1 次被留下來驗證的，是這位病人的『全部』檢查：")
show(held)
print(f"訓練集裡還找得到這位病人嗎？　{held['patient_id'].iloc[0] in set(multi.iloc[tr]['patient_id'])}")
print("👉 這樣，模型就不可能靠『認得這個人』來答題。")
''')
code(r'''
#@title 2-8　洩漏實驗：兩種切法，測試準確率差多少？
def fit_eval(tr, te, seed):
    sc = StandardScaler().fit(Xm[tr])        # 標準化：單元 3 會細講，這裡先用，讓模型學得更好
    model, _ = train_mlp(sc.transform(Xm[tr]), ym[tr], lr=0.01, seed=seed, hidden=(64, 32))
    # 這裡刻意用「大」的網路：越大的模型越會『背答案』，洩漏的傷害就越大
    return evaluate(model, sc.transform(Xm[te]), ym[te])["acc"]

acc_row, acc_pat = [], []
for seed in range(4):
    tr, te = train_test_split(np.arange(len(multi)), test_size=0.2, random_state=seed)              # ❌ A：以「列」切分
    acc_row.append(fit_eval(tr, te, seed))
    tr, te = next(GroupShuffleSplit(1, test_size=0.2, random_state=seed).split(Xm, ym, groups))     # ✅ B：以「病人」切分
    acc_pat.append(fit_eval(tr, te, seed))

print(f"❌ A：以『列』隨機切分（同一位病人跨訓練/測試）：測試準確率 {np.mean(acc_row):.1%}")
print(f"✅ B：以『病人』切分（每位病人只在一邊）　　　：測試準確率 {np.mean(acc_pat):.1%}")
plt.figure(figsize=(10.5, 6.8)); plt.bar(["A: split by row\n(leaky)", "B: split by patient\n(correct)"], [np.mean(acc_row), np.mean(acc_pat)],
                                      color=["#de2d26", "#3182bd"]); plt.ylim(0.5, 1); plt.ylabel("test accuracy"); plt.tight_layout(); plt.show()
print("\n👉 好到不真實的成績，要先懷疑『有沒有洩漏』，再開心。")
''')
md(r'''
**🎓 老師備註（單元 2）**
- 2-1 的準確率會明顯跳動（小網路、只做粗略縮放，準確率大約 70–90%）：這是刻意的，單元 3 會示範更好的縮放。
- 2-8 的差距大約是 99％ vs 78％。可以問：「如果這是一篇論文的結果，你會怎麼發現？」→ 看切分單位是病人還是影像／切片。
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
md(r'''
### 為什麼要標準化（正規化）？

想像兩個特徵：**體重（約 40–120 kg）**與**血鉀（約 3.5–5.5 mmol/L）**。
數字大的變數，在模型眼中「聲音比較大」。我們不講公式，只講：**把大家拉到同一把尺**。

「聲音大」為什麼會出問題？因為神經網路是靠**梯度**來調整參數的：**輸入越大，梯度越大，參數一次被「推」得越多**。
下面用**一個感知器**（ReLU）親手算一遍：把 loss 傳回去，看參數會被更新多少。
''')
fig('scale', '標準化：把大家拉到同一把尺')
fig('chain', '一個感知器：往前算、把 loss 傳回去')
code(r'''
#@title 3-1　動手算：一個感知器，把 loss 傳回去，看參數被更新多少
# 👈 可以改改看這些數字！
x  = [85, 4.6]       # 輸入：體重 85 kg、血鉀 4.6 mmol/L
w  = [0.1, 0.1]      # 兩個權重（一個對體重、一個對血鉀）
b  = 0.0             # 偏差
t  = 1.0             # 正確答案（希望模型輸出的值）
lr = 0.01            # 學習率：每次更新「走多大一步」

x, w = np.array(x, float), np.array(w, float)
# 【往前算】
z = (w * x).sum() + b                    # 加權總和
y = max(z, 0.0)                          # ReLU：負的變 0
loss = (y - t) ** 2                      # loss：和正確答案差多少（平方）
# 【把 loss 傳回去：連鎖律】
dL_dy = 2 * (y - t)                      # loss 對 y 的微分
dy_dz = 1.0 if z > 0 else 0.0            # ReLU 的微分：z>0 是 1，否則 0
dz_dw = x                                # z 對每個 w 的微分，就是對應的輸入 x
grad_w = dL_dy * dy_dz * dz_dw           # 三個相乘＝loss 對每個 w 的梯度
grad_b = dL_dy * dy_dz

print("【往前算】")
print(f"  z = w₁x₁ + w₂x₂ + b = {w[0]}×{x[0]:g} + {w[1]}×{x[1]:g} + {b} = {z:.3f}")
print(f"  y = ReLU(z) = max({z:.3f}, 0) = {y:.3f}")
print(f"  loss = (y − t)² = ({y:.3f} − {t:g})² = {loss:.3f}")
print("【把 loss 傳回去（連鎖律）】")
print(f"  ∂L/∂y = 2(y − t) = {dL_dy:.3f}")
print(f"  ∂y/∂z = {dy_dz:.0f}　（ReLU：z>0 → 1，否則 0）")
print(f"  ∂z/∂w₁ = x₁ = {x[0]:g}　　∂z/∂w₂ = x₂ = {x[1]:g}")
print(f"  ∂L/∂w₁ = {dL_dy:.3f} × {dy_dz:.0f} × {x[0]:g} = {grad_w[0]:.3f}")
print(f"  ∂L/∂w₂ = {dL_dy:.3f} × {dy_dz:.0f} × {x[1]:g} = {grad_w[1]:.3f}")
print("【更新】新的 w ＝ 舊的 w − 學習率 × 梯度")
print(f"  w₁（體重）：{w[0]} → {w[0] - lr * grad_w[0]:.3f}　（改變 {-lr * grad_w[0]:+.3f}）")
print(f"  w₂（血鉀）：{w[1]} → {w[1] - lr * grad_w[1]:.3f}　（改變 {-lr * grad_w[1]:+.3f}）")

# 用 PyTorch 的自動微分驗證
wt = torch.tensor(w, dtype=torch.float64, requires_grad=True)
(torch.relu((wt * torch.tensor(x)).sum() + b) - t).pow(2).backward()
print("\n✅ PyTorch 自動算出來的梯度也一樣：", np.allclose(wt.grad.numpy(), grad_w))
print(f"👉 體重的權重被推的幅度，是血鉀的 {abs(grad_w[0] / grad_w[1]):.1f} 倍：這就是「輸入大的變數，聲音比較大」。")
''')
code(r'''
#@title 3-2　同樣的規則，連續更新 5 步：沒標準化 vs 標準化
def run_steps(x, w, b, t, lr, steps=5):
    x, w = np.array(x, float), np.array(w, float)
    rows = []
    for k in range(steps):
        z = (w * x).sum() + b
        y = max(z, 0.0)
        loss = (y - t) ** 2
        g = 2 * (y - t) * (1.0 if z > 0 else 0.0)
        rows.append({"步": k, "z": round(z, 2), "y（ReLU 後）": round(y, 2), "loss": round(loss, 3),
                     "w₁（體重）": round(w[0], 3), "w₂（血鉀）": round(w[1], 3), "梯度 w₁": round(g * x[0], 2) + 0.0})
        w = w - lr * g * x
        b = b - lr * g
    return pd.DataFrame(rows)

# 標準化：假設體重平均 65、標準差 10；血鉀平均 4.2、標準差 0.4 → 85 kg 變成 2.0，4.6 變成 1.0
print("❌ 沒標準化：x ＝ [85, 4.6]，學習率 0.01")
show(run_steps([85, 4.6], [0.1, 0.1], 0.0, 1.0, 0.01))
print("✅ 標準化之後：x ＝ [2.0, 1.0]，同樣的學習率 0.01")
show(run_steps([2.0, 1.0], [0.1, 0.1], 0.0, 1.0, 0.01))
print("👉 沒標準化：第一步 w₁ 就被推到 −13 以下，z 變成很大的負數，ReLU 輸出 0、梯度也變 0 → 參數再也不動（神經元「死掉」）。")
print("   標準化之後：同樣的學習率，loss 穩穩下降。")
''')
code(r'''
#@title 3-3　標準化：把每個欄位拉到同一把尺上
cols = ["age", "resting_bp", "chol", "max_hr", "oldpeak", "ca"]
print("縮放之前，各欄位的『大小』差很多：")
show(trainval_df[cols].describe().loc[["mean", "std", "min", "max"]].round(1), index=True)
print("膽固醇動輒 200–300，oldpeak 只有 0–6。數值大的欄位會『壓過』數值小的欄位。")
print("（單元 1、2 的 to_xy 其實偷偷做了『粗略版』：每欄除以一個固定的數字，縮到大約 0–1。現在來比較三種做法。）\n")

def loss_curves(scale):
    prep = make_prep("onehot", scale)
    Xtr = prep.fit_transform(add_male(train_df)).astype("float32")       # 只用「訓練集」算出縮放需要的數字
    Xva = prep.transform(add_male(val_df)).astype("float32")             # 驗證集只能「套用」，不能參與計算
    ytr = (train_df["disease"] == "yes").values.astype("float32"); yva = (val_df["disease"] == "yes").values.astype("float32")
    return train_mlp(Xtr, ytr, X_val=Xva, y_val=yva)[1]

curves = {"no scaling": loss_curves("none"), "rough (divide by fixed number)": loss_curves("rough"), "standardized": loss_curves("standard")}
fig, ax = plt.subplots(1, 2, figsize=(15.0, 4.5), sharey=True)
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
#@title 3-4　類別資料：整數編碼 vs One-hot（之後的實驗都使用標準化）
print("原始表格：類別是文字")
show(trainval_df[["patient_id", "chest_pain", "thal"]].head(4))

print("\nOne-hot：每個類別變成一欄 0/1")
oh = pd.get_dummies(trainval_df[["chest_pain"]]).astype(int)
show(pd.concat([trainval_df[["patient_id", "chest_pain"]], oh], axis=1).head(4))

print("\n整數編碼會是 typical=0, nontypical=1, nonanginal=2, asymptomatic=3。")
print("模型會以為 asymptomatic(3) 是 typical(0) 的『三倍』、nonanginal 在『中間』，但這個順序是我們隨便排的。")
print("\n各類型實際的有病比例：")
show((trainval_df.groupby("chest_pain")["disease"].apply(lambda s: (s == "yes").mean())
      .round(2).to_frame("有病比例")), index=True)

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
md(r'''
### ⚠️ 洩漏（一）：標準化要用「誰」的平均與標準差？

標準化要先算出「平均」和「標準差」。問題是：這兩個數字，是用**哪些人**算的？

| | ❌ 洩漏的做法 | ✅ 正確的做法 |
|---|---|---|
| 第 1 步 | 用**全部** 242 人算平均與標準差 | 先分出「訓練折」與「驗證折」 |
| 第 2 步 | 再把 242 人分成訓練／驗證 | 只用「訓練折」的人算平均與標準差 |
| 結果 | 驗證的人**早就參與了計算**，他們的資訊「滲」進了訓練 | 驗證的人只是被「套用」這組數字，沒有參與計算 |

（這裡「驗證折」扮演的，就是「之後才會來考試的病人」。）
''')
code(r'''
#@title 3-5　⚠️ 洩漏：「先標準化全部資料、再切分」 vs 「只用訓練集的平均與標準差」
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
md(r'''
### ⚠️ 洩漏（二）：如果被偷看的是「答案」呢？

**設定**：242 位病人，每人有 **300 個純雜訊特徵**（隨機數字，沒有任何意義），標籤（有病／沒病）也是**隨機決定**的。
所以不管用什麼方法，理論上都只能猜對 **50%**。

**任務**：從 300 個特徵裡，挑出和標籤最相關的 10 個，再用這 10 個特徵訓練模型。

| | ❌ 洩漏的做法 | ✅ 正確的做法 |
|---|---|---|
| 怎麼挑特徵 | 用**全部** 242 人（包含之後要驗證的人）的答案來挑 | 只用「訓練折」的人的答案來挑 |
| 怎麼評分 | 在驗證折的人身上 | 在驗證折的人身上 |

（這裡**沒有動用鎖住的測試集**；「驗證折」扮演「之後才來考試的人」。）

**為什麼洩漏會變準？** 300 個雜訊裡，總會有幾個「碰巧」和答案相關。用全部人的答案去挑，等於偷看考卷，挑出「碰巧對」的特徵；之後驗證時，這些特徵當然「碰巧」又對。
''')
code(r'''
#@title 3-6　（加碼，時間夠再跑）洩漏有多可怕？用「完全隨機」的資料看看
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
#@title 3-7　資料擴增：表格資料加雜訊
_ = cv_eval(trainval_df, "E. + noise augmentation (x3)", encode="onehot", scale="standard", augment=noise_augment(3, 0.1))
print("\n👉 在這個小資料集上，擴增幾乎沒有幫助（差異在 ± 範圍內）。擴增不是萬靈丹，一樣要用驗證的方式檢查有沒有用。")
print("   而且：只能擴增『訓練集』。如果驗證集也擴增，同一個病人的變形版本就會跨過兩邊，又是洩漏。")
''')
code(r'''
#@title 3-8　影像資料的擴增（純示範，不訓練）：哪些變形在醫學上合理？
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
fig('imbalance', '類別不平衡')
md(r'''
### 先認識幾個詞：怎麼評量「抓病人」的能力？

假設 100 位病人，其中 15 位有病。模型的預測結果可以分成四格：

| | 實際有病 | 實際沒病 |
|---|---|---|
| **預測有病** | TP 真陽性（抓到了） | FP 偽陽性（誤報） |
| **預測沒病** | FN 偽陰性（**漏診**） | TN 真陰性 |

- **accuracy（準確率）**＝(TP＋TN) ÷ 全部人：全部猜對幾成
- **recall（召回率／敏感度）**＝TP ÷ (TP＋FN)：**有病的人，抓到幾成**
- **precision（精確率）**＝TP ÷ (TP＋FP)：**被預測有病的人，真的有病幾成**
- **AUC**：不看門檻，模型把「有病的人」排在「沒病的人」前面的能力（0.5＝亂猜，1＝完美）

三種訓練方式：

| 名稱 | 做法 |
|---|---|
| **plain**（一般訓練） | 什麼都不特別處理。有病的人太少，模型容易偏向猜「沒病」 |
| **class weight**（類別權重） | 訓練時，漏掉一個「有病」的人，損失乘上約（沒病人數 ÷ 有病人數）倍，逼模型重視少數 |
| **oversample**（過採樣） | 把「有病」的人**重複抽樣複製**，補到跟沒病的一樣多。只對**訓練集**做，驗證時不做 |
''')
code(r'''
#@title 3-9　資料不平衡：罕見疾病只占一小部分
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

from sklearn.metrics import confusion_matrix
n_pos_all, n_neg_all = int(yi.sum()), int((1 - yi).sum())
rows = [{"strategy": "always predict 'no disease'", "TP": 0, "FN": n_pos_all, "FP": 0, "TN": n_neg_all,
         "accuracy": 1 - yi.mean(), "recall": 0.0, "precision": float("nan"), "AUC": 0.5}]
for s in ["plain", "class weight", "oversample"]:
    P = oof_proba(s); pred = (P >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(yi, pred).ravel()
    rows.append({"strategy": s, "TP": tp, "FN": fn, "FP": fp, "TN": tn, "accuracy": accuracy_score(yi, pred),
                 "recall": recall_score(yi, pred), "precision": precision_score(yi, pred, zero_division=0), "AUC": roc_auc_score(yi, P)})
show(pd.DataFrame(rows).round(3))
print("👉 『全部猜沒病』的準確率就有 85％，recall（有病的人抓到幾成）卻是 0。罕見疾病不能只看 accuracy。")
print("   class weight、過採樣可以多抓到一些病人（recall 上升），代價通常是 precision 下降（誤報變多）。要選哪個，取決於漏診與誤報哪個代價更大。")
''')
code(r'''
#@title 3-10　總結：每一種前處理，到底值多少？
res = pd.DataFrame(RESULTS)
show(res.round(3))
plt.figure(figsize=(10.5, 4.5)); plt.barh(res["label"][::-1], res["acc"][::-1], xerr=res["acc_sd"][::-1], color="#3182bd")
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
#@title 4-1　定案：看看「前處理」到底記住了哪些數字，再訓練並存起來
import joblib
d_tv = add_male(trainval_df)

# 補缺值用的數字：只用「訓練＋驗證」資料算
fill = {**{c: trainval_df[c].median() for c in ALL_NUMERIC},
        **{c: trainval_df[c].mode()[0] for c in ["sex", "chest_pain", "thal"]}}

# 建立前處理，並且只用「訓練＋驗證」資料 fit（＝讓它記住這些資料的平均、標準差、類別順序）
final_prep = make_prep("onehot", "standard")
X_final = final_prep.fit_transform(d_tv).astype("float32")
y_final = (trainval_df["disease"] == "yes").astype("float32").values

scaler = final_prep.named_transformers_["num"]
print("① 標準化記住的數字：每個連續欄位，在訓練資料裡的平均與標準差")
stats = pd.DataFrame({"欄位": PREP_NUM, "mean 平均": scaler.mean_, "std 標準差": scaler.scale_,
                      "median 中位數（補缺值用）": [fill[c] for c in PREP_NUM]})
show(stats.round(2))
ex = trainval_df["chol"].iloc[0]
print(f"   例：某位病人的膽固醇 {ex:.0f} → (數值 − 平均) ÷ 標準差 = ({ex:.0f} − {scaler.mean_[2]:.1f}) ÷ {scaler.scale_[2]:.1f} = {(ex - scaler.mean_[2]) / scaler.scale_[2]:.2f}")

print("\n② One-hot 記住的：每個類別欄位有哪些類別、順序是什麼（順序錯了，模型就看錯欄位）")
oh = final_prep.named_transformers_["onehot"]
nice = lambda c: [x if isinstance(x, str) else int(x) for x in c]          # 把 0.0 顯示成 0
show(pd.DataFrame({"欄位": PREP_CAT, "類別（依序展開成 0/1 欄）": [nice(c) for c in oh.categories_]}))

print("\n③ 補缺值用的數字（缺值不能用『新資料自己的中位數』，要用訓練時算好的）：")
show(pd.DataFrame({"欄位": list(fill), "補上的值": list(fill.values())}).round(2))

print(f"\n④ 前處理後，模型看到的 {X_final.shape[1]} 個欄位（依序）：")
names = [n.split("__", 1)[1] for n in final_prep.get_feature_names_out()]
groups, i = [], 0
groups.append((f"標準化後的連續欄位（{len(PREP_NUM)}）", PREP_NUM)); i += len(PREP_NUM)
groups.append((f"本來就是 0/1（{len(PREP_BIN)}）", PREP_BIN)); i += len(PREP_BIN)
for col, cats in zip(PREP_CAT, oh.categories_):
    groups.append((f"one-hot：{col}（{len(cats)}）", names[i:i + len(cats)])); i += len(cats)
show(pd.DataFrame({"來源": [g[0] for g in groups], "展開後的欄位（依序）": [", ".join(g[1]) for g in groups]}))

final_model, _ = train_mlp(X_final, y_final, seed=0)
ART = os.path.join(WORKDIR, "artifacts"); os.makedirs(ART, exist_ok=True)
joblib.dump({"prep": final_prep, "fill": fill}, os.path.join(ART, "preprocessor.joblib"))
torch.save(final_model.state_dict(), os.path.join(ART, "model.pt"))
print("\n✅ 已存到 Drive：", sorted(os.listdir(ART)))
''')
code(r'''
#@title 4-1b　打開剛存的檔案看看：preprocessor.joblib 與 model.pt 裡面是什麼？
bundle = joblib.load(os.path.join(ART, "preprocessor.joblib"))
print("preprocessor.joblib 是一個『包裹』，裡面有：", list(bundle.keys()))
print(f"  • prep：{type(bundle['prep']).__name__}　→ 內含上面的 平均／標準差（①）與 類別順序（②）")
print(f"  • fill：{type(bundle['fill']).__name__}　→ 補缺值用的 {len(bundle['fill'])} 個數字（③）")
print("  → 推論時，只要把新病人丟給 prep.transform(...)，就會自動套用『訓練時』記住的一切。")

state = torch.load(os.path.join(ART, "model.pt"))
show(pd.DataFrame({"層": list(state.keys()), "形狀": [tuple(v.shape) for v in state.values()],
                   "參數個數": [v.numel() for v in state.values()]}))
print(f"model.pt 是模型學到的『權重』：總共只有 {sum(v.numel() for v in state.values())} 個數字（這就是整個 AI）。")
''')
code(r'''
#@title 4-2　🔓 打開鎖住的測試集（第一步：讀進來）
test_locked = pd.read_csv(os.path.join(WORKDIR, "heart_test_locked.csv"))
test_locked[ALL_NUMERIC] = test_locked[ALL_NUMERIC].astype(float)
print(f"鎖了很久的測試集：{len(test_locked)} 位病人。單元 2 鎖起來之後，沒有人看過它，也沒拿它調整過任何東西。")
print("它是「已經清理好」的資料（文字類別、原始數值，還沒標準化／one-hot）：")
show(test_locked.head(8))
''')
code(r'''
#@title 4-3　第二步：套用「訓練時記住的」前處理
X_test = final_prep.transform(add_male(test_locked)).astype("float32")
y_test = (test_locked["disease"] == "yes").astype("float32").values
X_test_df = pd.DataFrame(X_test, columns=names)
print("處理完的測試集（連續欄位已標準化、類別已展開成 0/1）：")
show(X_test_df.head(8).round(2))
i = 0
print(f"\n對照第一位病人：膽固醇 {test_locked['chol'].iloc[i]:.0f} → "
      f"({test_locked['chol'].iloc[i]:.0f} − {scaler.mean_[2]:.1f}) ÷ {scaler.scale_[2]:.1f} = {X_test_df['chol'].iloc[i]:.2f}　（用的是『訓練資料』的平均與標準差）")
print(f"胸痛類型 {test_locked['chest_pain'].iloc[i]} → "
      + ", ".join(f"{c}={X_test_df[c].iloc[i]:.0f}" for c in names if c.startswith("chest_pain")))
''')
code(r'''
#@title 4-4　第三步：模型的推論結果
p = predict_proba(final_model, X_test)
result = pd.DataFrame({"patient_id": test_locked["patient_id"], "預測有病的機率": p.round(3),
                       "預測": np.where(p >= 0.5, "yes", "no"), "實際": test_locked["disease"]})
result["對不對"] = np.where(result["預測"] == result["實際"], "✅", "❌")
print("前 12 位病人的推論結果：")
show(result.head(12))

res = evaluate(final_model, X_test, y_test)
tn = int(((p < 0.5) & (y_test == 0)).sum()); fp = int(((p >= 0.5) & (y_test == 0)).sum())
fn = int(((p < 0.5) & (y_test == 1)).sum()); tp = int(((p >= 0.5) & (y_test == 1)).sum())
print(f"\n全部 {len(y_test)} 位：TP {tp}　FN {fn}　FP {fp}　TN {tn}")
print(f"accuracy {res['acc']:.1%}　recall {res['recall']:.1%}　precision {res['precision']:.1%}　AUC {res['auc']:.3f}")
print("\n👉 這才是對『新病人』的誠實成績。如果現在回頭調整模型再重測，測試集就變成另一份驗證集，失去意義。")
''')
code(r'''
#@title 4-5　一位新病人走進診間：原始表單 → 同樣的前處理 → 預測
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
#@title 4-6　故意做錯：前處理跟訓練時不一致，會發生什麼事？
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
table["預測機率"] = table["預測機率"].map("{:.1%}".format)
show(table)

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
---
## 單元 4 補充｜去識別化與隱私

我們今天用的是**公開、已去識別**的資料集。但你們以後做研究，拿到的是**院內病人的資料**。
下面先看：研究資料裡，**哪些東西可以認出病人**？
''')
fig('privacy', '研究資料裡藏著哪些可以認出病人的東西')
md(r'''
### ① 影像：臉部可以被重建
頭部 **MRI／CT** 是 3D 的，把表面重建出來就是一張臉，已有研究證明可以用人臉辨識軟體比對到真人（例如 Schwarz 等人，*NEJM* 2019）。
所以影像研究常要做「**去臉（defacing）**」，或只提供去除顱骨與臉部的腦組織。

### ② DICOM 標頭：影像檔「自己帶著」病人資料
醫學影像（DICOM）檔案除了像素，還有一大段**標頭（header）**。下面是一個**虛構**的範例：
''')
code(r'''
#@title 4-7　DICOM 標頭長什麼樣子？（範例；病人與資料皆為虛構）
header = [
    # (tag, 名稱, 原始值, 風險, 處理方式, 去識別化後)
    ("(0010,0010)", "PatientName", "CHEN^DAWEN", "直接識別", "取代", "ANON-0042"),
    ("(0010,0020)", "PatientID（病歷號）", "A123456789", "直接識別", "取代（對照表另外保管）", "ANON-0042"),
    ("(0010,0030)", "PatientBirthDate", "19680312", "直接識別", "只留年份", "1968"),
    ("(0010,1040)", "PatientAddress", "台北市中正區某某路 1 號", "直接識別", "刪除", "（空）"),
    ("(0010,2154)", "PatientTelephoneNumbers", "0912-345-678", "直接識別", "刪除", "（空）"),
    ("(0008,0050)", "AccessionNumber", "H2025110400123", "直接識別", "刪除", "（空）"),
    ("(0008,0090)", "ReferringPhysicianName", "WANG^XIAOMING", "直接識別", "刪除", "（空）"),
    ("(0008,1070)", "OperatorsName", "LI^TECH", "直接識別", "刪除", "（空）"),
    ("(0010,0040)", "PatientSex", "M", "間接識別", "保留（研究需要）", "M"),
    ("(0010,1010)", "PatientAge", "057Y", "間接識別", "保留或分組", "050-059Y"),
    ("(0008,0020)", "StudyDate", "20251104", "間接識別", "日期位移（同一病人位移相同天數）", "20250918"),
    ("(0008,0030)", "StudyTime", "093015", "間接識別", "刪除", "（空）"),
    ("(0008,0080)", "InstitutionName", "Example General Hospital", "間接識別", "刪除", "（空）"),
    ("(0018,1000)", "DeviceSerialNumber", "SN-483920", "間接識別", "刪除", "（空）"),
    ("(0008,1030)", "StudyDescription", "MRI BRAIN, r/o stroke", "間接識別", "保留，但要檢查有沒有夾帶姓名", "MRI BRAIN, r/o stroke"),
    ("(0020,000D)", "StudyInstanceUID", "1.2.840.113619.2.55.3.60468…", "間接識別", "重新產生（對照表另外保管）", "2.25.3175…（新的）"),
    ("(0008,0060)", "Modality", "MR", "不需處理", "保留", "MR"),
    ("(0018,0050)", "SliceThickness", "1.0", "不需處理", "保留", "1.0"),
    ("(0028,0030)", "PixelSpacing", "0.5\\0.5", "不需處理", "保留", "0.5\\0.5"),
]
dicom = pd.DataFrame(header, columns=["Tag", "名稱", "原始值", "風險", "處理方式", "去識別化後"])
print("19 個標頭欄位（節錄）。風險：直接識別 ＝ 一看就知道是誰；間接識別 ＝ 湊在一起可能認出人。")
show(dicom)
print("⚠️ 還有一個陷阱：像素本身可能「燒」進了文字（例如超音波影像上的病人姓名），只改標頭不夠，要另外檢查影像內容。")
print("DICOM 標準有制式的去識別化設定檔（PS3.15 附錄 E：Basic Application Level Confidentiality Profile），可以當作清單參考。")
''')
md(r'''
### ③ 病歷文字：姓名、日期、號碼
''')
code(r'''
#@title 4-8　病歷文字的去識別化：用規則把姓名、日期、號碼拿掉（範例；內容為虛構）
import re
note = ("2025/11/04 門診。病人陳大文（病歷號 12345678，身分證 A123456789，手機 0912-345-678），57 歲男性，"
        "住台北市中正區某某路1號。主訴左側肢體無力 2 小時，由女兒陳小美陪同就醫。2025-11-04 09:30 頭部 MRI 顯示右側 MCA 梗塞。"
        "預計 2025年11月18日 回診，由王小明醫師追蹤。")
names_in_note = ["陳大文", "陳小美", "王小明"]            # 示範用：真實情況要靠名單或自然語言處理（NER）找人名
rules = [(r"[A-Z][12]\d{8}", "[身分證]"), (r"09\d{2}-?\d{3}-?\d{3}", "[電話]"), (r"病歷號\s*\d{6,8}", "病歷號 [病歷號]"),
         (r"\d{4}[/-]\d{1,2}[/-]\d{1,2}(\s*\d{1,2}:\d{2})?", "[日期]"), (r"\d{4}年\d{1,2}月\d{1,2}日", "[日期]"),
         (r"(台北|新北|桃園|台中|台南|高雄)市[^，。\s]*\d+號", "[地址]")]
clean_note, hits = note, {}
for pat, tag in rules:
    clean_note, n = re.subn(pat, tag, clean_note)
    hits[tag] = hits.get(tag, 0) + n
for nm in names_in_note:
    clean_note = clean_note.replace(nm, "[姓名]"); hits["[姓名]"] = hits.get("[姓名]", 0) + note.count(nm)
print("【原文】\n" + note + "\n")
print("【去識別化之後】\n" + clean_note + "\n")
show(pd.DataFrame({"取代成": list(hits), "次數": list(hits.values())}))
print("規則做不到的事：沒列在名單上的人名、「某某的女兒」這類關係、手寫或口語的日期寫法…")
print("更難的是「間接識別」：57 歲 ＋ 左側無力 ＋ 某區居民，湊在一起，罕見的話還是可能認出人。所以去識別化之後仍然要人工抽查。")
''')
md(r'''
### 台灣的相關規範
> 以下是教學用的**重點整理**（資料查詢日期：2026-11-04），**不是法律意見**。實際做研究時，請以全國法規資料庫的現行條文，以及你所屬醫院 **IRB** 的規定為準。

| 規範 | 與我們的研究資料有關的重點 |
|---|---|
| **個人資料保護法 第 6 條** | **病歷、醫療、基因、性生活、健康檢查、犯罪前科**屬「特種個資」，原則上**不得蒐集、處理或利用**。例外包括：① **公務機關或學術研究機構**基於醫療、衛生或犯罪預防的目的，為統計或學術研究而有必要，且資料經提供者處理後，或依蒐集者的揭露方式，**無從識別特定當事人**；② **經當事人書面同意**（有但書限制）；另外還有法律明文規定、法定職務或義務等例外。 |
| **人體研究法 第 5 條** ＋ 衛福部公告 | 人體研究原則上，**研究前要先擬計畫，經倫理審查委員會（IRB）審查通過**；屬衛福部公告「得免審查」範圍的，才不在此限。 |
| **衛福部公告：得免倫理審查委員會審查之人體研究案件範圍**（101.7.5 衛署醫字第 1010265075 號） | 免審有**嚴格前提**：研究對象不能是未成年人、收容人、原住民、孕婦、身心障礙、精神病患或其他可能無法自由決定者；範圍包含「非記名、非互動、非介入性且**無從辨識特定個人**」的研究，以及風險極低、經 IRB 評估後核發免審證明者。**要不要免審，是由 IRB 判定，不是研究者自己決定。** |
| **衛福部公告：得免取得研究對象同意之人體研究案件範圍**（101.7.5 衛署醫字第 1010265083 號） | 包含「自**合法生物資料庫**取得之**去連結或無法辨識特定個人**的資料、檔案、文件、資訊或檢體進行研究」，但**不包括涉及族群或群體利益者**。 |
| **衛福部函釋**（衛部醫字第 1030013183 號） | 對於未成年人、收容人、原住民、孕婦、身心障礙、精神病患等族群的資料，**無論是否去連結**，都**不屬於**免審範圍；另外，針對檢體，「去連結」不等於可以轉作原始目的以外的用途。 |
| **「去連結」與「去識別化」** | **去連結**：以匿名化等方式，讓資料**永久無法**以合理方式連結、比對出當事人；**去識別化**：資料經處理後，無從識別特定當事人。**只要還能連結回個人，就仍然是「可識別」**——把姓名換成代號、但自己手上保留對照表，通常還不算去連結。 |

**給未來研究者的三個實務提醒**
1. **先問你的 IRB**：這個研究是免審、簡易審查，還是一般審查？不要自己判斷。
2. **不要把含個資的資料貼到公開的 AI 服務**（包括聊天機器人），哪怕只是「幫我看一下這段病歷」。
3. 今天的資料集是**公開、已去識別**的；用**院內資料**做研究時，上面每一項都要處理。

資料來源：[個資法第 6 條（個人資料保護委員會籌備處）](https://www.pdpc.gov.tw/News_Content/100/293/)、[衛福部：得免倫理審查委員會審查之人體研究案件範圍](https://dep.mohw.gov.tw/DOMA/dl-16078-31bb0e95-2407-4323-9bbf-a62b7f8dcee3.html)、[臺北醫學大學轉知：衛福部函釋去連結檢體使用規範及免審範圍](https://ohr.tmu.edu.tw/index.php/zh-Hant/news/133)、[全國法規資料庫（人體研究法）](https://law.moj.gov.tw/)。
''')
md(r'''
### 收尾：把今天帶走

**資料不只是數字，它還有來源與隱私**
- **去識別化**：影像臉部、DICOM 標頭、病歷文字（上一節）。研究前要經 IRB，並依個資法處理。
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
