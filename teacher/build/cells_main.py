# -*- coding: utf-8 -*-
# 每個 cell 是 (type, source, flags)。flags: form=收合的程式格、teacher=只在老師版出現
CELLS = []
def md(s, teacher=False): CELLS.append(("md", s.strip("\n"), dict(teacher=teacher)))
def code(s, form=False, teacher=False): CELLS.append(("code", s.strip("\n"), dict(form=form, teacher=teacher)))

# ============================== 開頭 ==============================
md(r'''
# 🫀 AI 的資料前處理基礎：動手做做看
**兩小時工作坊｜給零程式背景的醫學生**

> 今天不教你寫程式。程式都已經寫好了，你只需要 **按 ▶ 執行（或 Shift+Enter）**，並在標示 **👉 你的任務** 的地方動手。
> 目標是讓你親眼看到：**資料怎麼處理，會直接決定 AI 學到什麼、你能不能相信它。**

**今天的流程**

| 單元 | 內容 | 你會看到 |
|---|---|---|
| 0 | 準備環境 | 授權 Google Drive |
| 1 | 資料如何才能「餵進去」 | 髒資料讓模型壞掉；你手動清理 vs 程式自動清理 |
| 2 | 資料怎麼分配 | 訓練／驗證／測試、交叉驗證、資料洩漏 |
| 3 | 前處理如何影響訓練 | One-hot、標準化、擴增、資料不平衡 |
| 4 | 推論 | 新病人進來，前處理必須跟訓練時一模一樣 |

**資料來源**：UCI Heart Disease（Cleveland 資料集），Detrano et al., *Am J Cardiol* 1989；ISLR 整理版；CC BY 4.0。
本 notebook 裡的「錯誤」是老師**刻意製造**的，用於教學；模型僅供學習，**不可用於臨床**。
''')

# ============================== 單元 0 ==============================
md('## 單元 0｜準備環境')
code(r'''
#@title 0-1　準備環境（會跳出 Google Drive 授權視窗，請按「允許」）
import os, sys, io, json, time, base64, zlib, warnings, urllib.request
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")

IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    from google.colab import drive
    drive.mount("/content/drive")
    WORKDIR = "/content/drive/MyDrive/AI_preprocessing_class"
else:                                            # 不在 Colab 時（老師測試用）
    WORKDIR = os.path.join(os.path.expanduser("~"), "AI_preprocessing_class")
os.makedirs(WORKDIR, exist_ok=True)
print("✅ 你的工作資料夾：", WORKDIR)
print("   之後所有檔案都會存在這裡，打開 Google 雲端硬碟就看得到。")
''')
code(r'''
#@title 0-2　備援資料（網路出問題時才會用到，不用展開）
_EMBEDDED = __EMBEDDED__
''', form=True)
code(r'''
#@title 0-3　載入資料
BASE_URL = "https://raw.githubusercontent.com/Po-yuLin/MedicalAICourse_DataPreprocessing/v1.0/data/"

def load_csv(name):
    """先從 GitHub 讀；失敗就改用內嵌的備援資料。全部以文字讀入，避免 pandas 偷偷幫你改資料。"""
    try:
        with urllib.request.urlopen(BASE_URL + name, timeout=10) as r:
            text, src = r.read().decode("utf-8"), "GitHub"
    except Exception:
        text, src = zlib.decompress(base64.b64decode(_EMBEDDED[name])).decode("utf-8"), "內嵌備援"
    print(f"  {name}：從「{src}」載入")
    return pd.read_csv(io.StringIO(text), dtype=str, keep_default_na=False)

df_dirty_small = load_csv("heart_dirty_small.csv")    # 40 位病人：給你手動清理
df_dirty_full  = load_csv("heart_dirty_full.csv")     # 全部病人：等一下讓程式自動清理
print(f"\n練習用小資料：{len(df_dirty_small)} 列；完整資料：{len(df_dirty_full)} 列")
''')
code(r'''
#@title 0-4　工具箱（檢查器、小型 MLP、前處理管線；不用展開）
import torch, torch.nn as nn
import matplotlib.pyplot as plt
torch.set_num_threads(2)
__CORE_CLEAN__
__CORE_ML__
__CORE_PREP__
print("✅ 工具箱就緒（PyTorch", torch.__version__ + "）")
''', form=True)

# ============================== 單元 1 ==============================
md(r'''
---
## 單元 1｜資料如何才能「餵進去」

AI 模型只吃**數字**。每一位病人是一列，每一欄是一個**特徵（feature）**，我們想預測的答案是**標籤（label）**：這裡是 `disease`（有沒有冠心病）。

### 資料字典

| 欄位 | 意義 | 合理範圍／標準寫法 |
|---|---|---|
| `patient_id` | 病人編號 | P001…；**每位病人只應出現一次** |
| `age` | 年齡（歲） | 18–100 |
| `sex` | 性別 | `male` / `female` |
| `chest_pain` | 胸痛類型 | `typical`（典型心絞痛）、`nontypical`（非典型）、`nonanginal`（非心絞痛）、`asymptomatic`（無症狀） |
| `resting_bp` | 休息時血壓（mmHg） | 70–250，**只要數字** |
| `chol` | 血清膽固醇（**mg/dL**） | 100–600，**只要數字** |
| `fbs` | 空腹血糖 > 120 mg/dL？ | `1`＝是、`0`＝否 |
| `rest_ecg` | 休息心電圖 | `0` 正常、`1` ST-T 異常、`2` 左心室肥厚 |
| `max_hr` | 運動測試最大心跳（bpm） | 60–220，**只要數字** |
| `exang` | 運動誘發心絞痛？ | `1`＝是、`0`＝否 |
| `oldpeak` | 運動後 ST 段下降（mm） | 0–8 |
| `slope` | 運動高峰 ST 段斜率 | `1` 上升、`2` 平、`3` 下降 |
| `ca` | 透視下有鈣化的主要血管數 | 0–3 |
| `thal` | 鉈（thallium）心肌灌注掃描 | `normal` / `fixed` / `reversible` |
| `disease` | **標籤**：有冠心病嗎？ | `yes` / `no` |

> 你不需要懂這些指標的臨床意義，只需要知道**每一欄「應該長什麼樣子」**。
''')
code(r'''
#@title 1-1　👀 先看看：直接把髒資料餵給模型會怎樣？
print("髒資料的前 8 列：")
display(df_dirty_small.head(8))

# 硬把每個欄位轉成數字；轉不了的（空白、?、NA、"120 mmHg"…）會變成 NaN（Not a Number）
d_bad = df_dirty_small.copy()
for c in ALL_NUMERIC:
    d_bad[c] = pd.to_numeric(d_bad[c], errors="coerce")
print("\n硬轉成數字後，這些欄位出現了 NaN：")
print(d_bad[ALL_NUMERIC].isna().sum()[lambda s: s > 0].to_string())

X_bad = add_male(d_bad)[FEATURES_V1].values.astype("float32")
y_bad = df_dirty_small["disease"].str.strip().str.lower().isin(["yes", "y", "1"]).values.astype("float32")
_, h_bad = train_mlp(X_bad, y_bad, epochs=10)
print("\n訓練 10 個 epoch，每一步的 loss（模型的『錯誤程度』）：")
print([round(v, 3) for v in h_bad["train"]])
print("\n👉 loss 變成 nan：模型一碰到 NaN，整個訓練就壞掉了，而且不會告訴你為什麼。")
''')
code(r'''
#@title 1-2　⚠️ 更可怕的是「不會報錯」的錯誤
print("sex 欄位實際出現的寫法：")
print(df_dirty_small["sex"].value_counts().to_string())

naive_male = (df_dirty_small["sex"] == "male").sum()
real_male  = df_dirty_small["sex"].str.strip().str.lower().isin(["m", "male"]).sum()
print(f"\n如果程式寫「sex == 'male'」，會數到 {naive_male} 位男性；")
print(f"但實際上男性有 {real_male} 位。有 {real_male - naive_male} 位被默默當成女性，程式完全沒有報錯。")
''')
code(r'''
#@title 1-3　建立練習用的 Google 試算表（第一次會跳出授權視窗）
MODE = "sheets"          # "sheets"＝Google 試算表（建議）；"csv"＝下載 CSV 編輯後再上傳（備案，見下方說明）
SHEET_TITLE = "heart_dirty_practice"
CLEANED_CSV = os.path.join(WORKDIR, "heart_cleaned_small.csv")      # csv 模式：請把清理好的檔案用這個名字存到 Drive 資料夾

df_dirty_small.to_csv(os.path.join(WORKDIR, "heart_dirty_small.csv"), index=False)   # 備案用：Drive 資料夾裡也放一份

def _to_cell(v):
    """數字寫成數字，其他維持文字（試算表裡看起來才像真的資料）。"""
    try:
        f = float(v)
        return int(f) if (f.is_integer() and "." not in v) else f
    except (ValueError, TypeError):
        return v

if MODE == "sheets" and not IN_COLAB:
    print("（不在 Colab 環境，自動改用 csv 模式）"); MODE = "csv"

if MODE == "sheets":
    from google.colab import auth
    auth.authenticate_user()
    import gspread
    from google.auth import default
    gc = gspread.authorize(default()[0])
    try:
        sh = gc.open(SHEET_TITLE)
        print("✅ 找到先前建立的試算表，你之前的修改都還在。")
    except gspread.SpreadsheetNotFound:
        sh = gc.create(SHEET_TITLE)
        rows = [list(df_dirty_small.columns)] + [[_to_cell(v) for v in r] for r in df_dirty_small.values.tolist()]
        sh.sheet1.update(values=rows, range_name="A1")
        print("✅ 已建立新的試算表。")
    print("\n👉 請點開這個連結，開始清理（也可以在 Google 雲端硬碟找到它）：")
    print("  ", sh.url)
    print("\n（想整個重來：把這個試算表丟進垃圾桶，再重新執行這一格）")
else:
    print("csv 模式：")
    print(f"  1. 到 Google 雲端硬碟 → AI_preprocessing_class 資料夾 → 找到 heart_dirty_small.csv")
    print( "  2. 用 Google 試算表（或 Excel）開啟，修改")
    print( "  3. 另存／下載成 CSV，檔名改成 heart_cleaned_small.csv，上傳回同一個資料夾")
''')
md(r'''
### 👉 你的任務（約 15 分鐘，兩人一組）

打開試算表，把這 40 位病人的資料**清理到可以餵給模型**。每種問題各有幾個，找出來並修好：

| 問題類型 | 怎麼辨認 | 可以怎麼處理 |
|---|---|---|
| 缺值 | 空白、`?`、`NA`、`unknown`、`N/A`… | 補一個合理的值（例如該欄中位數），或刪掉整列 |
| 不可能的數值 | 超出上面表格的合理範圍 | 能判斷是打錯（多打一個 0？）就改；不能判斷就當缺值 |
| 單位不一致 | 膽固醇應該是 100–600；出現 5.46 可能是 mmol/L | mg/dL ＝ mmol/L × 38.67 |
| 數字欄位混入文字 | `110 mmHg`、`143bpm` | 只留下數字 |
| 類別寫法不一 | `M`/`Male`、`ASYMPTOMATIC`、`reversable`、`1`/`Y`… | 改成上面表格的**標準寫法**（小寫、沒有多餘空格） |
| 重複列 | 同一個 `patient_id` 出現兩次 | 只留一列 |

**小技巧**：`Ctrl+H`（尋找與取代）、「資料 → 資料清理 → 移除重複項目」、點欄位標題排序，會讓你快很多。
**規則**：不要改標題列（第 1 列）的欄位名稱。
**想一想**：遇到「看起來很誇張」的數值，你要怎麼判斷它是**打錯**還是**真的**？刪掉它，有沒有代價？

修完之後，執行下面兩格：先檢查，再訓練。檢查會告訴你還剩什麼問題；卡住就把 `HINT` 調大一點。
''')
md(r'''
**🎓 老師備註（單元 1）**
- 40 列裡共注入：缺值 5＋原本就缺 2、不可能值 4、單位錯誤 3、文字混入 4、類別寫法 11、重複 2。完整答案：`teacher/answer_key_small.csv`。
- **陷阱題（刻意保留）**：`P153` 的 chol = 564、`P092` 的 oldpeak = 6.2，都是**真實**極端值，不是錯誤。檢查器不會抓它們。若學生刪除或改動，正好討論「由有臨床判斷力的人決定」。
- 原本就缺值的是 `P088`（thal）與 `P167`（ca）：不是我們造的，是資料集本來就有。
- 學生卡住或你想直接示範：執行下一格「老師捷徑」，等於一位完美的學生。
''', teacher=True)
code(r'''
#@title [老師] 捷徑：模擬一位把資料清得很完美的學生（學生版沒有這一格）
_perfect = auto_clean(df_dirty_small)
if MODE == "csv":
    _perfect.to_csv(CLEANED_CSV, index=False)
else:
    sh.sheet1.clear()
    sh.sheet1.update(values=[list(_perfect.columns)] + _perfect.values.tolist(), range_name="A1")
print("✅ 已用程式清理並寫回（相當於一位完美學生）。")
''', teacher=True)
code(r'''
#@title 1-4　讀回你清理的結果並檢查（可以重複執行，直到全部通過）
HINT = 1     # 1：只告訴你還有幾類問題　2：再告訴你是哪幾個欄位　3：連 patient_id 都告訴你

def read_student_table():
    if MODE == "sheets":
        rows = sh.sheet1.get_all_values()
        d = pd.DataFrame(rows[1:], columns=rows[0])
    else:
        d = pd.read_csv(CLEANED_CSV, dtype=str, keep_default_na=False)
    return d[d.apply(lambda r: "".join(map(str, r)).strip() != "", axis=1)].reset_index(drop=True)   # 去掉全空白的列

df_student = read_student_table()
print(f"讀到 {len(df_student)} 列（原本 {len(df_dirty_small)} 列）")
if len(df_student) < len(df_dirty_small) - 6:
    print("⚠️ 你刪掉的列比較多。刪掉的每一列都是少一位病人的資訊，確定每一列都真的該刪嗎？")
all_ok = check_cleaned(df_student, hint=HINT)
''')
code(r'''
#@title 1-5　用你清好的資料，訓練一個超小的神經網路（輸入 → 3 個感知器 → 1 個感知器，共 4 個）
def train_on_table(df, epochs=300):
    try:
        X, y = to_xy(df)
    except Exception as e:
        print("❌ 資料還沒辦法餵進模型。Python 的錯誤訊息是：\n  ", e)
        print("   → 回到 1-4 看檢查結果；通常是數字欄位裡還有文字或空白。")
        return None
    if np.isnan(X).any():
        print("⚠️ 資料裡還有 NaN，loss 會變成 nan。")
    model, hist = train_mlp(X, y, epochs=epochs)
    plt.figure(figsize=(5.5, 3)); plt.plot(hist["train"])
    plt.xlabel("epoch"); plt.ylabel("training loss"); plt.title("Training loss"); plt.tight_layout(); plt.show()
    acc = evaluate(model, X, y)["acc"]
    print(f"訓練 {epochs} 個 epoch：loss {hist['train'][0]:.3f} → {hist['train'][-1]:.3f}；"
          f"在這 {len(y)} 位病人上的準確率 {acc:.1%}")
    print("（to_xy 會先把每個欄位縮到大約 0–1，小網路才學得起來；單元 3 會解釋為什麼。）")
    return model, hist

result_student = train_on_table(df_student)
''')
md(r'''
### 同一件事，讓程式來做

你剛剛花了十幾分鐘，只清理了 **40 列**。真實世界的資料動輒幾萬、幾百萬列，不可能手動。
但請注意：程式做的，其實就是你剛剛做的每一個判斷，只是把它們**寫成規則**——所以**你得先懂資料有哪些問題**，程式才寫得出來。
''')
code(r'''
#@title 1-6　用一格程式清理「全部」資料
t0 = time.time()
df_clean = auto_clean(df_dirty_full)
elapsed = time.time() - t0
print(f"原始 {len(df_dirty_full)} 列 → 清理後 {len(df_clean)} 列（移除 {len(df_dirty_full) - len(df_clean)} 列重複），只花了 {elapsed:.2f} 秒")
check_cleaned(df_clean)
df_clean.to_csv(os.path.join(WORKDIR, "heart_clean_full.csv"), index=False)
display(df_clean.head())
result_full = train_on_table(df_clean)
''')
md(r'''
### 先打個問號 ❓

剛剛顯示的準確率，是模型在**它自己學過的資料**上算出來的。
**這個數字可信嗎？** 就像學生拿「考古題」當考試成績——這就是單元 2 要談的事。

**🎓 老師備註**：自動清理用「中位數／眾數」補缺值，且是用**全部資料**算的，這本身就是輕微的資訊洩漏；單元 4 會示範正確做法：補值用的數字也要「只用訓練資料算、存起來、推論時重複使用」。
''', teacher=True)
