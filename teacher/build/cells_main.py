# -*- coding: utf-8 -*-
# 每個 cell 是 (type, source, flags)。flags: form=收合的程式格、teacher=只在老師版出現
CELLS = []
def fig(name, title): CELLS.append(("code", f"#@title 圖｜{title}\nshow_fig('{name}')", dict(form=True, teacher=False, fig=name)))
def slide_cell(name, title): CELLS.append(("code", f"#@title 投影片｜{title}\nfrom IPython.display import HTML, display\ndisplay(HTML('<img src=\"https://raw.githubusercontent.com/Po-yuLin/MedicalAICourse_DataPreprocessing/main/slides/{name}\" style=\"width:100%;max-width:1100px;height:auto\">'))", dict(form=True, teacher=False, slide=name)))
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

slide_cell('intro.png', '自我介紹')
CELLS.append(("code", "__QR_CELL__", dict(form=True, teacher=False, qr=True)))

# ============================== 單元 0 ==============================
md('## 單元 0｜準備環境')
code(r'''
#@title 0-1　準備環境（會跳出 Google Drive 授權視窗，請按「允許」）
FONT_PX = 16       # 👈 表格與文字輸出的字體大小。建議：瀏覽器放大到 150%，這裡維持 16；表格太寬時會自動縮小配合視窗
import os, sys, io, json, time, base64, zlib, warnings, urllib.request
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")

REPO = "Po-yuLin/MedicalAICourse_DataPreprocessing"      # 本課程的 GitHub repo
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
#@title 0-2　工具箱（檢查器、顯示工具、小型神經網路、前處理管線；不用展開）
import torch, torch.nn as nn
import matplotlib.pyplot as plt
torch.set_num_threads(2)
__CORE_CLEAN__
__CORE_VIEW__
__CORE_FIGS__
__CORE_ML__
__CORE_PREP__
print("✅ 工具箱就緒（PyTorch", torch.__version__ + "）")
''', form=True)
code(r'''
#@title 0-3　載入資料（先從 GitHub 讀；讀不到就改用內嵌的備援資料）
_EMBEDDED = __EMBEDDED__      # 備援資料（壓縮後的文字），網路或 GitHub 出問題時才會用到
_TRUTH = __TRUTH__            # 檢查清理結果時用的「標準答案」（只有檢查器會用到）
BANDS = __BANDS__             # 各欄位「合理範圍」（缺值要補的值，落在這裡面都算合理）

BASE_URLS = [f"https://raw.githubusercontent.com/{REPO}/v1.0/data/",      # 先試固定版本（tag）
             f"https://raw.githubusercontent.com/{REPO}/main/data/"]       # 再試 main 分支

def load_csv(name):
    # 全部以文字讀入，避免 pandas 偷偷幫你改資料（例如把 NA 變成空值）
    text, src = None, None
    for base in BASE_URLS:
        try:
            with urllib.request.urlopen(base + name, timeout=10) as r:
                text, src = r.read().decode("utf-8"), "GitHub"
            break
        except Exception:
            continue
    if text is None:
        text, src = zlib.decompress(base64.b64decode(_EMBEDDED[name])).decode("utf-8"), "內嵌備援（GitHub 讀不到）"
    print(f"  {name}：從「{src}」載入")
    return pd.read_csv(io.StringIO(text), dtype=str, keep_default_na=False)

df_dirty_small = load_csv("heart_dirty_small.csv")    # 40 位病人：給你手動清理
df_dirty_full  = load_csv("heart_dirty_full.csv")     # 全部病人：等一下讓程式自動清理
df_truth_small = pd.read_csv(io.StringIO(zlib.decompress(base64.b64decode(_TRUTH)).decode("utf-8")), dtype=str, keep_default_na=False)

print(f"\n練習用小資料：40 位病人（其中 2 位重複出現，所以共 {len(df_dirty_small)} 列）；完整資料：{len(df_dirty_full)} 列")
print("👀 這就是等一下要清理的資料（␣＝多餘的空格，⟨空白⟩＝空的格子）：")
show(df_dirty_small)
''', form=True)

# ============================== 單元 1 ==============================
md(r'''
---
## 單元 1｜資料如何才能「餵進去」

AI 模型只吃**數字**。先認識兩個名詞：
''')
fig('features_labels', '什麼是特徵與標籤')
md(r'''
**特徵（feature）**是我們「給模型看」的資料（一欄一個）；**標籤（label）**是我們希望它「預測」的答案。這裡的標籤是 `disease`（有沒有冠心病）。

那麼，**為什麼電腦會拒絕吃髒資料？**
''')
fig('why_dirty', '為什麼電腦拒絕吃髒資料')
md(r'''
除了「轉不成數字」，還有一種最常見的資料問題：**缺值**。遇到缺值，第一個問題不是「怎麼補」，而是：**為什麼缺？**
''')
fig('missing', '為什麼缺？怎麼處理？')
md(r'''
**處理缺值的三個概念**：**刪除**（拿掉整列或整欄）、**填補**（用中位數、眾數等補上）、**標記**（多加一欄「是否缺值」）。
今天的練習會用到「刪除」與「填補」。

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

### 資料要怎麼變成數字？（等一下的程式就是照這張表做）

| 欄位 | 轉換規則 | 例子 |
|---|---|---|
| 數字欄位（`age`、`chol`…） | 直接當數字讀；**讀不了 → NaN**（Not a Number，「不是數字」） | `"54"` → 54　`"54y"` → **NaN**　`"?"` → **NaN** |
| `sex` | 剛好等於 `male` → 1；**其他一律 → 0** | `"male"` → 1　`"female"` → 0　`"Male"` → **0（錯了！）** |
| `disease`（標籤） | 剛好等於 `yes` → 1；**其他一律 → 0** | `"yes"` → 1　`"no"` → 0　`"Y"` → **0（錯了！）** |

`chest_pain`、`thal` 是文字類別，單元 3 才會處理；單元 1、2 的模型先不使用它們。
''')
code(r'''
#@title 1-1　👀 大家來找碴：這幾位病人的資料，哪裡怪怪的？
DEMO_IDS = __DEMO_IDS__
demo_rows = df_dirty_small[df_dirty_small["patient_id"].isin(DEMO_IDS)]
print(f"從 {len(df_dirty_small)} 列裡挑出 {len(demo_rows)} 列。給你 1 分鐘，找出『哪一格』怪怪的。")
show(demo_rows)
''')
code(r'''
#@title 1-2　🔍 公布答案：哪些格子有問題？
show_issues(df_dirty_small, rows=list(demo_rows.index))
''')
code(r'''
#@title 1-3　把資料「硬轉」成數字：到底發生什麼事？
d_bad = df_dirty_small.copy()

# 規則 1：數字欄位 → 直接當數字讀；讀不了的格子，變成 NaN
for c in ALL_NUMERIC:
    d_bad[c] = pd.to_numeric(d_bad[c], errors="coerce")
# 規則 2：sex → 剛好等於 "male" 的變成 1，其他『全部』變成 0
d_bad["male"] = (df_dirty_small["sex"] == "male").astype(int)
# 規則 3：標籤 disease → 剛好等於 "yes" 的變成 1，其他『全部』變成 0
d_bad["label"] = (df_dirty_small["disease"] == "yes").astype(int)

rule_rows = []
for c in ALL_NUMERIC:
    rule_rows.append({"欄位": c, "轉換規則": "當數字讀", "轉換後 NaN 的個數": int(d_bad[c].isna().sum())})
rule_rows.append({"欄位": "sex", "轉換規則": "male → 1，其他 → 0", "轉換後 NaN 的個數": 0})
rule_rows.append({"欄位": "disease", "轉換規則": "yes → 1，其他 → 0", "轉換後 NaN 的個數": 0})
rule_rows.append({"欄位": "chest_pain, thal", "轉換規則": "單元 1 的模型不用", "轉換後 NaN 的個數": "—"})
show(pd.DataFrame(rule_rows))
print("👉 sex 和 disease 的 NaN 是 0，不是因為它們沒問題，而是規則 2、3 把『不認得的寫法』全部當成 0，完全不會報錯。")

X_bad = add_male(d_bad)[FEATURES_V1].values.astype("float32")
y_bad = d_bad["label"].values.astype("float32")
_, h_bad = train_mlp(X_bad, y_bad, epochs=10)
print("\n帶著 NaN 去訓練 10 個 epoch，每一步的 loss（模型的『錯誤程度』）：")
print([round(v, 3) for v in h_bad["train"]])
print("👉 loss 變成 nan：模型一碰到 NaN，整個訓練就壞掉了，而且不會告訴你為什麼。")
''')
code(r'''
#@title 1-4　👀 再來找碴：sex 與 disease 轉成數字之後，對嗎？
CONV_IDS = __CONV_IDS__
conv = df_dirty_small[df_dirty_small["patient_id"].isin(CONV_IDS)][["patient_id", "sex", "disease"]].copy()
conv["male"] = (conv["sex"] == "male").astype(int)            # 規則 2
conv["label"] = (conv["disease"] == "yes").astype(int)        # 規則 3
conv = conv[["patient_id", "sex", "male", "disease", "label"]]
conv.columns = ["patient_id", "sex（轉換前）", "male（轉換後）", "disease（轉換前）", "label（轉換後）"]
print("每一列：左邊是原本的寫法，右邊是程式轉出來的數字（male：1＝男、0＝女；label：1＝有病、0＝沒病）。")
print("轉換『有成功』（沒有 NaN），但哪些轉換結果是錯的？")
show(conv)
''')
code(r'''
#@title 1-5　🔍 公布答案：哪些轉換「默默」出錯？
true_male  = conv["sex（轉換前）"].str.strip().str.lower().isin(["m", "male"]).astype(int)
true_label = conv["disease（轉換前）"].str.strip().str.lower().isin(["yes", "y", "1"]).astype(int)
bad_sex   = conv["male（轉換後）"] != true_male
bad_label = conv["label（轉換後）"] != true_label

def paint(_):
    css = pd.DataFrame("", index=conv.index, columns=conv.columns)
    css.loc[bad_sex,   ["sex（轉換前）", "male（轉換後）"]] = "background-color:#ff8a80 !important;color:#000;font-weight:bold"
    css.loc[bad_label, ["disease（轉換前）", "label（轉換後）"]] = "background-color:#ff8a80 !important;color:#000;font-weight:bold"
    return css
show(conv, styler=conv.style.apply(paint, axis=None))

all_true_male  = df_dirty_small["sex"].str.strip().str.lower().isin(["m", "male"])
all_true_label = df_dirty_small["disease"].str.strip().str.lower().isin(["yes", "y", "1"])
n_sex_bad   = int((all_true_male != (df_dirty_small["sex"] == "male")).sum())
n_label_bad = int((all_true_label != (df_dirty_small["disease"] == "yes")).sum())
print(f"全部 {len(df_dirty_small)} 列裡：{n_sex_bad} 位的性別被默默轉錯；{n_label_bad} 位『有病』的病人被默默當成『沒病』。")
print("👉 有些列是『運氣好剛好對』（例如 'female ' 多了空格，轉成 0，剛好還是女性）。")
print("   標籤被轉錯，等於你教模型的答案是錯的。這種錯誤沒有任何警告，最危險。")
''')
code(r'''
#@title 1-6　建立練習用的 Google 試算表（第一次會跳出授權視窗）
MODE = "sheets"          # "sheets"＝Google 試算表（建議）；"csv"＝下載 CSV 編輯後再上傳（備案，見下方說明）
SHEET_TITLE = "heart_dirty_practice"
CLEANED_CSV = os.path.join(WORKDIR, "heart_cleaned_small.csv")      # csv 模式：請把清理好的檔案用這個名字存到 Drive 資料夾

df_dirty_small.to_csv(os.path.join(WORKDIR, "heart_dirty_small.csv"), index=False)   # 備案用：Drive 資料夾裡也放一份

def _to_cell(v):
    # 數字寫成數字，其他維持文字（試算表裡看起來才像真的資料）
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
| 缺值 | 空白、`?`、`NA`、`unknown`、`N/A`… | 補一個**合理的值**（例如該欄中位數），或刪掉整列 |
| 不可能的數值 | 超出上面表格的合理範圍 | 能判斷是打錯就改；不能判斷就當缺值，補合理的值 |
| 單位不一致 | 膽固醇應該是 100–600；出現 5.46 可能是 mmol/L | mg/dL ＝ mmol/L × 38.67 |
| 數字欄位混入文字 | `110 mmHg`、`143bpm` | 只留下數字 |
| 類別寫法不一 | `M`/`Male`、`ASYMPTOMATIC`、`reversable`、`1`/`Y`… | 改成上面表格的**標準寫法**（小寫、沒有多餘空格） |
| 重複列 | 同一個 `patient_id` 出現兩次 | 只留一列 |

**小技巧**：`Ctrl+H`（尋找與取代）、「資料 → 資料清理 → 移除重複項目」、點欄位標題排序，會讓你快很多。
**規則**：不要改標題列（第 1 列）的欄位名稱。

⚠️ **檢查器會看三件事**：① 有問題的格子修好了沒；② **沒有問題的格子被你改掉了沒**（改壞正常資料是最糟的）；③ 有『固定答案』的格子（拼字、單位換算、去掉文字）答案要對，缺值與離譜的值則只要填『合理』的值。
**想一想**：遇到「看起來很誇張」的數值，你要怎麼判斷它是**打錯**還是**真的**？刪掉整位病人，有沒有代價？

修完之後，執行下面兩格：先檢查，再訓練。檢查會告訴你還剩什麼問題；卡住就把 `HINT` 調大一點。
''')
md(r'''
**🎓 老師備註（單元 1）**
- 1-1 挑的 12 列涵蓋所有錯誤類型（缺值、不可能值、單位、文字混入、類別寫法、重複）。給 1 分鐘讓學生喊，再用 1-2 公布。
- 1-3：先回答「為什麼 sex 沒有 NaN」——規則是 `== "male"`，不認得就當 0，不會報錯。1-4／1-5 讓學生親眼看到「轉換成功」≠「轉換正確」；`'female '` 是運氣好剛好對。
- 40 列裡共注入：缺值 5＋原本就缺 2、不可能值 4、單位錯誤 3、文字混入 4、類別寫法 11、重複 2。完整答案：`teacher/answer_key_small.csv`。
- **陷阱題（刻意保留）**：`P153` 的 chol = 564、`P092` 的 oldpeak = 6.2，都是**真實**極端值，不是錯誤。若學生改掉，檢查器會報「不該改的被改了」，正好討論「由有臨床判斷力的人決定」。
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
#@title 1-7　讀回你清理的結果並檢查（可以重複執行，直到全部通過）
HINT = 1     # 1：只告訴你還有幾類問題　2：再告訴你是哪幾個欄位　3：連 patient_id 都告訴你　4：連答案都告訴你

def read_student_table():
    if MODE == "sheets":
        rows = sh.sheet1.get_all_values()
        d = pd.DataFrame(rows[1:], columns=rows[0])
    else:
        d = pd.read_csv(CLEANED_CSV, dtype=str, keep_default_na=False)
    return d[d.apply(lambda r: "".join(map(str, r)).strip() != "", axis=1)].reset_index(drop=True)   # 去掉全空白的列

df_student = read_student_table()
print(f"讀到 {len(df_student)} 列（原本 {len(df_dirty_small)} 列）")
all_ok = check_cleaned(df_student, hint=HINT, dirty=df_dirty_small, truth=df_truth_small, bands=BANDS)
''')
fig('tiny_net', '我們的超小神經網路長什麼樣')
code(r'''
#@title 1-8　用你清好的資料，訓練一個超小的神經網路（輸入 → 3 個感知器 → 1 個感知器，共 4 個）
def train_on_table(df, epochs=300):
    try:
        X, y = to_xy(df)
    except Exception as e:
        print("❌ 資料還沒辦法餵進模型。Python 的錯誤訊息是：\n  ", e)
        print("   → 回到 1-7 看檢查結果；通常是數字欄位裡還有文字或空白。")
        return None
    if np.isnan(X).any():
        print("⚠️ 資料裡還有 NaN，loss 會變成 nan。")
    model, hist = train_mlp(X, y, epochs=epochs)
    plt.figure(figsize=(8, 4.5)); plt.plot(hist["train"])
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

下面把規則一條一條寫出來，每一步都列出「改了哪些格子」。
''')
code(r'''
#@title 1-9　程式清理（一）：去空格、統一「缺值」寫法、移除重複
t_start = time.time()
d_full = df_dirty_full.copy()                           # 從原始資料重新開始（可以重複執行）

# 規則 1：每個格子前後的空格都去掉（"female " → "female"）
before = d_full.copy()
d_full = d_full.apply(lambda s: s.str.strip())
show_changes(before, d_full, "規則 1：去掉前後空格")

# 規則 2：這些寫法都代表「缺值」→ 統一變成 NaN
MISSING = ["", "?", "NA", "N/A", "na", "n/a", "null", "unknown", "-", "--"]
before = d_full.copy()
d_full = d_full.replace({t: np.nan for t in MISSING})
show_changes(before, d_full, "規則 2：缺值的各種寫法 → NaN")

# 規則 3：同一個 patient_id 只留第一列
dups = d_full[d_full.duplicated("patient_id", keep=False)].sort_values("patient_id")
print(f"▶ 規則 3：重複的病人（共 {len(dups)} 列，只保留每位的第一列）")
show(dups.head(6))
d_full = d_full.drop_duplicates(subset="patient_id", keep="first")
print(f"目前 {len(d_full)} 列")
''')
code(r'''
#@title 1-10　程式清理（二）：抽出數字、單位換算、不可能的數值
# 規則 4：數字欄位只留下「數字」部分（"120 mmHg" → 120）
before = d_full.copy()
for c in ALL_NUMERIC:
    d_full[c] = pd.to_numeric(d_full[c].str.extract(r"(-?\d+\.?\d*)")[0], errors="coerce")
show_changes(before, d_full, "規則 4：抽出數字（去掉 mmHg、bpm…）")

# 規則 5：膽固醇小於 30 的，其實是 mmol/L → 乘 38.67 換成 mg/dL
before = d_full.copy()
d_full.loc[d_full["chol"] < 30, "chol"] *= 38.67
show_changes(before, d_full, "規則 5：膽固醇單位換算", cols=["chol"])

# 規則 6：超出合理範圍的 → 當成缺值（範圍就是上面資料字典那張表）
print("合理範圍：", {c: RANGES[c] for c in ["age", "resting_bp", "chol", "max_hr", "oldpeak"]})
before = d_full.copy()
for c, (lo, hi) in RANGES.items():
    d_full.loc[(d_full[c] < lo) | (d_full[c] > hi), c] = np.nan
show_changes(before, d_full, "規則 6：不可能的數值 → NaN")
''')
code(r'''
#@title 1-11　程式清理（三）：把文字類別的寫法統一
# 規則 7：先轉小寫，再用「對照表」把各種寫法對到標準寫法
FIX = {
    "sex": {"m": "male", "male": "male", "f": "female", "female": "female"},
    "chest_pain": {"typical": "typical", "typical angina": "typical",
                   "nontypical": "nontypical", "non-typical": "nontypical", "atypical": "nontypical",
                   "nonanginal": "nonanginal", "non-anginal": "nonanginal", "non anginal": "nonanginal",
                   "asymptomatic": "asymptomatic"},
    "thal": {"normal": "normal", "fixed": "fixed", "fixed defect": "fixed",
             "reversible": "reversible", "reversable": "reversible", "reversible defect": "reversible"},
    "disease": {"yes": "yes", "y": "yes", "1": "yes", "no": "no", "n": "no", "0": "no"},
}
print("對照表（例）：", {k: FIX["sex"][k] for k in ["m", "male", "f"]}, "…")
before = d_full.copy()
for c, m in FIX.items():
    d_full[c] = d_full[c].str.lower().map(m)
show_changes(before, d_full, "規則 7：類別寫法統一（大小寫、縮寫、拼字）", cols=list(FIX))
print("對照表裡找不到的寫法會變成 NaN；目前這四欄的 NaN 個數：", d_full[list(FIX)].isna().sum().to_dict())
''')
code(r'''
#@title 1-12　程式清理（四）：補上剩下的缺值，存檔，檢查，訓練
# 規則 8：剩下的缺值：數字欄位用「中位數」補；類別欄位用「眾數」（最常出現的）補
fill_values = {c: d_full[c].median() for c in ALL_NUMERIC}
fill_values.update({c: d_full[c].mode()[0] for c in CAT_COLS})
print("要補的值（每一欄的中位數／眾數）：")
show(pd.DataFrame({"欄位": list(fill_values), "補上的值": list(fill_values.values())}))
before = d_full.copy()
for c in ALL_NUMERIC + list(CAT_COLS):
    d_full[c] = d_full[c].fillna(fill_values[c])
show_changes(before, d_full, "規則 8：補缺值")

d_full["chol"] = d_full["chol"].round(0)
d_full[ALL_NUMERIC] = d_full[ALL_NUMERIC].astype(float)
df_clean = d_full.reset_index(drop=True)
elapsed = time.time() - t_start
print(f"\n原始 {len(df_dirty_full)} 列 → 清理後 {len(df_clean)} 列（移除 {len(df_dirty_full) - len(df_clean)} 列重複），8 條規則全部跑完只花了 {elapsed:.2f} 秒")
check_cleaned(df_clean)
df_clean.to_csv(os.path.join(WORKDIR, "heart_clean_full.csv"), index=False)
show(df_clean.head())
result_full = train_on_table(df_clean)
''')
code(r'''
#@title [老師] 確認：上面八條規則與工具箱的 auto_clean 結果完全一致（學生版沒有這一格）
assert df_clean.equals(auto_clean(df_dirty_full)), "1-9～1-12 與工具箱 auto_clean 結果不同！"
print("✅ 一致")
''', teacher=True)
md(r'''
### 先打個問號 ❓

剛剛顯示的準確率，是模型在**它自己學過的資料**上算出來的。
**這個數字可信嗎？** 就像學生拿「考古題」當考試成績——這就是單元 2 要談的事。

**🎓 老師備註**：自動清理用「中位數／眾數」補缺值，且是用**全部資料**算的，這本身就是輕微的資訊洩漏；單元 4 會示範正確做法：補值用的數字也要「只用訓練資料算、存起來、推論時重複使用」。
''', teacher=True)
