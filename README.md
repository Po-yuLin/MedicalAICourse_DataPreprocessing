# 醫學 AI 課程：資料前處理基礎（兩小時工作坊）

給**零程式背景的醫學生**。學生不寫程式，只執行 cell 並完成「👉 你的任務」。
資料集：UCI Heart Disease（Cleveland），303 位病人、13 個特徵。模型：PyTorch，**4 個感知器**（3 → 1）的超小神經網路。

## 開啟（學生）
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Po-yuLin/MedicalAICourse_DataPreprocessing/blob/main/notebooks/AI_preprocessing_student.ipynb)

點開後，請先 **檔案 → 在雲端硬碟中儲存副本**，再從副本開始執行。

老師版（含答案與老師備註、「完美學生」捷徑）：
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Po-yuLin/MedicalAICourse_DataPreprocessing/blob/main/notebooks/AI_preprocessing_teacher.ipynb)

## 課程結構
| 單元 | 內容 | 重點 |
|---|---|---|
| 1 | 資料如何才能「餵進去」 | 手動清理 40 列（Google 試算表）→ 檢查器 → 程式自動清理全部 |
| 2 | 資料怎麼分配 | 隨機切分的波動、train/val/test、K-fold、**病人層級洩漏** |
| 3 | 前處理如何影響訓練 | 標準化、one-hot、特徵選擇洩漏（加碼）、擴增、不平衡 |
| 4 | 推論 | 存下前處理＋模型、鎖住的測試集、新病人、故意做錯 |

## 上課前（老師）
1. 把整個資料夾 push 到本 repo（`main` 分支）。
2. **建立 tag `v1.0`**（notebook 優先從 `.../v1.0/data/` 讀資料）：
   ```bash
   git tag v1.0 && git push origin v1.0
   ```
   讀取順序：先試 `v1.0` tag → 再試 `main` → 最後才用 notebook 內嵌的備援資料（會在輸出明確標示「內嵌備援」）。所以沒有 tag 也不會壞。
3. **務必用學生會用的學校帳號，在真實 Colab 上完整跑一次**，特別是 Google 試算表授權（`gspread`）。若學校帳號限制了 Sheets 授權，把 1-3 的 `MODE` 改成 `"csv"`（下載→編輯→上傳）。
4. 請學生**課前**先開 notebook、跑完 0-1（Drive 授權），40 人同時授權很容易卡住。

## 投影與字體大小
- 建議**把瀏覽器放大到 150%**（`Ctrl` ＋ `+`）來放大所有文字；Colab 的「工具 → 設定 → 編輯器」可放大程式碼字體。
- **表格會自動縮小去配合視窗寬度**（Colab 內會偵測目前的瀏覽器寬度），所以放大到 150% 也不會超出螢幕。換了縮放比例後，重新執行該格即可。
- notebook 第一格的 `FONT_PX = 16` 是表格與文字輸出的**最大**字體；想更大就調高。
- 圖表是 SVG（跟著視窗寬度縮放）；Markdown 裡的表格是 17px 的 HTML 表格（想改大小：`teacher/build/build_nb.py` 的 `MD_TABLE_PX`）。

## 投影片與圖
- `slides/qr_github.png` 是指向本 repo 的 QR code（可以直接貼進 PPT）；notebook 開頭自我介紹之後也有一張 QR code 卡片（內嵌，不需要網路）。
- `slides/intro.png`（自我介紹）、`slides/title.png`（標題頁）是從 `人工智慧的資料前處理基礎.pptx` 匯出的圖片；notebook 開頭的「投影片｜自我介紹」直接讀 GitHub 上的這張圖。**要換圖：把新圖用同樣的檔名放進 `slides/` 即可，不用動 notebook。**
- 想再加投影片：把圖片放進 `slides/`，在 `teacher/build/cells_main*.py` 加一行 `slide_cell('檔名.png', '標題')`，重新建置。
- 其他教學圖（特徵與標籤、缺值、神經網路、洩漏三種、標準化、K 折／LOO／LOPO、不平衡、隱私）是用程式畫的 SVG（`teacher/build/core_figs.py`），已預先放好輸出，打開 notebook 就看得到，不用先執行。

## 檢查器怎麼判斷學生清得對不對
檢查器內嵌了 40 位病人的「標準答案」與各欄位的「合理範圍」（p1–p99），所以會分辨三種格子：
1. **本來就正常的格子**：必須維持原樣，改掉就報「不該改的被改了」（包含兩個真實極端值：P153 的 chol 564、P092 的 oldpeak 6.2）。
2. **有固定答案的格子**（拼字、單位換算、去掉 mmHg）：必須與答案一致（膽固醇單位換算容許 ±2 mg/dL）。
3. **缺值與不可能的值**（沒有唯一答案）：填任何落在合理範圍內的值都算對。
刪掉**本來就有問題**的病人是允許的；刪掉**本來完全正常**的病人會被抓到。`HINT` 1–4 逐級給提示，4 直接給答案。

## 資料夾
```
notebooks/   學生版、老師版 .ipynb（由 teacher/build 產生，請勿直接手改）
slides/      投影片圖片（自我介紹、標題）
data/        heart_dirty_small.csv（40 位練習用）、heart_dirty_full.csv（303 位＋重複列）
teacher/     answer_key_*.csv（注入錯誤的完整答案）、heart_clean_original.csv（乾淨原始資料）、
             make_dirty_data.py（重新產生髒資料；固定 seed）、build/（重新產生 notebook）
```
> `teacher/` 內有答案。若擔心學生翻到，可把它移到私有 repo。

## 資料來源與授權
- 資料：Detrano R. et al. *International application of a new probability algorithm for the diagnosis of coronary artery disease.* Am J Cardiol 1989;64:304–310。UCI Machine Learning Repository，**CC BY 4.0**；本專案使用 ISLR 整理版（James et al.）。
- `data/` 中的「錯誤」是**人為注入**的，僅供教學；模型不可用於臨床。
- 程式碼採本 repo 的 GPL-3.0 授權；資料請依 CC BY 4.0 保留上述出處。

## 修改課程內容
編輯 `teacher/build/cells_main.py`、`cells_main2.py`（cell 內容）或 `core_*.py`（工具函式），執行 `python teacher/build/build_nb.py` 重新產生兩份 notebook。
