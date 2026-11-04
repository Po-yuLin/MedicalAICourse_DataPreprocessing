# 醫學 AI 課程：資料前處理基礎（兩小時工作坊）

給**零程式背景的醫學生**的 AI 資料前處理入門課。學生不需要寫程式，只要在 Google Colab 執行 cell，並完成標示「👉 你的任務」的動手練習。

資料集是公開的 UCI Heart Disease（Cleveland），303 位病人；模型是用 PyTorch 寫的、只有 **4 個感知器**（輸入 → 3 → 1）的超小神經網路。課程的重點不是模型，而是讓學生親眼看到：**資料怎麼處理，會直接決定 AI 學到什麼、你能不能相信它**。

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Po-yuLin/MedicalAICourse_DataPreprocessing/blob/main/notebooks/AI_preprocessing_student.ipynb)　← 學生版（開啟後請先「檔案 → 在雲端硬碟中儲存副本」）

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Po-yuLin/MedicalAICourse_DataPreprocessing/blob/main/notebooks/AI_preprocessing_teacher.ipynb)　← 老師版（含答案、老師備註、「完美學生」捷徑）

---

## 關於這個專案的製作方式（AI 使用聲明）

**本專案的所有內容，是由 Claude（Anthropic 開發的 AI 助理）實際動手完成的**，包括：

- 兩份 Jupyter notebook（學生版與老師版）、清理結果檢查器、小型神經網路與前處理管線
- 「髒資料」的產生器與答案檔
- notebook 內所有教學示意圖（SVG）
- 教學投影片（`.pptx`）的產生與排版
- 本 README

**作者林伯昱（成大醫院神經部／基因醫學部）只負責提出課程設計與需求，並在多輪對話中給予修改意見與課堂回饋**，沒有自行撰寫程式。投影片的標題頁與自我介紹頁是作者原有的內容。

因此請特別留意：

- 內容由 AI 產生，**可能含有錯誤**；課堂上出現的實驗數字也需要在你的環境重新確認。
- 「台灣相關規範」的部分是教學用的整理，**不是法律意見**（見下方說明）。

---

## 課程結構

| 單元 | 內容 | 學生會看到什麼 |
|---|---|---|
| 0 | 準備環境 | 授權 Google 雲端硬碟，載入資料 |
| 1 | 資料如何才能「餵進去」 | 特徵與標籤、缺值、髒資料讓模型壞掉；手動清理 vs 程式清理；超小神經網路 |
| 2 | 資料怎麼分配 | 隨機種子、訓練／驗證／測試、stratify、K 折／留一法／留一病人法、**資料洩漏** |
| 3 | 前處理如何影響訓練 | 標準化（含一個感知器的連鎖律手算）、one-hot、擴增、資料不平衡 |
| 4 | 推論 | 前處理要和模型一起存、鎖住的測試集、新病人、故意做錯的示範 |
| 4 補充 | 去識別化與隱私 | 影像臉部、DICOM 標頭範例、病歷文字、台灣相關規範 |

教學上的幾個設計：

- **人工注入、可驗證的錯誤**：小資料集（40 位病人）裡有 6 類錯誤（缺值、不可能值、單位不一致、數字欄位混入文字、類別寫法不一、重複列），並保留兩個「看起來誇張但是真的」的極端值當陷阱題。
- **答案感知的檢查器**：會檢查有問題的格子修好了沒、**沒問題的格子有沒有被改壞**、有固定答案的格子答案對不對；缺值與離譜值則只要填「合理」的值。`HINT` 1–4 逐級給提示。
- **誠實呈現結果**：例如擴增在這份小資料上幾乎沒有幫助、one-hot 對整數編碼的差距在雜訊範圍內，notebook 都照實寫，並用 5 折交叉驗證報告「平均 ± 標準差」。

## 資料集與授權

- **資料**：Detrano R. et al. *International application of a new probability algorithm for the diagnosis of coronary artery disease.* Am J Cardiol 1989;64:304–310。UCI Machine Learning Repository，**CC BY 4.0**；本專案使用 ISLR 整理版（James et al.）。重新散布請保留這個出處。
- **`data/` 裡的「錯誤」是人為注入的**，僅供教學；模型僅供學習，**不可用於臨床**。
- **程式碼**採用本 repo 的 **GPL-3.0** 授權。
- **投影片**內含作者的個人照片與自我介紹頁，這些內容的著作權屬於作者，不包含在上述授權之內。

## 資料夾

```
notebooks/   學生版、老師版 .ipynb（由 teacher/build 產生，請不要直接手改）
data/        heart_dirty_small.csv（40 位練習用）、heart_dirty_full.csv（303 位＋重複列）
slides/      教學投影片（.pptx），以及 notebook 會讀取的投影片圖片（intro.png、title.png、qr_github.png）
teacher/     answer_key_*.csv（注入錯誤的完整答案）、heart_clean_original.csv（乾淨原始資料）、
             make_dirty_data.py（重新產生髒資料，固定 seed）、build/（重新產生 notebook 的原始碼）
```

> `teacher/` 內有答案。

## 給想使用這份教材的老師

### 上課前
1. 把整個資料夾放到你自己的 GitHub repo，並把 notebook 內的 `REPO` 與連結換成你的 repo。
2. **建立 tag `v1.0`**（notebook 優先讀 `.../v1.0/data/`，其次讀 `main`，最後才用內嵌的備援資料，並會在輸出中明確標示）：
   ```bash
   git tag v1.0 && git push origin v1.0
   ```
3. **用學生會用的帳號，在真實 Colab 完整跑一次**。若學校的 Google 帳號限制 Sheets 授權，把 notebook 中 1-6 的 `MODE` 改成 `"csv"`（下載 → 編輯 → 上傳）。
4. 請學生**課前**先開好 notebook、跑完第一格（Drive 授權）；多人同時授權很容易卡住。

### 投影與字體
- 建議**把瀏覽器放大到 150%**（`Ctrl` ＋ `+`）；Colab 內的表格會偵測視窗寬度，自動縮小去配合，換了縮放比例後重新執行該格即可。
- 第一格的 `FONT_PX = 16` 是表格與文字輸出的最大字體。
- Markdown 說明文字與程式碼的字體 notebook 無法放大，請用瀏覽器縮放，或到 Colab「工具 → 設定 → 編輯器」調整程式碼字體。

### 投影片與 QR code
- `slides/` 裡的圖片是 notebook 開頭直接從 GitHub 讀取的；要換圖，用同樣的檔名取代即可，不必動 notebook。
- notebook 開頭有一張內嵌的 QR code 卡片，指向本 repo 頁面。

### 修改課程內容
編輯 `teacher/build/cells_main.py`、`cells_main2.py`（cell 內容）、`core_*.py`（工具函式）或 `core_figs.py`（示意圖），再執行：
```bash
python teacher/build/build_nb.py
```
重新產生兩份 notebook。

## 環境與已知限制

- 設計給 **Google Colab**（PyTorch、scikit-learn、pandas、matplotlib、scikit-image 皆為 Colab 預設環境）。本機測試使用 CPU 版 PyTorch 2.14、scikit-learn 1.9、pandas 3.1；Colab 的版本不同，數字可能略有出入。
- 小網路與小資料集下，實驗結果會有隨機波動；課程文字與投影片中的數字是作者環境下的實測值，請以現場為準。
- 圖表標籤、說明文字為繁體中文；notebook 內的 matplotlib 圖為英文標籤（Colab 預設字型沒有中文）。

## 關於「台灣相關規範」的說明

「單元 4 補充」中整理了個人資料保護法第 6 條、人體研究法第 5 條、衛福部關於免審與免取得同意的公告與函釋、「去連結」與「去識別化」的概念。這些是**教學用的重點整理**（資料查詢日期：2026-11-04），**不是法律意見**，也可能已隨法規修正而過時。實際做研究時，請以[全國法規資料庫](https://law.moj.gov.tw/)的現行條文，以及你所屬醫院 **IRB** 的規定為準。

主要資料來源：個人資料保護委員會籌備處（個資法第 6 條）、衛福部公告（得免倫理審查委員會審查之人體研究案件範圍；得免取得研究對象同意之人體研究案件範圍）、全國法規資料庫（人體研究法）。

## 引用與回饋

如果你使用或改作這份教材，請保留資料集出處（見上），並建議標註本專案由 Claude 協助製作。發現錯誤或有改進建議，歡迎開 Issue。
