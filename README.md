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
2. **建立 tag `v1.0`**（notebook 從 `.../v1.0/data/` 讀資料）：
   ```bash
   git tag v1.0 && git push origin v1.0
   ```
   沒有 tag 也不會壞：讀取失敗時會自動改用 notebook 內嵌的備援資料。
3. **務必用學生會用的學校帳號，在真實 Colab 上完整跑一次**，特別是 Google 試算表授權（`gspread`）。若學校帳號限制了 Sheets 授權，把 1-3 的 `MODE` 改成 `"csv"`（下載→編輯→上傳）。
4. 請學生**課前**先開 notebook、跑完 0-1（Drive 授權），40 人同時授權很容易卡住。

## 資料夾
```
notebooks/   學生版、老師版 .ipynb（由 teacher/build 產生，請勿直接手改）
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
