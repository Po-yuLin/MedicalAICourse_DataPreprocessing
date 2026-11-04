# ===== 教學用圖表（SVG，用瀏覽器字型顯示，所以中文不會變方塊）=====
_FONT = "'Noto Sans TC','Microsoft JhengHei','PingFang TC','Heiti TC','Noto Sans CJK TC',sans-serif"
_C = {"blue": ("#dbeafe", "#2563eb"), "red": ("#fee2e2", "#dc2626"), "green": ("#dcfce7", "#16a34a"),
      "yellow": ("#fef9c3", "#ca8a04"), "gray": ("#f3f4f6", "#6b7280"), "purple": ("#ede9fe", "#7c3aed"),
      "orange": ("#ffedd5", "#ea580c"), "white": ("#ffffff", "#9ca3af")}


class _Svg:
    def __init__(self, w, h):
        self.w, self.h, self.p = w, h, [f'<rect width="{w}" height="{h}" fill="#ffffff"/>']

    def rect(self, x, y, w, h, c="gray", r=10, sw=2, dash=None, fill=None, stroke=None):
        f, s = _C[c]
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.p.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{fill or f}" stroke="{stroke or s}" stroke-width="{sw}"{d}/>')

    def text(self, x, y, s, size=22, anchor="middle", weight="normal", fill="#111827", lh=1.35):
        lines = str(s).split("\n")
        tsp = "".join(f'<tspan x="{x}" dy="{0 if i == 0 else size * lh}">{_esc(t)}</tspan>' for i, t in enumerate(lines))
        self.p.append(f'<text x="{x}" y="{y}" font-size="{size}" text-anchor="{anchor}" font-weight="{weight}" fill="{fill}">{tsp}</text>')

    def line(self, x1, y1, x2, y2, color="#374151", sw=3, dash=None, arrow=False):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        m = ' marker-end="url(#arr)"' if arrow else ""
        self.p.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{sw}"{d}{m}/>')

    def circle(self, x, y, r, c="gray", sw=2, fill=None):
        f, s = _C[c]
        self.p.append(f'<circle cx="{x}" cy="{y}" r="{r}" fill="{fill or f}" stroke="{s}" stroke-width="{sw}"/>')

    def html(self):
        return (f'<div style="max-width:100%;overflow-x:auto"><svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {self.w} {self.h}" '
                f'style="width:100%;max-width:{self.w}px;height:auto;font-family:{_FONT}">'
                '<defs><marker id="arr" markerUnits="userSpaceOnUse" markerWidth="14" markerHeight="14" refX="11" refY="7" orient="auto">'
                '<path d="M0,0 L14,7 L0,14 z" fill="#374151"/></marker></defs>' + "".join(self.p) + "</svg></div>")


def _esc(t):
    return str(t).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _box(s, x, y, w, h, text, c="blue", size=21, weight="normal", r=10, sw=2, dash=None):
    s.rect(x, y, w, h, c, r, sw, dash)
    n = str(text).count("\n") + 1
    s.text(x + w / 2, y + h / 2 - (n - 1) * size * 0.675 + size * 0.35, text, size, weight=weight)


def fig_features_labels():
    s = _Svg(1000, 520)
    s.text(500, 42, "一位病人＝一列　一個欄位＝一個特徵　想預測的答案＝標籤", 27, weight="bold")
    cols = ["age", "sex", "chest_pain", "resting_bp", "chol", "max_hr", "disease"]
    rows = [["63", "male", "typical", "145", "233", "150", "no"], ["67", "male", "nonanginal", "160", "286", "108", "yes"],
            ["41", "female", "nontypical", "130", "204", "172", "no"]]
    x0, y0, cw, rh = 60, 175, 125, 52
    s.text(x0 + 3 * cw, 112, "特徵 features（X）", 26, weight="bold", fill="#1d4ed8")
    s.text(x0 + 3 * cw, 142, "我們「給模型看」的資料", 21, fill="#1d4ed8")
    s.line(x0, 160, x0 + 6 * cw - 6, 160, "#2563eb", 4)
    for j, c in enumerate(cols):
        lab = j == 6
        s.rect(x0 + j * cw, y0, cw - 6, rh, "red" if lab else "blue", 6)
        s.text(x0 + j * cw + (cw - 6) / 2, y0 + 34, c, 21, weight="bold")
        for i, r in enumerate(rows):
            s.rect(x0 + j * cw, y0 + (i + 1) * (rh + 4) + 4, cw - 6, rh, "red" if lab else "white", 6, 1.5)
            s.text(x0 + j * cw + (cw - 6) / 2, y0 + (i + 1) * (rh + 4) + 4 + 35, r[j], 21)
    lx = x0 + 6 * cw + (cw - 6) / 2
    s.line(lx, 470, lx, 440, "#dc2626", 4, arrow=True)
    s.text(975, 498, "標籤 label（y）：想預測的答案", 22, anchor="end", weight="bold", fill="#b91c1c")
    s.text(300, 465, "模型的工作：看 X → 猜 y\n（像實習醫師看過很多病歷，學著判斷診斷）", 22, fill="#374151")
    return s.html()


def fig_why_dirty():
    s = _Svg(1000, 500)
    s.text(500, 40, "為什麼電腦「拒絕」吃髒資料？", 28, weight="bold")
    _box(s, 40, 75, 250, 90, "原始資料\n（各系統匯出）", "gray", 22)
    _box(s, 375, 75, 250, 90, "轉成數字", "yellow", 24, "bold")
    _box(s, 710, 75, 250, 90, "模型\n（只吃數字）", "blue", 22)
    s.line(290, 120, 370, 120, arrow=True); s.line(625, 120, 705, 120, arrow=True)
    cards = [("red", "文字混在數字裡", '"54y"　"110 mmHg"\n→ 轉不成數字 → NaN'),
             ("red", "空白、?、NA、unknown", '"" "?" "NA"\n→ 全部變成 NaN'),
             ("orange", "轉得成，但是錯！", "膽固醇 5.46（mmol/L）\n被當成 5.46 mg/dL\n\"Male\" 被當成 0\n→ 不會報錯")]
    for i, (c, t, b) in enumerate(cards):
        x = 30 + i * 322
        s.rect(x, 210, 300, 165, c, 12)
        s.text(x + 150, 250, t, 22, weight="bold")
        s.text(x + 150, 292, b, 20)
        s.line(500, 165, x + 150, 208, "#9ca3af", 2, "6 5")
    s.rect(30, 395, 940, 90, "red", 12, 3)
    s.text(500, 432, "NaN 會「傳染」：1 ＋ NaN ＝ NaN　→　loss ＝ NaN　→　整個訓練壞掉", 24, weight="bold", fill="#991b1b")
    s.text(500, 466, "更可怕的是第三種：沒有任何警告，模型卻學到錯的東西", 21, fill="#7f1d1d")
    return s.html()


def fig_missing():
    s = _Svg(1000, 590)
    s.text(500, 40, "為什麼缺？缺值本身，可能就是資訊", 28, weight="bold")
    s.rect(30, 70, 455, 190, "green", 12)
    s.text(257, 108, "① 隨機缺", 25, weight="bold", fill="#166534")
    s.text(257, 150, "例：檢體溶血、系統漏存、單純忘了填\n\n缺的原因跟病情無關\n→ 刪除或填補通常比較安全", 20)
    s.rect(515, 70, 455, 190, "red", 12)
    s.text(742, 108, "② 因為「沒做這項檢查」而缺", 25, weight="bold", fill="#991b1b")
    s.text(742, 150, "例：胸痛病人沒驗心肌酵素（troponin）\n因為醫師覺得不像心肌梗塞\n\n「沒做」本身就帶有臨床資訊\n→ 隨便填補，會把這個訊號抹掉", 20)
    s.text(500, 302, "處理方式（只講概念）", 26, weight="bold")
    items = [("blue", "刪除", "整列或整欄拿掉", "簡單\n但損失資料；如果缺值\n有規律，會造成偏誤"),
             ("yellow", "填補", "中位數、眾數、預測值…", "保留資料\n但填進去的是「猜的」\n（要記得補值用的數字\n也只能從訓練資料算）"),
             ("purple", "標記", "多加一欄「是否缺值」", "讓模型自己學\n「缺」代表什麼\n（特別適合第②種）")]
    for i, (c, t, sub, body) in enumerate(items):
        x = 30 + i * 322
        s.rect(x, 322, 300, 250, c, 12)
        s.text(x + 150, 364, t, 26, weight="bold")
        s.text(x + 150, 396, sub, 19, fill="#374151")
        s.text(x + 150, 440, body, 19)
    return s.html()


def fig_tiny_net():
    s = _Svg(1000, 700)
    s.text(500, 38, "我們的超小神經網路：只有 4 個感知器", 28, weight="bold")
    ins = ["age", "male", "resting_bp", "chol", "fbs", "rest_ecg", "max_hr", "exang", "oldpeak", "slope", "ca"]
    ix, hx, ox = 150, 520, 830
    iy = [95 + i * 46 for i in range(len(ins))]
    hy = [210, 330, 450]
    oy = 330
    for y in iy:
        for h in hy:
            s.line(ix + 18, y, hx - 40, h, "#cbd5e1", 1.2)
    for h in hy:
        s.line(hx + 40, h, ox - 42, oy, "#94a3b8", 2.5)
    for n, y in zip(ins, iy):
        s.circle(ix, y, 16, "gray")
        s.text(ix - 28, y + 7, n, 19, anchor="end")
    for i, h in enumerate(hy):
        s.circle(hx, h, 40, "blue", 3)
        s.text(hx, h + 5, f"感知器 {i+1}", 17, weight="bold")
        s.text(hx, h + 24, "tanh", 15, fill="#1e40af")
    s.circle(ox, oy, 42, "red", 3)
    s.text(ox, oy + 2, "感知器 4", 17, weight="bold")
    s.text(ox, oy + 22, "sigmoid", 15, fill="#991b1b")
    s.line(ox + 44, oy, ox + 120, oy, "#dc2626", 3, arrow=True)
    s.text(ox + 46, oy - 52, "有病的機率\n（0 到 1）", 17, anchor="start", fill="#991b1b")
    s.text(ix, 612, "輸入：11 個特徵", 21, weight="bold")
    s.text(hx, 560, "隱藏層：3 個感知器", 21, weight="bold")
    s.text(ox, 560, "輸出層：1 個", 21, weight="bold")
    s.rect(150, 625, 700, 62, "yellow", 8)
    s.text(500, 652, "每個感知器：Σ（輸入 × 權重）＋ 偏差 → 激活函數", 19)
    s.text(500, 677, "參數：11×3＋3＋3＋1 ＝ 40 個（單元 4 的 22 個輸入則是 73 個）", 17, fill="#78350f")
    return s.html()


def _grid_cell(s, x, y, w, kind):
    s.rect(x, y, w, w, "red" if kind == "T" else "blue", 5, 2)


def fig_leak_patient():
    s = _Svg(1000, 520)
    s.text(500, 38, "洩漏 ①　病人層級：同一位病人的多次檢查，分到訓練與測試兩邊", 25, weight="bold")
    A = ["TXTTT", "XTTXT", "TTXTT", "TTTTX"]
    B = ["TTTTT", "XXXXX", "TTTTT", "TTTTT"]
    for k, (title, pat, x0, cc) in enumerate([("❌ 以「列」切分", A, 60, "red"), ("✅ 以「病人」切分", B, 560, "green")]):
        s.text(x0 + 190, 88, title, 25, weight="bold", fill="#991b1b" if k == 0 else "#166534")
        for i, row in enumerate(pat):
            s.text(x0 - 8, 148 + i * 66 + 8, f"P00{i+1}", 20, anchor="end")
            for j, ch in enumerate(row):
                _grid_cell(s, x0 + 10 + j * 74, 120 + i * 66, 58, "T" if ch == "X" else "B")
                s.text(x0 + 39 + j * 74, 156 + i * 66, f"#{j+1}", 17, fill="#374151")
    s.rect(60, 400, 14, 14, "blue", 3); s.text(84, 413, "訓練集", 19, anchor="start")
    s.rect(180, 400, 14, 14, "red", 3); s.text(204, 413, "測試集", 19, anchor="start")
    s.text(500, 462, "同一位病人的 5 次檢查長得非常像、答案也一樣。\n左邊：模型只要「認得這個人」就能答對測試題 → 成績虛高。", 21, fill="#374151")
    s.text(500, 508, "", 10)
    return s.html()


def fig_leak_preproc():
    s = _Svg(1000, 520)
    s.text(500, 38, "洩漏 ②　前處理：先對「全部」資料標準化，再切分", 26, weight="bold")
    lanes = [("❌ 洩漏", 80, "red", ["全部資料", "用全部資料\n算平均與標準差", "標準化", "切成\n訓練／測試"], 1),
             ("✅ 正確", 290, "green", ["全部資料", "先切成\n訓練／測試", "只用訓練集\n算平均與標準差", "套用到\n訓練與測試"], 2)]
    for title, y, c, names, hi in lanes:
        s.text(30, y - 8, title, 25, weight="bold", fill="#991b1b" if c == "red" else "#166534", anchor="start")
        for i, n in enumerate(names):
            x = 30 + i * 245
            _box(s, x, y + 10, 195, 100, n, c if i == hi else "gray", 20, "bold" if i == hi else "normal", 10, 3 if i == hi else 2)
            if i < 3:
                s.line(x + 197, y + 60, x + 243, y + 60, arrow=True)
    s.text(500, 235, "測試集的資訊已經「滲」進平均與標準差 → 模型偷看了考卷", 22, fill="#991b1b", weight="bold")
    s.text(500, 445, "任何「從資料算出來的東西」：平均、標準差、補缺值用的中位數、挑特徵…\n都只能用訓練集算，再「套用」到驗證、測試、新病人。", 22, fill="#166534")
    return s.html()


def fig_leak_feature():
    s = _Svg(1000, 560)
    s.text(500, 38, "洩漏 ③　特徵：不小心包含了「答案的線索」", 26, weight="bold")
    s.text(250, 85, "例一：影像上的醫院標記", 23, weight="bold")
    for k, (x, lab, c) in enumerate([(40, "A 醫院（重症較多）", "red"), (270, "B 醫院（門診）", "blue")]):
        s.rect(x, 105, 190, 190, c, 8)
        s.circle(x + 95, 205, 62, "white", 2)
        s.rect(x + 8, 112, 64, 28, "yellow", 4, 1.5)
        s.text(x + 40, 133, "ICU" if k == 0 else "OPD", 18, weight="bold")
        s.text(x + 95, 325, lab, 19)
    s.text(250, 372, "病人多的醫院，影像角落的字也特別多", 19, fill="#374151")
    s.line(250, 395, 250, 430, "#374151", 3, arrow=True)
    s.rect(40, 435, 420, 95, "orange", 10)
    s.text(250, 468, "模型學會「看角落的字」\n而不是「看肺部」→ 換家醫院就失靈", 21, weight="bold")
    s.text(745, 85, "例二：特徵發生在診斷「之後」", 23, weight="bold")
    s.line(560, 190, 950, 190, "#374151", 4, arrow=True)
    for x, t in [(600, "症狀出現"), (740, "診斷"), (880, "開始用藥")]:
        s.circle(x, 190, 9, "gray")
        s.text(x, 232, t, 20)
    s.line(670, 130, 670, 250, "#dc2626", 3, "8 6")
    s.text(670, 120, "預測的時間點", 19, fill="#dc2626", weight="bold")
    s.rect(780, 255, 190, 56, "red", 8)
    s.text(875, 291, "特徵：用藥紀錄", 20, weight="bold")
    s.rect(520, 340, 450, 190, "orange", 10)
    s.text(745, 380, "「正在使用硝化甘油」\n早就暗示了心絞痛的診斷\n\n特徵一定要是「預測當下」\n就拿得到的資訊", 21, weight="bold")
    return s.html()


def fig_imbalance():
    s = _Svg(1000, 540)
    s.text(500, 38, "類別不平衡：罕見疾病只占 1%", 28, weight="bold")
    for i in range(100):
        r, c = divmod(i, 10)
        sick = i == 37
        s.circle(70 + c * 34, 100 + r * 34, 12, "red" if sick else "gray", 2, fill="#dc2626" if sick else None)
    s.text(240, 470, "100 位病人裡，只有 1 位有病（紅點）", 20)
    s.rect(430, 100, 250, 270, "yellow", 12)
    s.text(555, 140, "模型只會一招：\n全部猜「沒病」", 22, weight="bold")
    s.text(555, 215, "accuracy＝99％", 24, weight="bold", fill="#166534")
    s.text(555, 268, "recall＝0％", 28, weight="bold", fill="#b91c1c")
    s.text(555, 310, "（有病的那一位，\n一個都沒抓到）", 19, fill="#7f1d1d")
    s.line(682, 235, 722, 235, arrow=True)
    s.rect(730, 80, 250, 400, "green", 12)
    s.text(855, 118, "怎麼辦？", 25, weight="bold")
    s.text(855, 165, "重抽樣\n過採樣：複製少數類別\n欠採樣：減少多數類別", 19)
    s.text(855, 275, "調整權重\nclass weight：漏掉\n少數類別，罰得更重", 19)
    s.text(855, 385, "選對指標\nrecall、precision、AUC\n別只看 accuracy", 19, weight="bold")
    return s.html()


def fig_scale():
    s = _Svg(1000, 520)
    s.text(500, 38, "標準化：把大家拉到「同一把尺」", 28, weight="bold")
    s.text(30, 95, "標準化之前（同一條數線）", 23, weight="bold", anchor="start")
    ax0, ax1 = 130, 940
    sx = lambda v: ax0 + (v - 0) / 130 * (ax1 - ax0)
    s.line(ax0, 235, ax1, 235, "#374151", 2)
    for v in (0, 40, 80, 120):
        s.line(sx(v), 230, sx(v), 241, "#374151", 2); s.text(sx(v), 262, str(v), 18)
    s.rect(sx(40), 120, sx(120) - sx(40), 36, "blue", 6)
    s.text(sx(80), 146, "體重　40–120 kg", 20, weight="bold")
    s.rect(sx(3.5), 170, max(sx(5.5) - sx(3.5), 8), 36, "red", 4)
    s.text(sx(5.5) + 12, 196, "血鉀　3.5–5.5 mmol/L（擠在一小點）", 20, anchor="start", weight="bold")
    s.text(500, 305, "體重的數字比血鉀大 20 倍以上 → 在模型眼中，「體重」的聲音大得多", 21, fill="#991b1b")
    s.text(30, 355, "標準化之後（減平均、除以標準差）", 23, weight="bold", anchor="start")
    bx0, bx1 = 130, 940
    tx = lambda v: bx0 + (v + 3) / 6 * (bx1 - bx0)
    s.line(bx0, 455, bx1, 455, "#374151", 2)
    for v in (-2, -1, 0, 1, 2):
        s.line(tx(v), 450, tx(v), 461, "#374151", 2); s.text(tx(v), 482, str(v), 18)
    s.rect(tx(-2), 385, tx(2) - tx(-2), 28, "blue", 6); s.text(tx(0), 407, "體重", 19, weight="bold")
    s.rect(tx(-2), 420, tx(2) - tx(-2), 28, "red", 6); s.text(tx(0), 442, "血鉀", 19, weight="bold")
    s.text(500, 510, "兩個變數都落在差不多的範圍 → 公平競爭", 20, fill="#166534")
    return s.html()


def fig_chain():
    s = _Svg(1000, 410)
    s.text(500, 36, "一個感知器：往前算（預測），再把 loss 一路「傳回去」（連鎖律）", 25, weight="bold")
    s.p.append('<g transform="translate(0,-60)">')
    s.circle(70, 150, 34, "blue"); s.text(70, 157, "x₁", 24)
    s.text(70, 205, "體重", 18)
    s.circle(70, 290, 34, "red"); s.text(70, 297, "x₂", 24)
    s.text(70, 345, "血鉀", 18)
    _box(s, 235, 190, 190, 70, "z＝w₁x₁＋w₂x₂＋b", "yellow", 18)
    s.line(105, 150, 235, 215, arrow=True); s.line(105, 290, 235, 240, arrow=True)
    s.text(172, 168, "×w₁", 19, fill="#1d4ed8"); s.text(172, 278, "×w₂", 19, fill="#b91c1c")
    _box(s, 470, 190, 130, 70, "ReLU\ny＝max(z, 0)", "purple", 18)
    s.line(425, 225, 470, 225, arrow=True)
    _box(s, 680, 190, 170, 70, "loss＝(y − t)²\nt＝正確答案", "orange", 18)
    s.line(600, 225, 680, 225, arrow=True)
    s.text(905, 232, "L", 30, weight="bold")
    s.line(850, 225, 885, 225, arrow=True)
    for x, t in [(765, "∂L/∂y ＝ 2(y−t)"), (535, "∂y/∂z ＝ 1（z>0）或 0"), (325, "∂z/∂wᵢ ＝ xᵢ")]:
        s.line(x + 40 if x == 765 else x + 60, 285, x - 40 if x != 765 else x - 30, 285, "#dc2626", 3, arrow=True)
        s.text(x, 320, t, 20, fill="#b91c1c", weight="bold")
    s.p.append("</g>")
    s.rect(120, 312, 760, 84, "yellow", 12)
    s.text(500, 345, "梯度 ∂L/∂wᵢ ＝ 2(y−t) × ∂y/∂z × xᵢ　　更新：wᵢ ← wᵢ − 學習率 × 梯度", 21, weight="bold")
    s.text(500, 380, "輸入 xᵢ 越大，梯度越大，這個權重被「推」得越多", 21, fill="#991b1b", weight="bold")
    return s.html()


def fig_splits():
    s = _Svg(1000, 600)
    s.text(500, 36, "不同的「輪流當驗證」方式", 28, weight="bold")
    def panel(x0, title, sub, rounds, n, groups=None):
        s.text(x0 + 150, 80, title, 23, weight="bold")
        s.text(x0 + 150, 108, sub, 17, fill="#374151")
        cw = 300 / n
        for r, val in enumerate(rounds):
            y = 135 + r * 52
            for j in range(n):
                isv = j in val
                s.rect(x0 + j * cw, y, cw - 3, 36, "red" if isv else "blue", 3, 1.5)
            if groups:
                for g in range(1, n // groups):
                    s.line(x0 + g * groups * cw - 1.5, y - 4, x0 + g * groups * cw - 1.5, y + 40, "#111827", 2)
        return 135 + len(rounds) * 52
    y1 = panel(30, "K 折（K＝5）", "切成 5 份，輪流 1 份當驗證", [{0, 1}, {2, 3}, {4, 5}, {6, 7}, {8, 9}], 10)
    s.text(180, y1 + 28, "訓練 5 次", 20, weight="bold")
    y2 = panel(350, "Leave-one-out", "每次只留「1 個樣本」驗證", [{0}, {1}, {2}, {3}, {4}], 10)
    s.text(500, y2 + 4, "⋮", 26); s.text(500, y2 + 40, "N 個樣本 → 訓練 N 次", 20, weight="bold")
    y3 = panel(670, "Leave-one-patient-out", "每次留「1 位病人」（的所有列）驗證", [{0, 1, 2}, {3, 4, 5}, {6, 7, 8}, {9, 10, 11}, {12, 13, 14}], 15, 3)
    s.text(820, y3 + 28, "N 位病人 → 訓練 N 次", 20, weight="bold")
    s.rect(30, 470, 940, 110, "yellow", 12)
    s.text(500, 503, "資料很少時：Leave-one-out 讓訓練資料用得最多，但要訓練 N 次，而且每次驗證只有 1 筆，成績不穩定。", 19)
    s.text(500, 535, "一位病人有多筆資料時：一定要用 Leave-one-patient-out（或 K 折的「以病人分組」版本），否則會洩漏。", 19, weight="bold")
    s.text(500, 565, "紅＝驗證，藍＝訓練；黑色粗線分隔不同病人。", 17, fill="#374151")
    return s.html()


def fig_privacy():
    s = _Svg(1000, 520)
    s.text(500, 38, "研究資料裡，藏著哪些「可以認出病人」的東西？", 27, weight="bold")
    cards = [("purple", "🧑 影像：臉部", ["3D 頭部 MRI／CT 可以重建出臉，", "再用人臉辨識比對到真人", "", "對策：去臉（defacing）、", "只提供去除顱骨的腦組織"]),
             ("blue", "🗂 DICOM 標頭", ["姓名、病歷號、生日", "檢查日期、醫院、醫師", "機器序號、各種 UID", "", "對策：刪除／取代／日期位移", "UID 重新產生；小心影像內嵌文字"]),
             ("yellow", "📝 病歷文字", ["姓名、身分證字號、電話", "地址、日期、病歷號", "「王大明的女兒」", "", "對策：規則＋自然語言處理，", "再人工抽查"])]
    for i, (c, t, body) in enumerate(cards):
        x = 25 + i * 325
        s.rect(x, 70, 305, 250, c, 14)
        s.text(x + 152, 112, t, 24, weight="bold")
        s.text(x + 152, 158, "\n".join(body), 19)
    s.rect(25, 345, 950, 150, "red", 14)
    s.text(500, 382, "去識別化 ≠ 絕對安全", 25, weight="bold", fill="#991b1b")
    s.text(500, 420, "還有「間接識別」：罕見疾病 ＋ 精確日期 ＋ 居住地區，湊在一起也可能認出人。\n所以研究要經過 IRB，資料要依法處理，不要自己帶走、不要放到公開的 AI 服務。", 20)
    return s.html()


FIGS = {k[4:]: v for k, v in globals().items() if k.startswith("fig_") and callable(v)}


def show_fig(name):
    display(HTML(FIGS[name]()))


def slide(name, base=None):
    """顯示 repo 裡 slides/ 資料夾的投影片圖片。"""
    bases = [base] if base else [f"https://raw.githubusercontent.com/{REPO}/v1.0/slides/", f"https://raw.githubusercontent.com/{REPO}/main/slides/"]
    url = bases[-1] + name
    display(HTML(f'<div style="max-width:100%"><img src="{url}" alt="（找不到投影片 {name}：請把圖片放進 repo 的 slides/ 資料夾）" '
                 f'style="width:100%;max-width:1100px;height:auto"></div>'))
