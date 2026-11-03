# ===== 投影用的顯示工具：大字表格、找碴標色、前後對照 =====
import html as _html
from IPython.display import display, HTML
if "FONT_PX" not in globals():
    FONT_PX = 20
_builtin_print = print


def print(*args, sep=" ", end="\n", **kwargs):
    """把 print 換成『大字版』（字體大小由 FONT_PX 決定），方便投影。"""
    if kwargs.get("file") is not None:
        return _builtin_print(*args, sep=sep, end=end, **kwargs)
    text = sep.join(str(a) for a in args) or "\u00a0"
    display(HTML(f'<div style="font-size:{FONT_PX}px;line-height:1.55;white-space:pre-wrap;'
                 f'font-family:monospace;color:inherit">{_html.escape(text)}</div>'))


plt.rcParams.update({"font.size": FONT_PX * 0.75, "axes.titlesize": FONT_PX * 0.85, "axes.labelsize": FONT_PX * 0.8,
                     "legend.fontsize": FONT_PX * 0.7, "xtick.labelsize": FONT_PX * 0.7, "ytick.labelsize": FONT_PX * 0.7})


def _fmt(v):
    if v is None or (isinstance(v, (float, np.floating)) and pd.isna(v)):
        return "NaN"
    if isinstance(v, (float, np.floating)):
        return f"{v:g}"
    if isinstance(v, (int, np.integer)):
        return str(v)
    s = str(v)
    if s == "":
        return "⟨空白⟩"
    lead, trail = len(s) - len(s.lstrip()), len(s) - len(s.rstrip())
    return ("␣" * lead + s.strip() + "␣" * trail) if (lead or trail) else s


def show(df, styler=None, index=False):
    """用大字顯示表格（字體大小由 FONT_PX 決定）。␣＝多餘的空格，⟨空白⟩＝空的格子。"""
    st = styler if styler is not None else df.style
    st = st.format(_fmt)
    if not index:
        try:
            st = st.hide(axis="index")
        except Exception:
            st = st.hide_index()
    px = FONT_PX
    st = st.set_table_styles([
        {"selector": "th", "props": [("font-size", f"{px}px"), ("padding", "6px 12px"), ("text-align", "left"),
                                     ("background-color", "#e8eaf6"), ("color", "#000")]},
        {"selector": "td", "props": [("font-size", f"{px}px"), ("padding", "6px 12px"),
                                     ("background-color", "#fff"), ("color", "#000")]}])
    display(st)


def _legend(items):
    boxes = "".join(f'<span style="background:{c};color:#000;padding:4px 10px;margin-right:12px;border-radius:4px">{t}</span>'
                    for c, t in items)
    display(HTML(f'<div style="font-size:{FONT_PX * 0.85}px;margin:8px 0">{boxes}</div>'))


ISSUE_COLORS = {"缺值": "#ff8a80", "數字欄位混入文字": "#ff8a80", "不可能的數值": "#ffcc80", "單位疑似錯誤": "#ffcc80",
                "類別寫法不一致": "#fff59d", "重複列": "#bbdefb"}


def show_issues(df, rows=None):
    """把有問題的格子標上顏色。rows：只顯示這幾列（用 df 的列編號）。"""
    res, _ = find_issues(df)
    css = pd.DataFrame("", index=df.index, columns=df.columns)
    dup = df.duplicated("patient_id", keep=False)
    for i in df.index[dup]:
        css.loc[i, :] = "background-color:#bbdefb !important;color:#000"
    for r in res.itertuples():
        if r.column != "(整列)":
            css.loc[r.row, r.column] = f"background-color:{ISSUE_COLORS[r.type]} !important;color:#000;font-weight:bold"
    sub = df if rows is None else df.loc[rows]
    show(sub, styler=sub.style.apply(lambda _: css.loc[sub.index], axis=None))
    _legend([("#ff8a80", "紅：轉成數字時會變成 NaN（缺值、混入文字）"), ("#ffcc80", "橘：轉得成數字，但數值不可能或單位不對"),
             ("#fff59d", "黃：文字寫法不一致"), ("#bbdefb", "藍：重複的病人")])


def show_changes(before, after, title, cols=None, n=6, key="patient_id"):
    """列出這一步『改了哪些格子』：改之前 → 改之後。"""
    b = before.drop_duplicates(key).set_index(key)
    a = after.drop_duplicates(key).set_index(key)
    def same(x, y):
        if pd.isna(x) and pd.isna(y):
            return True
        if pd.isna(x) or pd.isna(y):
            return False
        fx, fy = _num(x), _num(y)
        return abs(fx - fy) < 1e-9 if (fx is not None and fy is not None) else str(x) == str(y)
    rows = []
    for pid in a.index.intersection(b.index):
        for c in (cols or list(a.columns)):
            if c in b.columns and not same(b.at[pid, c], a.at[pid, c]):
                rows.append({"patient_id": pid, "欄位": c, "改之前": b.at[pid, c], "改之後": a.at[pid, c]})
    print(f"▶ {title}　（共改了 {len(rows)} 格" + (f"，列出前 {n} 格）" if len(rows) > n else "）"))
    if rows:
        show(pd.DataFrame(rows).head(n))

