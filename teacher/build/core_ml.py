# ===== 小型 MLP 與評估工具 =====
import torch, torch.nn as nn
from sklearn.metrics import roc_auc_score, recall_score, precision_score, accuracy_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder

FEATURES_V1 = NUM_COLS + ["male"] + BIN_COLS + CODE_COLS      # 單元一、二使用：11 個全數字欄位
CATEGORIES = {"chest_pain": CAT_COLS["chest_pain"], "thal": CAT_COLS["thal"],
              "rest_ecg": [0.0, 1.0, 2.0], "slope": [1.0, 2.0, 3.0]}


def add_male(df):
    d = df.copy()
    d["male"] = (d["sex"] == "male").astype(float)
    return d


ROUGH = {"age": 100, "resting_bp": 200, "chol": 400, "max_hr": 200, "oldpeak": 6, "ca": 3, "rest_ecg": 2, "slope": 3}


def to_xy(df, features=FEATURES_V1, rough=True):
    """轉成模型吃的數字矩陣。rough=True：每個欄位除以一個固定的數字，縮到大約 0–1（單元 3 會解釋為什麼）。"""
    d = add_male(df)
    X = d[features].astype("float32").values
    if rough:
        for j, c in enumerate(features):
            if c in ROUGH:
                X[:, j] /= ROUGH[c]
    y = (d["disease"] == "yes").astype("float32").values
    return X, y


class MLP(nn.Module):
    """hidden=(3,)：輸入 → 3 個隱藏感知器（tanh，一點點非線性）→ 1 個輸出感知器。共 4 個感知器。"""
    def __init__(self, n_in, hidden=(3,)):
        super().__init__()
        layers, prev = [], n_in
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.Tanh()]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_mlp(X, y, epochs=300, lr=0.003, seed=0, pos_weight=None, X_val=None, y_val=None, hidden=(3,)):
    """整批資料一起訓練。回傳 (model, history)。history 裡有每個 epoch 的 loss。"""
    torch.manual_seed(seed)
    Xt, yt = torch.tensor(np.asarray(X), dtype=torch.float32), torch.tensor(np.asarray(y), dtype=torch.float32)
    model = MLP(Xt.shape[1], hidden)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    pw = None if pos_weight is None else torch.tensor(float(pos_weight))
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pw)
    hist = {"train": [], "val": []}
    for ep in range(epochs):
        model.train()
        opt.zero_grad()
        loss = loss_fn(model(Xt), yt)
        loss.backward()
        opt.step()
        hist["train"].append(loss.item())
        if X_val is not None:
            model.eval()
            with torch.no_grad():
                hist["val"].append(nn.BCEWithLogitsLoss()(model(torch.tensor(np.asarray(X_val), dtype=torch.float32)),
                                                          torch.tensor(np.asarray(y_val), dtype=torch.float32)).item())
    return model, hist


def predict_proba(model, X):
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(torch.tensor(np.asarray(X), dtype=torch.float32))).numpy()


def evaluate(model, X, y, threshold=0.5):
    p = predict_proba(model, X)
    pred = (p >= threshold).astype(int)
    out = {"acc": accuracy_score(y, pred), "recall": recall_score(y, pred, zero_division=0),
           "precision": precision_score(y, pred, zero_division=0)}
    out["auc"] = roc_auc_score(y, p) if len(set(y)) > 1 and np.isfinite(p).all() else float("nan")
    return out
