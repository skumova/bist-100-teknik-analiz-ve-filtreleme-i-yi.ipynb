"""
src/model.py
=============

XGBoost tabanlı yön tahmin modeli, Walk-Forward Validation (ileriye yönelik
adımsal test) motoru ve otomatik hiperparametre optimizasyonu.

Model sabit kalmaz: `WalkForwardEngine.run()` her `step_bars` barda bir
geçmiş `train_bars` barlık pencereyle modeli yeniden eğitir, hiperparametreleri
(n_estimators, max_depth, learning_rate) periyodik olarak yeniden arar ve
her adımın başarı istatistiklerini (precision, recall, f1) şeffaf biçimde
loglar / CSV'ye kaydeder.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from xgboost import XGBClassifier

from src import config
from src.features import FEATURE_COLUMNS

logger = logging.getLogger("bist_bot.model")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


# ---------------------------------------------------------------------- #
# Etiketleme: X bar sonraki yön (yukarı / diğer)
# ---------------------------------------------------------------------- #
def make_labels(
    df: pd.DataFrame,
    horizon: int = config.LABEL_HORIZON_BARS,
    threshold: float = config.LABEL_UP_THRESHOLD,
    price_col: str = "close",
) -> pd.Series:
    """`horizon` bar ileride fiyat `threshold` üzerinde yükseldiyse 1, aksi halde 0."""
    future_log_return = np.log(df[price_col].shift(-horizon) / df[price_col])
    label = (future_log_return > threshold).astype("Int64")
    label.name = "label"
    return label


# ---------------------------------------------------------------------- #
# Hiperparametre optimizasyonu
# ---------------------------------------------------------------------- #
def optimize_hyperparameters(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_trials: int = 20,
    random_state: int = config.RANDOM_STATE,
) -> dict:
    """Optuna ile n_estimators / max_depth / learning_rate araması yapar.

    Optuna kurulu değilse veya yetersiz veri varsa varsayılan parametrelere
    (config.XGB_DEFAULT_PARAMS) düşer.
    """
    if len(X_train) < 100:
        logger.info("Eğitim verisi optimizasyon için çok küçük (%d satır); varsayılan parametreler kullanılacak.", len(X_train))
        return dict(config.XGB_DEFAULT_PARAMS)

    try:
        import optuna

        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        logger.warning("optuna kurulu değil; varsayılan XGBoost parametreleri kullanılacak.")
        return dict(config.XGB_DEFAULT_PARAMS)

    split_idx = int(len(X_train) * 0.8)
    X_fit, X_val = X_train.iloc[:split_idx], X_train.iloc[split_idx:]
    y_fit, y_val = y_train.iloc[:split_idx], y_train.iloc[split_idx:]

    if y_fit.nunique() < 2 or y_val.nunique() < 2:
        logger.info("Optimizasyon alt kümesinde tek sınıf var; varsayılan parametreler kullanılacak.")
        return dict(config.XGB_DEFAULT_PARAMS)

    n_est_lo, n_est_hi = config.XGB_PARAM_SPACE["n_estimators"]
    depth_lo, depth_hi = config.XGB_PARAM_SPACE["max_depth"]
    lr_lo, lr_hi = config.XGB_PARAM_SPACE["learning_rate"]

    def objective(trial: "optuna.Trial") -> float:
        params = dict(config.XGB_DEFAULT_PARAMS)
        params.update(
            n_estimators=trial.suggest_int("n_estimators", n_est_lo, n_est_hi, step=50),
            max_depth=trial.suggest_int("max_depth", depth_lo, depth_hi),
            learning_rate=trial.suggest_float("learning_rate", lr_lo, lr_hi, log=True),
        )
        clf = XGBClassifier(**params, random_state=random_state)
        clf.fit(X_fit, y_fit, verbose=False)
        preds = clf.predict(X_val)
        return f1_score(y_val, preds, zero_division=0)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=random_state))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_params = dict(config.XGB_DEFAULT_PARAMS)
    best_params.update(study.best_params)
    logger.info("Optuna en iyi parametreler: %s (f1=%.4f)", study.best_params, study.best_value)
    return best_params


# ---------------------------------------------------------------------- #
# Tekil eğitim / değerlendirme
# ---------------------------------------------------------------------- #
def train_xgb(X_train: pd.DataFrame, y_train: pd.Series, params: Optional[dict] = None) -> XGBClassifier:
    params = dict(params or config.XGB_DEFAULT_PARAMS)
    model = XGBClassifier(**params, random_state=config.RANDOM_STATE)
    model.fit(X_train, y_train, verbose=False)
    return model


def evaluate_predictions(y_true: pd.Series, y_pred: np.ndarray) -> dict:
    return {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "accuracy": accuracy_score(y_true, y_pred),
        "positive_rate": float(np.mean(y_pred)),
        "n_test": int(len(y_true)),
    }


# ---------------------------------------------------------------------- #
# Walk-Forward Validation motoru
# ---------------------------------------------------------------------- #
@dataclass
class WalkForwardFold:
    fold_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    params: dict
    metrics: dict
    financial_metrics: dict = field(default_factory=dict)


class WalkForwardEngine:
    """XGBoost modelini periyodik olarak yeniden eğiten walk-forward motoru.

    Her adımda:
    1. Son `train_bars` barla model (yeniden) eğitilir; `retrain_every` fold'da
       bir Optuna ile hiperparametreler yeniden aranır.
    2. Bir sonraki `step_bars` bar üzerinde tahmin yapılır (out-of-sample).
    3. Precision/Recall/F1 hesaplanır ve loglanır; istenirse finansal
       metrikler (Sharpe/Drawdown/WinRate) için `financial_metrics_fn`
       callback'i çağrılır (bkz. src.backtester.compute_financial_metrics).
    """

    def __init__(
        self,
        feature_columns: Optional[list[str]] = None,
        train_bars: int = config.WALK_FORWARD_TRAIN_BARS,
        step_bars: int = config.WALK_FORWARD_STEP_BARS,
        min_test_bars: int = config.WALK_FORWARD_MIN_TEST_BARS,
        retrain_every: int = 1,
        optuna_trials: int = 20,
        use_optuna: bool = True,
        probability_threshold: float = config.ENTRY_PROBABILITY_THRESHOLD,
        financial_metrics_fn: Optional[Callable[[pd.DataFrame], dict]] = None,
        log_path: Optional[Path] = None,
    ):
        self.feature_columns = feature_columns or list(FEATURE_COLUMNS)
        self.train_bars = train_bars
        self.step_bars = step_bars
        self.min_test_bars = min_test_bars
        self.retrain_every = max(1, retrain_every)
        self.optuna_trials = optuna_trials
        self.use_optuna = use_optuna
        self.probability_threshold = probability_threshold
        self.financial_metrics_fn = financial_metrics_fn
        self.log_path = Path(log_path) if log_path else config.LOG_DIR / "walk_forward_metrics.csv"

        self.folds: list[WalkForwardFold] = []
        self.model: Optional[XGBClassifier] = None
        self.current_params: dict = dict(config.XGB_DEFAULT_PARAMS)

    def run(self, feat_df: pd.DataFrame) -> pd.DataFrame:
        """feat_df: build_feature_matrix çıktısı + ham OHLCV kolonlarını içermelidir."""
        df = feat_df.copy()
        df["label"] = make_labels(df)
        df = df.dropna(subset=self.feature_columns + ["label"])
        df["label"] = df["label"].astype(int)

        n = len(df)
        if n < self.train_bars + self.min_test_bars:
            raise ValueError(
                f"Walk-forward için yetersiz veri: {n} satır var, en az "
                f"{self.train_bars + self.min_test_bars} gerekiyor."
            )

        oos_records = []
        fold_id = 0
        start = 0
        while start + self.train_bars + self.min_test_bars <= n:
            train_slice = df.iloc[start : start + self.train_bars]
            test_end = min(start + self.train_bars + self.step_bars, n)
            test_slice = df.iloc[start + self.train_bars : test_end]
            if len(test_slice) < self.min_test_bars:
                break

            X_train, y_train = train_slice[self.feature_columns], train_slice["label"]
            X_test, y_test = test_slice[self.feature_columns], test_slice["label"]

            if fold_id % self.retrain_every == 0:
                if self.use_optuna:
                    self.current_params = optimize_hyperparameters(X_train, y_train, n_trials=self.optuna_trials)
                else:
                    self.current_params = dict(config.XGB_DEFAULT_PARAMS)

            self.model = train_xgb(X_train, y_train, self.current_params)

            proba = self.model.predict_proba(X_test)[:, 1]
            preds = (proba >= self.probability_threshold).astype(int)
            metrics = evaluate_predictions(y_test, preds)

            fold_result_df = test_slice.copy()
            fold_result_df["proba_up"] = proba
            fold_result_df["signal"] = preds
            fold_result_df["fold_id"] = fold_id
            oos_records.append(fold_result_df)

            financial_metrics = {}
            if self.financial_metrics_fn is not None:
                try:
                    financial_metrics = self.financial_metrics_fn(fold_result_df)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Fold %d finansal metrik hesaplama hatası: %s", fold_id, exc)

            fold = WalkForwardFold(
                fold_id=fold_id,
                train_start=train_slice.index[0],
                train_end=train_slice.index[-1],
                test_start=test_slice.index[0],
                test_end=test_slice.index[-1],
                params=dict(self.current_params),
                metrics=metrics,
                financial_metrics=financial_metrics,
            )
            self.folds.append(fold)
            self._log_fold(fold)

            fold_id += 1
            start += self.step_bars

        self._write_log_csv()
        return pd.concat(oos_records) if oos_records else pd.DataFrame()

    def _log_fold(self, fold: WalkForwardFold) -> None:
        logger.info(
            "Fold %d | test %s -> %s | precision=%.3f recall=%.3f f1=%.3f | params=%s | finansal=%s",
            fold.fold_id,
            fold.test_start,
            fold.test_end,
            fold.metrics["precision"],
            fold.metrics["recall"],
            fold.metrics["f1"],
            fold.params,
            fold.financial_metrics,
        )

    def _write_log_csv(self) -> None:
        rows = []
        for fold in self.folds:
            row = {
                "fold_id": fold.fold_id,
                "train_start": fold.train_start,
                "train_end": fold.train_end,
                "test_start": fold.test_start,
                "test_end": fold.test_end,
                **{f"param_{k}": v for k, v in fold.params.items() if k in config.XGB_PARAM_SPACE},
                **fold.metrics,
                **{f"fin_{k}": v for k, v in fold.financial_metrics.items()},
            }
            rows.append(row)
        if rows:
            pd.DataFrame(rows).to_csv(self.log_path, index=False)
            logger.info("Walk-forward metrik logu yazıldı: %s", self.log_path)

    def summary(self) -> dict:
        if not self.folds:
            return {}
        metrics_df = pd.DataFrame([f.metrics for f in self.folds])
        return {
            "n_folds": len(self.folds),
            "avg_precision": metrics_df["precision"].mean(),
            "avg_recall": metrics_df["recall"].mean(),
            "avg_f1": metrics_df["f1"].mean(),
            "last_params": self.folds[-1].params,
        }

    def save_model(self, path: Optional[Path] = None) -> Path:
        if self.model is None:
            raise RuntimeError("Kaydedilecek eğitilmiş model yok; önce run() çalıştırın.")
        path = Path(path) if path else config.MODEL_CACHE_DIR / "xgb_latest.json"
        self.model.save_model(str(path))
        meta_path = path.with_suffix(".meta.json")
        meta_path.write_text(json.dumps({"params": self.current_params, "feature_columns": self.feature_columns}, indent=2))
        logger.info("Model kaydedildi: %s", path)
        return path

    def load_model(self, path: Optional[Path] = None) -> XGBClassifier:
        path = Path(path) if path else config.MODEL_CACHE_DIR / "xgb_latest.json"
        meta_path = path.with_suffix(".meta.json")
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        params = meta.get("params", config.XGB_DEFAULT_PARAMS)
        self.feature_columns = meta.get("feature_columns", self.feature_columns)

        model = XGBClassifier(**{k: v for k, v in params.items() if k in config.XGB_DEFAULT_PARAMS})
        model.load_model(str(path))
        self.model = model
        self.current_params = params
        return model


def predict_latest(model: XGBClassifier, feature_row: pd.Series, feature_columns: list[str]) -> float:
    """Canlı işlem döngüsü için tek satırlık (en güncel bar) yukarı yön olasılığı.

    `feature_row`, filtre (bool) kolonlarıyla karışık bir DataFrame'den
    `.iloc[-1]` ile alınmış olabilir; bu durumda Series dtype'ı object'e
    yükselir. XGBoost'un sayısal olmayan dtype'ları reddetmemesi için
    açıkça float'a çevriliyor.
    """
    X = feature_row[feature_columns].to_frame().T.astype(float)
    return float(model.predict_proba(X)[:, 1][0])
