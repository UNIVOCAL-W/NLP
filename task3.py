"""
Task 3: Sentiment classification.

This script trains and evaluates classical sentiment classifiers:
    - Naive Bayes
    - Feed-forward neural network, implemented with PyTorch and optional CUDA

It supports both:
    - multiclass classification: negative / neutral / positive
    - binary classification: negative / positive, with neutral removed

It also compares different text representations:
    - bag-of-words
    - TF-IDF

Usage:
    python task3.py
    python task3.py --config task3_config.json
    python task3.py --sample-size 5000 --skip-ffnn
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

try:
    import torch
    from torch import nn
except ImportError:
    torch = None
    nn = None


DEFAULT_CONFIG_FILE = "task3_config.json"
SCRIPT_DIR = Path(__file__).resolve().parent

DEFAULT_CONFIG: Dict[str, Any] = {
    "data": {
        "csv_path": "outputs/tasks1_2/preprocessed_tweets.csv",
        "text_column": "processed_text",
        "raw_text_column": "text",
        "label_column": "sentiment",
        "sample_size": None,
        "random_seed": 42,
        "encoding": "utf-8-sig",
    },
    "task": {
        "test_size": 0.2,
        "run_multiclass": True,
        "run_binary": True,
        "binary_drop_label": "neutral",
        "multiclass_label_order": ["negative", "neutral", "positive"],
        "binary_label_order": ["negative", "positive"],
    },
    "output": {
        "output_dir": "outputs/task3",
        "save_models": False,
        "plot_format": "png",
        "figure_dpi": 160,
        "error_examples_per_experiment": 50,
        "save_full_predictions": True,
    },
    "progress": {
        "enabled": True,
        "level": "data",
        "leave": True,
        "ncols": 140,
        "log_each_experiment": False,
        "ascii": True,
        "show_stage": False,
        "show_row_ids": True,
        "data_batch_size": 512,
    },
    "class_handling": {
        "strategy": "none",
        "oversample_random_seed": 42,
    },
    "vectorizers": {
        "bow": {
            "enabled": True,
            "type": "count",
            "lowercase": False,
            "ngram_range": [1, 1],
            "min_df": 2,
            "max_df": 1.0,
            "max_features": 20000,
            "binary": False,
            "token_pattern": r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+(?:\.\d+)?|[!?]+",
        },
        "tfidf": {
            "enabled": True,
            "type": "tfidf",
            "lowercase": False,
            "ngram_range": [1, 2],
            "min_df": 2,
            "max_df": 0.95,
            "max_features": 30000,
            "sublinear_tf": True,
            "use_idf": True,
            "norm": "l2",
            "token_pattern": r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+(?:\.\d+)?|[!?]+",
        },
    },
    "models": {
        "naive_bayes": {
            "enabled": True,
            "alpha": 1.0,
            "fit_prior": True,
        },
        "ffnn": {
            "enabled": True,
            "framework": "pytorch",
            "device": "auto",
            "hidden_layer_sizes": [128],
            "activation": "relu",
            "optimizer": "adam",
            "dropout": 0.0,
            "alpha": 0.0001,
            "batch_size": 128,
            "learning_rate_init": 0.001,
            "epochs": 25,
            "max_iter": 25,
            "early_stopping": True,
            "validation_fraction": 0.1,
            "n_iter_no_change": 5,
            "verbose": False,
        },
    },
}


LOGGER = logging.getLogger("task3")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Task 3 classification experiments.")
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG_FILE,
        help=(
            "JSON config that overrides built-in defaults. "
            f"Defaults to {DEFAULT_CONFIG_FILE} if present."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional output directory override.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Optional sample size override for quick experiments.",
    )
    parser.add_argument(
        "--skip-nb",
        action="store_true",
        help="Disable Naive Bayes experiments.",
    )
    parser.add_argument(
        "--skip-ffnn",
        action="store_true",
        help="Disable feed-forward neural network experiments.",
    )
    return parser.parse_args()


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def deep_update(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def resolve_project_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return SCRIPT_DIR / path


def resolve_existing_config_path(path_value: str | Path) -> Optional[Path]:
    path = Path(path_value)
    if path.is_absolute():
        return path if path.exists() else None

    cwd_path = Path.cwd() / path
    if cwd_path.exists():
        return cwd_path

    script_path = SCRIPT_DIR / path
    if script_path.exists():
        return script_path

    return None


def load_config(args: argparse.Namespace) -> Dict[str, Any]:
    config = deepcopy(DEFAULT_CONFIG)

    config_path = resolve_existing_config_path(args.config) if args.config else None
    if config_path:
        with config_path.open("r", encoding="utf-8") as file:
            deep_update(config, json.load(file))
    elif args.config != DEFAULT_CONFIG_FILE:
        raise FileNotFoundError(f"Config file not found: {args.config}")

    if args.output_dir:
        config["output"]["output_dir"] = args.output_dir
    if args.sample_size is not None:
        config["data"]["sample_size"] = args.sample_size
    if args.skip_nb:
        config["models"]["naive_bayes"]["enabled"] = False
    if args.skip_ffnn:
        config["models"]["ffnn"]["enabled"] = False

    return config


def ensure_output_dirs(config: Dict[str, Any]) -> Dict[str, Path]:
    root = resolve_project_path(config["output"]["output_dir"])
    dirs = {
        "root": root,
        "reports": root / "classification_reports",
        "confusion": root / "confusion_matrices",
        "predictions": root / "predictions",
        "errors": root / "errors",
        "models": root / "models",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def save_json(data: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)


def save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_").lower()


def make_progress_bar(total: int, desc: str, config: Dict[str, Any]):
    progress_config = config.get("progress", {})
    if not progress_config.get("enabled", True):
        return None

    if tqdm is None:
        LOGGER.warning("tqdm is not installed; progress bar is disabled.")
        return None

    progress_bar = tqdm(
        total=total,
        desc=desc,
        leave=progress_config.get("leave", True),
        ncols=progress_config.get("ncols", 120),
        dynamic_ncols=progress_config.get("ncols") is None,
        ascii=progress_config.get("ascii", False),
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}",
    )
    progress_bar.task3_progress_config = progress_config
    return progress_bar


def set_progress_stage(progress_bar: Any, stage: str) -> None:
    if progress_bar is not None:
        progress_config = getattr(progress_bar, "task3_progress_config", {})
        if progress_config.get("show_stage", False):
            progress_bar.set_postfix_str(stage, refresh=True)


def update_progress(progress_bar: Any, postfix: Optional[str] = None, n: int = 1) -> None:
    if progress_bar is not None:
        if postfix:
            progress_bar.set_postfix_str(postfix, refresh=False)
        progress_bar.update(n)


def should_log_experiment(config: Dict[str, Any], progress_bar: Any) -> bool:
    if progress_bar is None:
        return True
    return bool(config.get("progress", {}).get("log_each_experiment", False))


def progress_level(config: Dict[str, Any]) -> str:
    return str(config.get("progress", {}).get("level", "experiment")).lower()


def use_data_progress(config: Dict[str, Any]) -> bool:
    progress_config = config.get("progress", {})
    return bool(progress_config.get("enabled", True)) and progress_level(config) in {
        "data",
        "batch",
        "row",
        "rows",
    }


def progress_batch_size(config: Dict[str, Any]) -> int:
    value = int(config.get("progress", {}).get("data_batch_size", 512))
    return max(1, value)


def row_id_text(df: pd.DataFrame) -> str:
    if df.empty:
        return "ids=n/a"

    if "source_row_id" in df.columns:
        ids = df["source_row_id"].tolist()
    else:
        ids = df.index.tolist()

    first = ids[0]
    last = ids[-1]
    if len(ids) == 1:
        return f"id={first}"
    return f"ids={first}..{last}"


def iter_batches(
    matrix: Any,
    df: pd.DataFrame,
    y: Optional[np.ndarray],
    batch_size: int,
):
    total_rows = matrix.shape[0]
    for start in range(0, total_rows, batch_size):
        end = min(start + batch_size, total_rows)
        batch_df = df.iloc[start:end]
        batch_y = y[start:end] if y is not None else None
        yield start, end, batch_df, matrix[start:end], batch_y


def adjust_ffnn_batch_size(model: Any, model_name: str, model_config: Dict[str, Any], rows: int) -> None:
    if model_name != "ffnn":
        return

    configured_batch_size = model_config.get("batch_size", getattr(model, "batch_size", "auto"))
    if isinstance(configured_batch_size, int):
        model.batch_size = max(1, min(configured_batch_size, rows))


def load_dataset(config: Dict[str, Any]) -> pd.DataFrame:
    data_config = config["data"]
    csv_path = resolve_project_path(data_config["csv_path"])
    text_col = data_config["text_column"]
    label_col = data_config["label_column"]
    raw_text_col = data_config.get("raw_text_column")

    LOGGER.info("Loading data from %s", csv_path)
    df = pd.read_csv(csv_path, encoding=data_config["encoding"])

    required = [text_col, label_col]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {csv_path}: {missing}")

    if raw_text_col and raw_text_col not in df.columns:
        LOGGER.warning(
            "raw_text_column=%s is missing. Falling back to text_column=%s.",
            raw_text_col,
            text_col,
        )
        raw_text_col = text_col
        config["data"]["raw_text_column"] = text_col

    keep_columns = [text_col, label_col]
    if raw_text_col and raw_text_col not in keep_columns:
        keep_columns.append(raw_text_col)
    df = df.loc[:, keep_columns].copy()

    df[text_col] = df[text_col].fillna("").astype(str)
    if raw_text_col:
        df[raw_text_col] = df[raw_text_col].fillna("").astype(str)
    df[label_col] = df[label_col].astype(str).str.strip().str.lower()

    df = df[df[text_col].str.strip().astype(bool)].copy()
    df = df[df[label_col].str.strip().astype(bool)].copy()

    sample_size = data_config.get("sample_size")
    if sample_size is not None and sample_size < len(df):
        df, _ = train_test_split(
            df,
            train_size=sample_size,
            random_state=data_config["random_seed"],
            stratify=df[label_col] if df[label_col].nunique() > 1 else None,
        )

    df = df.reset_index(drop=True)
    df.insert(0, "source_row_id", df.index)
    LOGGER.info("Loaded %d rows with labels: %s", len(df), sorted(df[label_col].unique()))
    return df


def get_enabled_items(config: Dict[str, Any], section: str) -> Dict[str, Dict[str, Any]]:
    items = {}
    for name, item_config in config[section].items():
        if item_config.get("enabled", True):
            items[name] = item_config
    return items


def make_dataset_for_setting(
    df: pd.DataFrame,
    setting: str,
    config: Dict[str, Any],
) -> Tuple[pd.DataFrame, List[str]]:
    label_col = config["data"]["label_column"]
    task_config = config["task"]

    if setting == "binary":
        drop_label = task_config["binary_drop_label"]
        setting_df = df[df[label_col] != drop_label].copy()
        labels = [label for label in task_config["binary_label_order"] if label in set(setting_df[label_col])]
    elif setting == "multiclass":
        setting_df = df.copy()
        labels = [
            label
            for label in task_config["multiclass_label_order"]
            if label in set(setting_df[label_col])
        ]
    else:
        raise ValueError(f"Unknown setting: {setting}")

    if len(labels) < 2:
        raise ValueError(f"{setting} needs at least two labels, got: {labels}")

    return setting_df.reset_index(drop=True), labels


def split_dataset(df: pd.DataFrame, config: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    label_col = config["data"]["label_column"]
    train_df, test_df = train_test_split(
        df,
        test_size=config["task"]["test_size"],
        random_state=config["data"]["random_seed"],
        stratify=df[label_col],
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def oversample_training_data(train_df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    strategy = config["class_handling"].get("strategy", "none").lower()
    if strategy in {"none", "off", "false"}:
        return train_df
    if strategy != "balanced_oversample":
        raise ValueError(f"Unsupported class handling strategy: {strategy}")

    label_col = config["data"]["label_column"]
    seed = config["class_handling"].get("oversample_random_seed", config["data"]["random_seed"])
    counts = train_df[label_col].value_counts()
    target_count = int(counts.max())

    groups = []
    for offset, (label, group) in enumerate(train_df.groupby(label_col)):
        groups.append(
            group.sample(
                n=target_count,
                replace=len(group) < target_count,
                random_state=seed + offset,
            )
        )

    balanced = pd.concat(groups, ignore_index=True)
    balanced = balanced.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    LOGGER.info(
        "Oversampled training data from %d to %d rows.",
        len(train_df),
        len(balanced),
    )
    return balanced


def build_vectorizer(name: str, config: Dict[str, Any]):
    vectorizer_type = config.get("type", name).lower()
    common = {
        "lowercase": config.get("lowercase", True),
        "ngram_range": tuple(config.get("ngram_range", [1, 1])),
        "min_df": config.get("min_df", 1),
        "max_df": config.get("max_df", 1.0),
        "max_features": config.get("max_features"),
        "token_pattern": config.get("token_pattern"),
    }

    if vectorizer_type == "tfidf":
        return TfidfVectorizer(
            **common,
            sublinear_tf=config.get("sublinear_tf", False),
            use_idf=config.get("use_idf", True),
            norm=config.get("norm", "l2"),
        )

    if vectorizer_type in {"count", "bow", "bag_of_words"}:
        return CountVectorizer(
            **common,
            binary=config.get("binary", False),
        )

    raise ValueError(f"Unsupported vectorizer type for {name}: {vectorizer_type}")


def require_torch() -> None:
    if torch is None or nn is None:
        raise ImportError(
            "PyTorch is required for the FFNN model. Install it with a CUDA build if you want GPU acceleration."
        )


def resolve_torch_device(model_config: Dict[str, Any]) -> Any:
    require_torch()
    requested = str(model_config.get("device", "auto")).lower()

    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if requested.startswith("cuda") and not torch.cuda.is_available():
        LOGGER.warning("CUDA was requested but is not available. Falling back to CPU.")
        return torch.device("cpu")

    return torch.device(requested)


def activation_layer(name: str) -> Any:
    require_torch()
    name = str(name).lower()
    if name == "relu":
        return nn.ReLU()
    if name == "tanh":
        return nn.Tanh()
    if name == "gelu":
        return nn.GELU()
    if name in {"leaky_relu", "leaky-relu"}:
        return nn.LeakyReLU()
    raise ValueError(f"Unsupported PyTorch FFNN activation: {name}")


class TorchFFNNNetwork(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, model_config: Dict[str, Any]):
        super().__init__()
        hidden_sizes = [int(size) for size in model_config.get("hidden_layer_sizes", [128])]
        dropout = float(model_config.get("dropout", 0.0))
        activation = model_config.get("activation", "relu")

        layers = []
        previous_dim = input_dim
        for hidden_dim in hidden_sizes:
            layers.append(nn.Linear(previous_dim, hidden_dim))
            layers.append(activation_layer(activation))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            previous_dim = hidden_dim

        layers.append(nn.Linear(previous_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x: Any) -> Any:
        return self.network(x)


def sparse_matrix_to_tensor(batch_matrix: Any, device: Any) -> Any:
    require_torch()
    if hasattr(batch_matrix, "toarray"):
        batch_array = batch_matrix.toarray()
    else:
        batch_array = np.asarray(batch_matrix)
    batch_array = batch_array.astype(np.float32, copy=False)
    return torch.as_tensor(batch_array, dtype=torch.float32, device=device)


class TorchFFNNClassifier:
    def __init__(self, model_config: Dict[str, Any], random_seed: int):
        require_torch()
        self.model_config = deepcopy(model_config)
        self.random_seed = random_seed
        self.device_ = resolve_torch_device(model_config)
        self.model_: Optional[TorchFFNNNetwork] = None
        self.classes_: Optional[np.ndarray] = None
        self.history_: List[Dict[str, float]] = []

    @property
    def framework_(self) -> str:
        return "pytorch"

    def initialize(self, input_dim: int, classes: np.ndarray) -> None:
        require_torch()
        torch.manual_seed(self.random_seed)
        if self.device_.type == "cuda":
            torch.cuda.manual_seed_all(self.random_seed)
        self.classes_ = np.asarray(classes)
        self.model_ = TorchFFNNNetwork(input_dim, len(classes), self.model_config).to(self.device_)

    def optimizer(self) -> Any:
        require_torch()
        if self.model_ is None:
            raise RuntimeError("Torch FFNN has not been initialized.")

        learning_rate = float(self.model_config.get("learning_rate_init", 0.001))
        weight_decay = float(self.model_config.get("weight_decay", self.model_config.get("alpha", 0.0)))
        optimizer_name = str(self.model_config.get("optimizer", "adam")).lower()

        if optimizer_name == "adam":
            return torch.optim.Adam(self.model_.parameters(), lr=learning_rate, weight_decay=weight_decay)
        if optimizer_name == "sgd":
            momentum = float(self.model_config.get("momentum", 0.0))
            return torch.optim.SGD(
                self.model_.parameters(),
                lr=learning_rate,
                momentum=momentum,
                weight_decay=weight_decay,
            )
        raise ValueError(f"Unsupported PyTorch FFNN optimizer: {optimizer_name}")

    def fit_batches(
        self,
        x_train: Any,
        y_train: np.ndarray,
        train_df: pd.DataFrame,
        labels: List[str],
        experiment_id: str,
        config: Dict[str, Any],
    ) -> Tuple[float, str]:
        require_torch()
        start_time = time.perf_counter()
        classes = np.arange(len(labels), dtype=int)
        self.initialize(input_dim=int(x_train.shape[1]), classes=classes)

        if self.model_ is None:
            raise RuntimeError("Torch FFNN initialization failed.")

        epochs = int(self.model_config.get("epochs", self.model_config.get("max_iter", 25)))
        epochs = max(1, epochs)
        batch_size = max(1, int(self.model_config.get("batch_size", progress_batch_size(config))))
        loss_fn = nn.CrossEntropyLoss()
        optimizer = self.optimizer()
        total_steps = len(train_df) * epochs
        progress_bar = make_progress_bar(total_steps, f"{experiment_id} train", config) if use_data_progress(config) else None

        try:
            for epoch in range(epochs):
                self.model_.train()
                epoch_loss = 0.0
                epoch_seen = 0

                for _, _, batch_df, x_batch, y_batch in iter_batches(
                    x_train,
                    train_df,
                    y_train,
                    batch_size,
                ):
                    features = sparse_matrix_to_tensor(x_batch, self.device_)
                    targets = torch.as_tensor(y_batch, dtype=torch.long, device=self.device_)

                    optimizer.zero_grad(set_to_none=True)
                    logits = self.model_(features)
                    loss = loss_fn(logits, targets)
                    loss.backward()
                    optimizer.step()

                    batch_rows = len(batch_df)
                    epoch_loss += float(loss.item()) * batch_rows
                    epoch_seen += batch_rows
                    postfix = f"{row_id_text(batch_df)}, epoch={epoch + 1}/{epochs}, loss={loss.item():.4f}"
                    update_progress(progress_bar, postfix=postfix, n=batch_rows)

                if epoch_seen:
                    self.history_.append({"epoch": epoch + 1, "loss": epoch_loss / epoch_seen})
        finally:
            if progress_bar is not None:
                progress_bar.close()

        if self.device_.type == "cuda":
            torch.cuda.synchronize(self.device_)
        return time.perf_counter() - start_time, "pytorch_cuda_batches" if self.device_.type == "cuda" else "pytorch_cpu_batches"

    def predict_batches(
        self,
        x_test: Any,
        test_df: pd.DataFrame,
        experiment_id: str,
        config: Dict[str, Any],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        require_torch()
        if self.model_ is None or self.classes_ is None:
            raise RuntimeError("Torch FFNN has not been fitted.")

        start_time = time.perf_counter()
        batch_size = max(1, int(self.model_config.get("batch_size", progress_batch_size(config))))
        progress_bar = make_progress_bar(len(test_df), f"{experiment_id} predict", config) if use_data_progress(config) else None
        prediction_chunks = []
        probability_chunks = []

        try:
            self.model_.eval()
            with torch.no_grad():
                for _, _, batch_df, x_batch, _ in iter_batches(
                    x_test,
                    test_df,
                    None,
                    batch_size,
                ):
                    features = sparse_matrix_to_tensor(x_batch, self.device_)
                    logits = self.model_(features)
                    probabilities = torch.softmax(logits, dim=1).detach().cpu().numpy()
                    predictions = self.classes_[np.argmax(probabilities, axis=1)]

                    probability_chunks.append(probabilities)
                    prediction_chunks.append(np.asarray(predictions))
                    update_progress(progress_bar, postfix=row_id_text(batch_df), n=len(batch_df))
        finally:
            if progress_bar is not None:
                progress_bar.close()

        if self.device_.type == "cuda":
            torch.cuda.synchronize(self.device_)

        y_pred = np.concatenate(prediction_chunks) if prediction_chunks else np.asarray([])
        probabilities = np.vstack(probability_chunks) if probability_chunks else np.empty((0, len(self.classes_)))
        return y_pred, probabilities, self.classes_, time.perf_counter() - start_time


def build_model(name: str, model_config: Dict[str, Any], random_seed: int):
    if name == "naive_bayes":
        return MultinomialNB(
            alpha=model_config.get("alpha", 1.0),
            fit_prior=model_config.get("fit_prior", True),
        )

    if name == "ffnn":
        return TorchFFNNClassifier(model_config=model_config, random_seed=random_seed)

    raise ValueError(f"Unsupported model: {name}")


def compute_metrics(
    y_true: Iterable[str],
    y_pred: Iterable[str],
    labels: List[str],
) -> Dict[str, float]:
    y_true = list(y_true)
    y_pred = list(y_pred)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_precision": precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
        "weighted_precision": precision_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0),
        "weighted_recall": recall_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0),
    }


def save_classification_report(
    y_true: Iterable[str],
    y_pred: Iterable[str],
    labels: List[str],
    path: Path,
) -> None:
    report = classification_report(
        list(y_true),
        list(y_pred),
        labels=labels,
        target_names=labels,
        output_dict=True,
        zero_division=0,
    )
    report_df = pd.DataFrame(report).transpose().reset_index().rename(columns={"index": "label"})
    save_csv(report_df, path)


def save_confusion_outputs(
    y_true: Iterable[str],
    y_pred: Iterable[str],
    labels: List[str],
    experiment_id: str,
    dirs: Dict[str, Path],
    config: Dict[str, Any],
) -> None:
    matrix = confusion_matrix(list(y_true), list(y_pred), labels=labels)
    matrix_df = pd.DataFrame(matrix, index=labels, columns=labels)
    save_csv(matrix_df.reset_index().rename(columns={"index": "true_label"}), dirs["confusion"] / f"{experiment_id}.csv")

    plt.figure(figsize=(6.5, 5.5))
    sns.heatmap(
        matrix_df,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        linewidths=0.5,
        linecolor="white",
    )
    plt.title(f"Confusion matrix: {experiment_id}")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.tight_layout()
    plt.savefig(
        dirs["confusion"] / f"{experiment_id}.{config['output']['plot_format']}",
        dpi=config["output"]["figure_dpi"],
    )
    plt.close()


def prediction_dataframe(
    test_df: pd.DataFrame,
    y_pred: np.ndarray,
    probabilities: Optional[np.ndarray],
    model_classes: Optional[np.ndarray],
    labels: List[str],
    config: Dict[str, Any],
) -> pd.DataFrame:
    text_col = config["data"]["text_column"]
    raw_text_col = config["data"].get("raw_text_column") or text_col
    label_col = config["data"]["label_column"]

    pred_df = pd.DataFrame(
        {
            "row_id": test_df["source_row_id"].tolist() if "source_row_id" in test_df.columns else test_df.index,
            "true_label": test_df[label_col].tolist(),
            "predicted_label": y_pred,
            "raw_text": test_df[raw_text_col].tolist() if raw_text_col in test_df.columns else test_df[text_col].tolist(),
            "model_text": test_df[text_col].tolist(),
        }
    )
    pred_df["correct"] = pred_df["true_label"] == pred_df["predicted_label"]

    if probabilities is not None and model_classes is not None:
        class_to_index = {label: idx for idx, label in enumerate(model_classes)}
        confidence = []
        for row_idx, pred_label in enumerate(y_pred):
            confidence.append(float(probabilities[row_idx, class_to_index[pred_label]]))
        pred_df["confidence"] = confidence

        for label in labels:
            if label in class_to_index:
                pred_df[f"proba_{label}"] = probabilities[:, class_to_index[label]]

    return pred_df


def save_error_analysis(
    pred_df: pd.DataFrame,
    experiment_id: str,
    dirs: Dict[str, Path],
    config: Dict[str, Any],
) -> Dict[str, Any]:
    errors = pred_df[~pred_df["correct"]].copy()
    if "confidence" in errors.columns:
        errors = errors.sort_values("confidence", ascending=False)

    max_examples = config["output"]["error_examples_per_experiment"]
    save_csv(errors.head(max_examples), dirs["errors"] / f"{experiment_id}_examples.csv")

    if errors.empty:
        summary = pd.DataFrame(columns=["true_label", "predicted_label", "count"])
    else:
        summary = (
            errors.groupby(["true_label", "predicted_label"])
            .size()
            .reset_index(name="count")
            .sort_values("count", ascending=False)
        )
    save_csv(summary, dirs["errors"] / f"{experiment_id}_summary.csv")

    top_confusions = summary.head(5).to_dict(orient="records")
    return {
        "error_count": int(len(errors)),
        "top_confusions": top_confusions,
    }


def save_model_if_requested(
    vectorizer: Any,
    model: Any,
    experiment_id: str,
    dirs: Dict[str, Path],
    config: Dict[str, Any],
) -> None:
    if not config["output"].get("save_models"):
        return

    if isinstance(model, TorchFFNNClassifier):
        model_path = dirs["models"] / f"{experiment_id}.pt"
        torch.save(
            {
                "state_dict": model.model_.state_dict() if model.model_ is not None else None,
                "classes": model.classes_,
                "model_config": model.model_config,
                "device": str(model.device_),
                "history": model.history_,
                "config": config,
            },
            model_path,
        )
        vectorizer_path = dirs["models"] / f"{experiment_id}_vectorizer.joblib"
        joblib.dump(vectorizer, vectorizer_path)
        return

    model_path = dirs["models"] / f"{experiment_id}.joblib"
    joblib.dump({"vectorizer": vectorizer, "model": model, "config": config}, model_path)


def fit_model(
    model: Any,
    model_name: str,
    model_config: Dict[str, Any],
    x_train: Any,
    y_train: np.ndarray,
    train_df: pd.DataFrame,
    labels: List[str],
    experiment_id: str,
    config: Dict[str, Any],
) -> Tuple[float, str]:
    start_time = time.perf_counter()

    if isinstance(model, TorchFFNNClassifier):
        return model.fit_batches(
            x_train=x_train,
            y_train=y_train,
            train_df=train_df,
            labels=labels,
            experiment_id=experiment_id,
            config=config,
        )

    if not use_data_progress(config) or not hasattr(model, "partial_fit"):
        model.fit(x_train, y_train)
        return time.perf_counter() - start_time, "fit"

    class_ids = np.arange(len(labels), dtype=int)
    batch_size = progress_batch_size(config)
    first_batch = True

    if model_name == "ffnn":
        epochs = max(1, int(model_config.get("max_iter", 1)))
        if getattr(model, "early_stopping", False):
            model.early_stopping = False
        total_steps = len(train_df) * epochs
        progress_bar = make_progress_bar(total_steps, f"{experiment_id} train", config)

        try:
            for epoch in range(epochs):
                for _, end, batch_df, x_batch, y_batch in iter_batches(
                    x_train,
                    train_df,
                    y_train,
                    batch_size,
                ):
                    adjust_ffnn_batch_size(model, model_name, model_config, len(batch_df))
                    if first_batch:
                        model.partial_fit(x_batch, y_batch, classes=class_ids)
                        first_batch = False
                    else:
                        model.partial_fit(x_batch, y_batch)
                    postfix = f"epoch={epoch + 1}/{epochs}, {row_id_text(batch_df)}"
                    update_progress(progress_bar, postfix=postfix, n=len(batch_df))
        finally:
            if progress_bar is not None:
                progress_bar.close()

        return time.perf_counter() - start_time, "partial_fit_batches"

    total_steps = len(train_df)
    progress_bar = make_progress_bar(total_steps, f"{experiment_id} train", config)

    try:
        for _, _, batch_df, x_batch, y_batch in iter_batches(
            x_train,
            train_df,
            y_train,
            batch_size,
        ):
            if first_batch:
                model.partial_fit(x_batch, y_batch, classes=class_ids)
                first_batch = False
            else:
                model.partial_fit(x_batch, y_batch)
            update_progress(progress_bar, postfix=row_id_text(batch_df), n=len(batch_df))
    finally:
        if progress_bar is not None:
            progress_bar.close()

    return time.perf_counter() - start_time, "partial_fit_batches"


def predict_model(
    model: Any,
    x_test: Any,
    test_df: pd.DataFrame,
    experiment_id: str,
    config: Dict[str, Any],
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray], float]:
    start_time = time.perf_counter()

    if isinstance(model, TorchFFNNClassifier):
        return model.predict_batches(
            x_test=x_test,
            test_df=test_df,
            experiment_id=experiment_id,
            config=config,
        )

    if not use_data_progress(config):
        y_pred_ids = model.predict(x_test)
        probabilities = None
        model_classes = None
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(x_test)
            model_classes = model.classes_
        return y_pred_ids, probabilities, model_classes, time.perf_counter() - start_time

    batch_size = progress_batch_size(config)
    progress_bar = make_progress_bar(len(test_df), f"{experiment_id} predict", config)
    predictions = []
    probability_chunks = []

    try:
        for _, _, batch_df, x_batch, _ in iter_batches(
            x_test,
            test_df,
            None,
            batch_size,
        ):
            if hasattr(model, "predict_proba"):
                batch_probabilities = model.predict_proba(x_batch)
                batch_predictions = model.classes_[np.argmax(batch_probabilities, axis=1)]
                probability_chunks.append(batch_probabilities)
            else:
                batch_predictions = model.predict(x_batch)
            predictions.append(np.asarray(batch_predictions))
            update_progress(progress_bar, postfix=row_id_text(batch_df), n=len(batch_df))
    finally:
        if progress_bar is not None:
            progress_bar.close()

    y_pred_ids = np.concatenate(predictions) if predictions else np.asarray([])
    probabilities = np.vstack(probability_chunks) if probability_chunks else None
    model_classes = model.classes_ if probability_chunks else None
    return y_pred_ids, probabilities, model_classes, time.perf_counter() - start_time


def run_single_experiment(
    setting: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    labels: List[str],
    vectorizer_name: str,
    vectorizer_config: Dict[str, Any],
    model_name: str,
    model_config: Dict[str, Any],
    dirs: Dict[str, Path],
    config: Dict[str, Any],
    progress_bar: Any = None,
) -> Dict[str, Any]:
    experiment_id = safe_name(f"{setting}_{vectorizer_name}_{model_name}")
    text_col = config["data"]["text_column"]
    label_col = config["data"]["label_column"]

    if should_log_experiment(config, progress_bar):
        LOGGER.info("Running experiment: %s", experiment_id)
    set_progress_stage(progress_bar, f"{experiment_id}: vectorizing")
    vectorizer = build_vectorizer(vectorizer_name, vectorizer_config)
    model = build_model(model_name, model_config, config["data"]["random_seed"])

    x_train_text = train_df[text_col].astype(str).tolist()
    y_train_labels = train_df[label_col].astype(str).tolist()
    x_test_text = test_df[text_col].astype(str).tolist()
    y_test = test_df[label_col].astype(str).tolist()
    label_to_id = {label: index for index, label in enumerate(labels)}
    id_to_label = {index: label for label, index in label_to_id.items()}
    y_train = np.asarray([label_to_id[label] for label in y_train_labels], dtype=int)

    start_time = time.perf_counter()
    x_train = vectorizer.fit_transform(x_train_text)
    x_test = vectorizer.transform(x_test_text)
    vectorize_seconds = time.perf_counter() - start_time

    set_progress_stage(progress_bar, f"{experiment_id}: training")
    train_seconds, training_mode = fit_model(
        model=model,
        model_name=model_name,
        model_config=model_config,
        x_train=x_train,
        y_train=y_train,
        train_df=train_df,
        labels=labels,
        experiment_id=experiment_id,
        config=config,
    )

    set_progress_stage(progress_bar, f"{experiment_id}: predicting")
    y_pred_ids, probabilities, model_classes, predict_seconds = predict_model(
        model=model,
        x_test=x_test,
        test_df=test_df,
        experiment_id=experiment_id,
        config=config,
    )
    y_pred = np.asarray([id_to_label[int(label_id)] for label_id in y_pred_ids])

    label_model_classes = None
    if model_classes is not None:
        label_model_classes = np.asarray([id_to_label[int(label_id)] for label_id in model_classes])

    set_progress_stage(progress_bar, f"{experiment_id}: saving")
    metrics = compute_metrics(y_test, y_pred, labels)
    metrics.update(
        {
            "experiment_id": experiment_id,
            "setting": setting,
            "representation": vectorizer_name,
            "model": model_name,
            "train_rows": len(train_df),
            "test_rows": len(test_df),
            "features": int(x_train.shape[1]),
            "class_handling": config["class_handling"].get("strategy", "none"),
            "framework": getattr(model, "framework_", "sklearn"),
            "device": str(getattr(model, "device_", "cpu")),
            "training_mode": training_mode,
            "vectorize_seconds": round(vectorize_seconds, 4),
            "train_seconds": round(train_seconds, 4),
            "predict_seconds": round(predict_seconds, 4),
        }
    )

    save_classification_report(
        y_test,
        y_pred,
        labels,
        dirs["reports"] / f"{experiment_id}.csv",
    )
    save_confusion_outputs(y_test, y_pred, labels, experiment_id, dirs, config)

    pred_df = prediction_dataframe(test_df, y_pred, probabilities, label_model_classes, labels, config)
    if config["output"].get("save_full_predictions"):
        save_csv(pred_df, dirs["predictions"] / f"{experiment_id}.csv")
    error_info = save_error_analysis(pred_df, experiment_id, dirs, config)
    metrics.update(error_info)

    save_model_if_requested(vectorizer, model, experiment_id, dirs, config)

    if should_log_experiment(config, progress_bar):
        LOGGER.info(
            "%s | accuracy=%.4f macro_f1=%.4f",
            experiment_id,
            metrics["accuracy"],
            metrics["macro_f1"],
        )
    update_progress(
        progress_bar,
        postfix=f"last acc={metrics['accuracy']:.4f}, f1={metrics['macro_f1']:.4f}",
    )
    return metrics


def setting_names(config: Dict[str, Any]) -> List[str]:
    names = []
    if config["task"].get("run_multiclass"):
        names.append("multiclass")
    if config["task"].get("run_binary"):
        names.append("binary")
    return names


def run_experiments(df: pd.DataFrame, config: Dict[str, Any], dirs: Dict[str, Path]) -> pd.DataFrame:
    vectorizers = get_enabled_items(config, "vectorizers")
    models = get_enabled_items(config, "models")

    if not vectorizers:
        raise ValueError("No vectorizers are enabled.")
    if not models:
        raise ValueError("No models are enabled.")

    all_metrics = []
    settings = setting_names(config)
    total_experiments = len(settings) * len(vectorizers) * len(models)
    progress_bar = None
    if progress_level(config) == "experiment":
        progress_bar = make_progress_bar(total_experiments, "Task 3 experiments", config)

    try:
        for setting in settings:
            setting_df, labels = make_dataset_for_setting(df, setting, config)
            train_df, test_df = split_dataset(setting_df, config)
            train_df = oversample_training_data(train_df, config)

            label_counts = {
                "setting": setting,
                "train_counts": train_df[config["data"]["label_column"]].value_counts().to_dict(),
                "test_counts": test_df[config["data"]["label_column"]].value_counts().to_dict(),
            }
            save_json(label_counts, dirs["root"] / f"{setting}_split_counts.json")

            for vectorizer_name, vectorizer_config in vectorizers.items():
                for model_name, model_config in models.items():
                    metrics = run_single_experiment(
                        setting=setting,
                        train_df=train_df,
                        test_df=test_df,
                        labels=labels,
                        vectorizer_name=vectorizer_name,
                        vectorizer_config=vectorizer_config,
                        model_name=model_name,
                        model_config=model_config,
                        dirs=dirs,
                        config=config,
                        progress_bar=progress_bar,
                    )
                    all_metrics.append(metrics)
    finally:
        if progress_bar is not None:
            progress_bar.close()

    metrics_df = pd.DataFrame(all_metrics)
    ordered_cols = [
        "experiment_id",
        "setting",
        "representation",
        "model",
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "weighted_precision",
        "weighted_recall",
        "weighted_f1",
        "error_count",
        "features",
        "train_rows",
        "test_rows",
        "class_handling",
        "framework",
        "device",
        "training_mode",
        "vectorize_seconds",
        "train_seconds",
        "predict_seconds",
        "top_confusions",
    ]
    metrics_df = metrics_df[[col for col in ordered_cols if col in metrics_df.columns]]
    save_csv(metrics_df, dirs["root"] / "metrics_summary.csv")
    return metrics_df


def generate_report_notes(metrics_df: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
    notes = []
    if metrics_df.empty:
        return {"notes": ["No experiments were completed."]}

    for setting in sorted(metrics_df["setting"].unique()):
        subset = metrics_df[metrics_df["setting"] == setting]
        best = subset.sort_values("macro_f1", ascending=False).iloc[0]
        notes.append(
            f"For {setting}, the best macro F1 is {best['macro_f1']:.4f} "
            f"from {best['representation']} + {best['model']}."
        )

    multiclass = metrics_df[metrics_df["setting"] == "multiclass"]
    binary = metrics_df[metrics_df["setting"] == "binary"]
    if not multiclass.empty and not binary.empty:
        notes.append(
            "Compare binary and multiclass results carefully: binary classification usually removes "
            "ambiguous neutral examples, so it may show higher scores and different error patterns."
        )

    notes.append(
        "For error analysis, inspect high-confidence incorrect predictions and the most frequent "
        "true/predicted label pairs saved in the errors directory."
    )
    notes.append(
        "When writing the report, discuss both performance and behavior: representation choice, "
        "class balance, ambiguous tweets, missing context, and preprocessing effects."
    )

    return {
        "notes": notes,
        "metrics_file": str(resolve_project_path(config["output"]["output_dir"]) / "metrics_summary.csv"),
    }


def main() -> None:
    setup_logging()
    args = parse_args()
    config = load_config(args)
    dirs = ensure_output_dirs(config)
    save_json(config, dirs["root"] / "config_used.json")

    sns.set_theme(style="whitegrid")
    df = load_dataset(config)
    metrics_df = run_experiments(df, config, dirs)
    report_notes = generate_report_notes(metrics_df, config)
    save_json(report_notes, dirs["root"] / "report_notes_task3.json")

    LOGGER.info("Task 3 completed. Main output: %s", dirs["root"].resolve())


if __name__ == "__main__":
    main()
