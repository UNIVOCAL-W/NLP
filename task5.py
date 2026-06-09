"""
Task 5: Fine-tune a pre-trained transformer for tweet sentiment.

This script fine-tunes DistilBERT in both:
    - multiclass classification: negative / neutral / positive
    - binary classification: negative / positive, with neutral removed

It uses raw tweet text, an 80/20 train-test split, max length 128, batch size 8,
3 epochs, learning rate 2e-5, weight decay 0.01, and warmup ratio 0.1 by default.

Usage:
    python task5.py
    python task5.py --sample-size 3000 --epochs 1
    python task5.py --config task5_config.json
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None


DEFAULT_CONFIG_FILE = "task5_config.json"
SCRIPT_DIR = Path(__file__).resolve().parent
LOGGER = logging.getLogger("task5")

DEFAULT_CONFIG: Dict[str, Any] = {
    "data": {
        "csv_path": "outputs/tasks1_2/preprocessed_tweets.csv",
        "text_column": "text",
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
    "model": {
        "model_name": "distilbert-base-uncased",
        "max_length": 128,
        "batch_size": 8,
        "epochs": 3,
        "learning_rate": 2e-5,
        "weight_decay": 0.01,
        "warmup_ratio": 0.1,
        "device": "auto",
    },
    "output": {
        "output_dir": "outputs/task5",
        "save_model": False,
        "plot_format": "png",
        "figure_dpi": 160,
        "error_examples_per_setting": 50,
        "save_full_predictions": True,
    },
    "progress": {
        "enabled": True,
    },
}


class TweetDataset(Dataset):
    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        tokenizer: Any,
        max_length: int,
    ) -> None:
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        encoded = self.tokenizer(
            self.texts[index],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {key: value.squeeze(0) for key, value in encoded.items()}
        item["labels"] = torch.tensor(self.labels[index], dtype=torch.long)
        return item


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Task 5 transformer experiments.")
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG_FILE,
        help=f"JSON config that overrides defaults. Defaults to {DEFAULT_CONFIG_FILE}.",
    )
    parser.add_argument("--output-dir", default=None, help="Optional output directory override.")
    parser.add_argument("--sample-size", type=int, default=None, help="Optional sample size for quick runs.")
    parser.add_argument("--epochs", type=int, default=None, help="Optional epoch override.")
    parser.add_argument("--batch-size", type=int, default=None, help="Optional batch size override.")
    parser.add_argument("--learning-rate", type=float, default=None, help="Optional learning rate override.")
    parser.add_argument("--multiclass-only", action="store_true", help="Run only multiclass classification.")
    parser.add_argument("--binary-only", action="store_true", help="Run only binary classification.")
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


def load_config(args: argparse.Namespace) -> Dict[str, Any]:
    config = deepcopy(DEFAULT_CONFIG)
    config_path = resolve_project_path(args.config)

    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as file:
            deep_update(config, json.load(file))
    elif args.config != DEFAULT_CONFIG_FILE:
        raise FileNotFoundError(f"Config file not found: {args.config}")

    if args.output_dir:
        config["output"]["output_dir"] = args.output_dir
    if args.sample_size is not None:
        config["data"]["sample_size"] = args.sample_size
    if args.epochs is not None:
        config["model"]["epochs"] = args.epochs
    if args.batch_size is not None:
        config["model"]["batch_size"] = args.batch_size
    if args.learning_rate is not None:
        config["model"]["learning_rate"] = args.learning_rate
    if args.multiclass_only and args.binary_only:
        raise ValueError("Use either --multiclass-only or --binary-only, not both.")
    if args.multiclass_only:
        config["task"]["run_binary"] = False
    if args.binary_only:
        config["task"]["run_multiclass"] = False

    return config


def set_random_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(config: Dict[str, Any]) -> torch.device:
    requested = str(config["model"].get("device", "auto")).lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested.startswith("cuda") and not torch.cuda.is_available():
        LOGGER.warning("CUDA was requested but is unavailable. Falling back to CPU.")
        return torch.device("cpu")
    return torch.device(requested)


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


def load_dataset(config: Dict[str, Any]) -> pd.DataFrame:
    data_config = config["data"]
    csv_path = resolve_project_path(data_config["csv_path"])
    text_col = data_config["text_column"]
    label_col = data_config["label_column"]

    LOGGER.info("Loading data from %s", csv_path)
    df = pd.read_csv(csv_path, encoding=data_config["encoding"])
    missing = [column for column in [text_col, label_col] if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {csv_path}: {missing}")

    df = df[[text_col, label_col]].copy()
    df[text_col] = df[text_col].fillna("").astype(str)
    df[label_col] = df[label_col].fillna("").astype(str).str.strip().str.lower()
    df = df[df[text_col].str.strip().astype(bool)]
    df = df[df[label_col].str.strip().astype(bool)]

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


def make_dataset_for_setting(
    df: pd.DataFrame,
    setting: str,
    config: Dict[str, Any],
) -> Tuple[pd.DataFrame, List[str]]:
    label_col = config["data"]["label_column"]
    task_config = config["task"]

    if setting == "binary":
        setting_df = df[df[label_col] != task_config["binary_drop_label"]].copy()
        labels = [label for label in task_config["binary_label_order"] if label in set(setting_df[label_col])]
    elif setting == "multiclass":
        setting_df = df.copy()
        labels = [label for label in task_config["multiclass_label_order"] if label in set(setting_df[label_col])]
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


def make_data_loader(
    df: pd.DataFrame,
    labels: List[str],
    tokenizer: Any,
    config: Dict[str, Any],
    shuffle: bool,
) -> DataLoader:
    text_col = config["data"]["text_column"]
    label_col = config["data"]["label_column"]
    label_to_id = {label: index for index, label in enumerate(labels)}
    dataset = TweetDataset(
        texts=df[text_col].astype(str).tolist(),
        labels=[label_to_id[label] for label in df[label_col].astype(str).tolist()],
        tokenizer=tokenizer,
        max_length=int(config["model"]["max_length"]),
    )
    return DataLoader(
        dataset,
        batch_size=int(config["model"]["batch_size"]),
        shuffle=shuffle,
    )


def progress_bar(iterable: Iterable[Any], desc: str, config: Dict[str, Any]) -> Iterable[Any]:
    if tqdm is not None and config.get("progress", {}).get("enabled", True):
        return tqdm(iterable, desc=desc, leave=True)
    return iterable


def train_one_epoch(
    model: Any,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    device: torch.device,
    config: Dict[str, Any],
    desc: str,
) -> float:
    model.train()
    losses = []
    for batch in progress_bar(loader, desc, config):
        batch = {key: value.to(device) for key, value in batch.items()}
        optimizer.zero_grad()
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses)) if losses else 0.0


def predict(
    model: Any,
    loader: DataLoader,
    device: torch.device,
    config: Dict[str, Any],
    desc: str,
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    predictions = []
    probabilities = []

    with torch.no_grad():
        for batch in progress_bar(loader, desc, config):
            batch = {key: value.to(device) for key, value in batch.items()}
            labels = batch.pop("labels")
            _ = labels
            outputs = model(**batch)
            probs = torch.softmax(outputs.logits, dim=1)
            pred_ids = torch.argmax(probs, dim=1)
            predictions.append(pred_ids.detach().cpu().numpy())
            probabilities.append(probs.detach().cpu().numpy())

    y_pred = np.concatenate(predictions) if predictions else np.asarray([])
    y_proba = np.vstack(probabilities) if probabilities else np.asarray([])
    return y_pred, y_proba


def compute_metrics(y_true: List[str], y_pred: List[str], labels: List[str]) -> Dict[str, float]:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_precision": precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
        "weighted_precision": precision_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0),
        "weighted_recall": recall_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0),
    }


def save_classification_report(y_true: List[str], y_pred: List[str], labels: List[str], path: Path) -> None:
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=labels,
        output_dict=True,
        zero_division=0,
    )
    report_df = pd.DataFrame(report).transpose().reset_index().rename(columns={"index": "label"})
    save_csv(report_df, path)


def save_confusion_outputs(
    y_true: List[str],
    y_pred: List[str],
    labels: List[str],
    experiment_id: str,
    dirs: Dict[str, Path],
    config: Dict[str, Any],
) -> None:
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
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
    y_pred: List[str],
    probabilities: np.ndarray,
    labels: List[str],
    config: Dict[str, Any],
) -> pd.DataFrame:
    text_col = config["data"]["text_column"]
    label_col = config["data"]["label_column"]
    pred_df = pd.DataFrame(
        {
            "row_id": test_df["source_row_id"].tolist(),
            "true_label": test_df[label_col].tolist(),
            "predicted_label": y_pred,
            "text": test_df[text_col].tolist(),
        }
    )
    pred_df["correct"] = pred_df["true_label"] == pred_df["predicted_label"]

    if probabilities.size:
        pred_ids = [labels.index(label) for label in y_pred]
        pred_df["confidence"] = [float(probabilities[index, pred_id]) for index, pred_id in enumerate(pred_ids)]
        for label_index, label in enumerate(labels):
            pred_df[f"proba_{label}"] = probabilities[:, label_index]

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

    max_examples = int(config["output"]["error_examples_per_setting"])
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

    return {
        "error_count": int(len(errors)),
        "top_confusions": summary.head(5).to_dict(orient="records"),
    }


def save_model_if_requested(model: Any, tokenizer: Any, experiment_id: str, dirs: Dict[str, Path], config: Dict[str, Any]) -> None:
    if not config["output"].get("save_model"):
        return
    model_dir = dirs["models"] / experiment_id
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)


def run_single_setting(
    setting: str,
    df: pd.DataFrame,
    tokenizer: Any,
    device: torch.device,
    dirs: Dict[str, Path],
    config: Dict[str, Any],
) -> Dict[str, Any]:
    experiment_id = f"{setting}_distilbert"
    setting_df, labels = make_dataset_for_setting(df, setting, config)
    train_df, test_df = split_dataset(setting_df, config)
    save_json(
        {
            "setting": setting,
            "train_counts": train_df[config["data"]["label_column"]].value_counts().to_dict(),
            "test_counts": test_df[config["data"]["label_column"]].value_counts().to_dict(),
        },
        dirs["root"] / f"{setting}_split_counts.json",
    )

    id_to_label = {index: label for index, label in enumerate(labels)}
    label_to_id = {label: index for index, label in id_to_label.items()}

    LOGGER.info("Loading model %s for %s", config["model"]["model_name"], setting)
    model = AutoModelForSequenceClassification.from_pretrained(
        config["model"]["model_name"],
        num_labels=len(labels),
        id2label=id_to_label,
        label2id=label_to_id,
    )
    model.to(device)

    train_loader = make_data_loader(train_df, labels, tokenizer, config, shuffle=True)
    test_loader = make_data_loader(test_df, labels, tokenizer, config, shuffle=False)

    total_steps = len(train_loader) * int(config["model"]["epochs"])
    warmup_steps = int(total_steps * float(config["model"]["warmup_ratio"]))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["model"]["learning_rate"]),
        weight_decay=float(config["model"]["weight_decay"]),
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    start_time = time.perf_counter()
    epoch_losses = []
    for epoch in range(int(config["model"]["epochs"])):
        loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            device,
            config,
            desc=f"{experiment_id} epoch {epoch + 1}/{config['model']['epochs']}",
        )
        epoch_losses.append({"epoch": epoch + 1, "train_loss": loss})
        LOGGER.info("%s | epoch=%d train_loss=%.4f", experiment_id, epoch + 1, loss)

    train_seconds = time.perf_counter() - start_time
    save_csv(pd.DataFrame(epoch_losses), dirs["root"] / f"{experiment_id}_training_history.csv")

    predict_start = time.perf_counter()
    y_pred_ids, probabilities = predict(model, test_loader, device, config, desc=f"{experiment_id} predict")
    predict_seconds = time.perf_counter() - predict_start

    y_true = test_df[config["data"]["label_column"]].astype(str).tolist()
    y_pred = [id_to_label[int(index)] for index in y_pred_ids]
    metrics = compute_metrics(y_true, y_pred, labels)
    metrics.update(
        {
            "experiment_id": experiment_id,
            "setting": setting,
            "model": config["model"]["model_name"],
            "train_rows": len(train_df),
            "test_rows": len(test_df),
            "epochs": int(config["model"]["epochs"]),
            "max_length": int(config["model"]["max_length"]),
            "batch_size": int(config["model"]["batch_size"]),
            "learning_rate": float(config["model"]["learning_rate"]),
            "weight_decay": float(config["model"]["weight_decay"]),
            "warmup_ratio": float(config["model"]["warmup_ratio"]),
            "device": str(device),
            "train_seconds": round(train_seconds, 4),
            "predict_seconds": round(predict_seconds, 4),
        }
    )

    save_classification_report(y_true, y_pred, labels, dirs["reports"] / f"{experiment_id}.csv")
    save_confusion_outputs(y_true, y_pred, labels, experiment_id, dirs, config)

    pred_df = prediction_dataframe(test_df, y_pred, probabilities, labels, config)
    if config["output"].get("save_full_predictions"):
        save_csv(pred_df, dirs["predictions"] / f"{experiment_id}.csv")
    metrics.update(save_error_analysis(pred_df, experiment_id, dirs, config))
    save_model_if_requested(model, tokenizer, experiment_id, dirs, config)

    LOGGER.info(
        "%s | accuracy=%.4f macro_f1=%.4f",
        experiment_id,
        metrics["accuracy"],
        metrics["macro_f1"],
    )
    return metrics


def setting_names(config: Dict[str, Any]) -> List[str]:
    settings = []
    if config["task"].get("run_multiclass"):
        settings.append("multiclass")
    if config["task"].get("run_binary"):
        settings.append("binary")
    return settings


def generate_report_notes(metrics_df: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
    notes = [
        (
            f"DistilBERT was fine-tuned on raw tweet text with max length {config['model']['max_length']}, "
            f"batch size {config['model']['batch_size']}, {config['model']['epochs']} epochs, learning rate "
            f"{config['model']['learning_rate']}, weight decay {config['model']['weight_decay']}, and warmup ratio "
            f"{config['model']['warmup_ratio']}."
        ),
        "Compare the transformer results with Task 3 classical models in terms of performance, error patterns, and computational cost.",
    ]

    for setting in sorted(metrics_df["setting"].unique()):
        row = metrics_df[metrics_df["setting"] == setting].iloc[0]
        notes.append(
            f"For {setting}, DistilBERT achieved accuracy={row['accuracy']:.4f} and macro F1={row['macro_f1']:.4f}."
        )

    notes.append(
        "Use the saved error examples to discuss ambiguity, missing tweet context, neutral-class difficulty, and cases where the model is confidently wrong."
    )
    return {
        "notes": notes,
        "metrics_file": str(resolve_project_path(config["output"]["output_dir"]) / "metrics_summary.csv"),
    }


def main() -> None:
    setup_logging()
    args = parse_args()
    config = load_config(args)
    set_random_seeds(int(config["data"]["random_seed"]))
    dirs = ensure_output_dirs(config)
    save_json(config, dirs["root"] / "config_used.json")

    sns.set_theme(style="whitegrid")
    device = resolve_device(config)
    LOGGER.info("Using device: %s", device)

    df = load_dataset(config)
    LOGGER.info("Loading tokenizer: %s", config["model"]["model_name"])
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["model_name"])

    all_metrics = []
    for setting in setting_names(config):
        all_metrics.append(run_single_setting(setting, df, tokenizer, device, dirs, config))

    metrics_df = pd.DataFrame(all_metrics)
    ordered_cols = [
        "experiment_id",
        "setting",
        "model",
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "weighted_precision",
        "weighted_recall",
        "weighted_f1",
        "error_count",
        "train_rows",
        "test_rows",
        "epochs",
        "max_length",
        "batch_size",
        "learning_rate",
        "weight_decay",
        "warmup_ratio",
        "device",
        "train_seconds",
        "predict_seconds",
        "top_confusions",
    ]
    metrics_df = metrics_df[[column for column in ordered_cols if column in metrics_df.columns]]
    save_csv(metrics_df, dirs["root"] / "metrics_summary.csv")
    save_json(generate_report_notes(metrics_df, config), dirs["root"] / "report_notes_task5.json")
    LOGGER.info("Task 5 completed. Main output: %s", dirs["root"].resolve())


if __name__ == "__main__":
    main()
