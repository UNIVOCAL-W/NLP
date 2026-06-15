import json
import random
import time
from pathlib import Path

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
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None


CONFIG_FILE = "task5_config.json"

TEXT_COL = "text"
LABEL_COL = "sentiment"
MULTICLASS_LABELS = ["negative", "neutral", "positive"]
BINARY_LABELS = ["negative", "positive"]

MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 128
BATCH_SIZE = 8
EPOCHS = 3
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
TEST_SIZE = 0.2
RANDOM_SEED = 42

ERROR_EXAMPLES = 50
PLOT_FORMAT = "png"
DPI = 160


class TweetDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
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


def load_config():
    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def save_csv(df, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def save_json(data, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def set_seed():
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def progress_bar(items, desc):
    if tqdm is None:
        return items
    return tqdm(items, desc=desc, leave=True)


def make_output_dirs(output_dir):
    root = Path(output_dir)
    dirs = {
        "root": root,
        "reports": root / "classification_reports",
        "confusion": root / "confusion_matrices",
        "errors": root / "errors",
    }
    for folder in dirs.values():
        folder.mkdir(parents=True, exist_ok=True)
    return dirs


def load_data(csv_path):
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    df = df[[TEXT_COL, LABEL_COL]].copy()
    df[TEXT_COL] = df[TEXT_COL].fillna("").astype(str)
    df[LABEL_COL] = df[LABEL_COL].fillna("").astype(str).str.strip().str.lower()
    df = df[df[TEXT_COL].str.strip().astype(bool)]
    df = df[df[LABEL_COL].str.strip().astype(bool)]
    df = df.reset_index(drop=True)
    df.insert(0, "source_row_id", df.index)
    return df


def split_for_setting(df, setting):
    if setting == "binary":
        labels = BINARY_LABELS
        data = df[df[LABEL_COL] != "neutral"].copy()
    else:
        labels = MULTICLASS_LABELS
        data = df.copy()

    data = data[data[LABEL_COL].isin(labels)].reset_index(drop=True)
    train_df, test_df = train_test_split(
        data,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=data[LABEL_COL],
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True), labels


def make_loader(df, labels, tokenizer, shuffle):
    label_to_id = {label: i for i, label in enumerate(labels)}
    dataset = TweetDataset(
        texts=df[TEXT_COL].astype(str).tolist(),
        labels=[label_to_id[label] for label in df[LABEL_COL].astype(str).tolist()],
        tokenizer=tokenizer,
        max_length=MAX_LENGTH,
    )
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=shuffle)


def train_one_epoch(model, loader, optimizer, scheduler, device, description):
    model.train()
    losses = []

    for batch in progress_bar(loader, description):
        batch = {key: value.to(device) for key, value in batch.items()}

        optimizer.zero_grad(set_to_none=True)
        output = model(**batch)
        loss = output.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        losses.append(float(loss.detach().cpu()))

    return float(np.mean(losses)) if losses else 0.0


def predict(model, loader, labels, device, description):
    model.eval()
    predictions = []
    probabilities = []

    with torch.no_grad():
        for batch in progress_bar(loader, description):
            batch = {key: value.to(device) for key, value in batch.items()}
            batch.pop("labels")

            output = model(**batch)
            probs = torch.softmax(output.logits, dim=1)
            pred_ids = torch.argmax(probs, dim=1)

            predictions.extend(labels[i] for i in pred_ids.detach().cpu().numpy())
            probabilities.append(probs.detach().cpu().numpy())

    probabilities = np.vstack(probabilities) if probabilities else np.empty((0, len(labels)))
    return predictions, probabilities


def calculate_metrics(y_true, y_pred, labels):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_precision": precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
        "weighted_precision": precision_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0),
        "weighted_recall": recall_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0),
    }


def save_classification_report(y_true, y_pred, labels, path):
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=labels,
        output_dict=True,
        zero_division=0,
    )
    table = pd.DataFrame(report).transpose().reset_index().rename(columns={"index": "label"})
    save_csv(table, path)


def save_confusion_matrix(y_true, y_pred, labels, experiment_id, dirs):
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
    plt.savefig(dirs["confusion"] / f"{experiment_id}.{PLOT_FORMAT}", dpi=DPI)
    plt.close()


def make_prediction_table(test_df, y_pred, probabilities, labels):
    table = pd.DataFrame(
        {
            "row_id": test_df["source_row_id"].tolist(),
            "true_label": test_df[LABEL_COL].tolist(),
            "predicted_label": y_pred,
            "text": test_df[TEXT_COL].tolist(),
        }
    )
    table["correct"] = table["true_label"] == table["predicted_label"]

    pred_ids = [labels.index(label) for label in y_pred]
    table["confidence"] = [float(probabilities[i, pred_id]) for i, pred_id in enumerate(pred_ids)]
    for i, label in enumerate(labels):
        table[f"proba_{label}"] = probabilities[:, i]

    return table


def save_error_analysis(predictions, experiment_id, dirs):
    errors = predictions[~predictions["correct"]].copy()
    errors = errors.sort_values("confidence", ascending=False)
    save_csv(errors.head(ERROR_EXAMPLES), dirs["errors"] / f"{experiment_id}_examples.csv")

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


def train_and_evaluate(setting, df, tokenizer, dirs, device):
    experiment_id = f"{setting}_distilbert"
    train_df, test_df, labels = split_for_setting(df, setting)

    save_json(
        {
            "setting": setting,
            "train_counts": train_df[LABEL_COL].value_counts().to_dict(),
            "test_counts": test_df[LABEL_COL].value_counts().to_dict(),
        },
        dirs["root"] / f"{setting}_split_counts.json",
    )

    id_to_label = {i: label for i, label in enumerate(labels)}
    label_to_id = {label: i for i, label in id_to_label.items()}

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(labels),
        id2label=id_to_label,
        label2id=label_to_id,
    )
    model.to(device)

    train_loader = make_loader(train_df, labels, tokenizer, shuffle=True)
    test_loader = make_loader(test_df, labels, tokenizer, shuffle=False)

    total_steps = len(train_loader) * EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    print(f"\nRunning {experiment_id}")
    start = time.perf_counter()
    history = []
    for epoch in range(EPOCHS):
        loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            device,
            f"{experiment_id} epoch {epoch + 1}/{EPOCHS}",
        )
        history.append({"epoch": epoch + 1, "train_loss": loss})
        print(f"{experiment_id} epoch {epoch + 1}: loss={loss:.4f}")

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    train_seconds = time.perf_counter() - start
    save_csv(pd.DataFrame(history), dirs["root"] / f"{experiment_id}_training_history.csv")

    start = time.perf_counter()
    y_pred, probabilities = predict(model, test_loader, labels, device, f"{experiment_id} predict")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    predict_seconds = time.perf_counter() - start

    y_true = test_df[LABEL_COL].astype(str).tolist()
    metrics = calculate_metrics(y_true, y_pred, labels)
    metrics.update(
        {
            "experiment_id": experiment_id,
            "setting": setting,
            "model": MODEL_NAME,
            "train_rows": len(train_df),
            "test_rows": len(test_df),
            "epochs": EPOCHS,
            "max_length": MAX_LENGTH,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "warmup_ratio": WARMUP_RATIO,
            "device": str(device),
            "train_seconds": round(train_seconds, 4),
            "predict_seconds": round(predict_seconds, 4),
        }
    )

    save_classification_report(y_true, y_pred, labels, dirs["reports"] / f"{experiment_id}.csv")
    save_confusion_matrix(y_true, y_pred, labels, experiment_id, dirs)

    predictions = make_prediction_table(test_df, y_pred, probabilities, labels)
    metrics.update(save_error_analysis(predictions, experiment_id, dirs))

    print(f"{experiment_id}: accuracy={metrics['accuracy']:.4f}, macro_f1={metrics['macro_f1']:.4f}")
    return metrics


def main():
    sns.set_theme(style="whitegrid")
    set_seed()

    config = load_config()
    dirs = make_output_dirs(config["output_dir"])

    device = get_device()
    print(f"Using device: {device}")

    df = load_data(config["input_csv"])
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    metrics = []
    for setting in ["multiclass", "binary"]:
        metrics.append(train_and_evaluate(setting, df, tokenizer, dirs, device))

    metrics_df = pd.DataFrame(metrics)
    columns = [
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
    metrics_df = metrics_df[columns]
    save_csv(metrics_df, dirs["root"] / "metrics_summary.csv")

    print(f"Done. Outputs saved to {Path(config['output_dir']).resolve()}")


if __name__ == "__main__":
    main()
