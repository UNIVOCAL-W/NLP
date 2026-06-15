import json
import re
from pathlib import Path

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


import torch
from torch import nn


from tqdm.auto import tqdm



CONFIG_FILE = "task3_config.json"

TEXT_COL = "processed_text"
RAW_TEXT_COL = "text"
LABEL_COL = "sentiment"
RANDOM_SEED = 42
TEST_SIZE = 0.2

MULTICLASS_LABELS = ["negative", "neutral", "positive"]
BINARY_LABELS = ["negative", "positive"]
TOKEN_PATTERN = r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+(?:\.\d+)?|[!?]+"

HIDDEN_SIZE = 128
BATCH_SIZE = 128
EPOCHS = 25
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001

ERROR_EXAMPLES = 50
PLOT_FORMAT = "png"
DPI = 160


def load_config():
    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def save_csv(df, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def safe_name(text):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_").lower()


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
        "predictions": root / "predictions",
        "errors": root / "errors",
    }
    for folder in dirs.values():
        folder.mkdir(parents=True, exist_ok=True)
    return dirs


def load_data(csv_path):
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    df = df[[TEXT_COL, RAW_TEXT_COL, LABEL_COL]].copy()
    df[TEXT_COL] = df[TEXT_COL].fillna("").astype(str)
    df[RAW_TEXT_COL] = df[RAW_TEXT_COL].fillna("").astype(str)
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


def build_vectorizer(name):
    if name == "bow":
        return CountVectorizer(
            lowercase=False,
            token_pattern=TOKEN_PATTERN,
            ngram_range=(1, 1),
            min_df=2,
            max_df=1.0,
            max_features=20000,
            binary=False,
        )

    return TfidfVectorizer(
        lowercase=False,
        token_pattern=TOKEN_PATTERN,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        max_features=30000,
        sublinear_tf=True,
        use_idf=True,
        norm="l2",
    )


def resolve_device():
    if torch is None or nn is None:
        raise ImportError("PyTorch is required for the FFNN part of Task 3.")

    if torch.cuda.is_available():
        return torch.device("cuda")

    print("CUDA is not available in this environment. FFNN will use CPU.")
    return torch.device("cpu")


class FeedForwardNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, output_dim),
        )

    def forward(self, x):
        return self.layers(x)


def matrix_to_tensor(matrix, device):
    if hasattr(matrix, "toarray"):
        array = matrix.toarray()
    else:
        array = np.asarray(matrix)
    array = array.astype(np.float32, copy=False)
    return torch.as_tensor(array, dtype=torch.float32, device=device)


def train_ffnn(x_train, train_labels, labels, experiment_id, device):
    torch.manual_seed(RANDOM_SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(RANDOM_SEED)

    label_to_id = {label: i for i, label in enumerate(labels)}
    y_train = np.asarray([label_to_id[label] for label in train_labels], dtype=np.int64)

    model = FeedForwardNN(input_dim=int(x_train.shape[1]), output_dim=len(labels)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.CrossEntropyLoss()
    rng = np.random.default_rng(RANDOM_SEED)

    for epoch in range(EPOCHS):
        model.train()
        order = rng.permutation(len(y_train))
        epoch_loss = 0.0
        seen = 0

        batches = range(0, len(order), BATCH_SIZE)
        iterator = progress_bar(batches, f"{experiment_id} epoch {epoch + 1}/{EPOCHS}")
        for start in iterator:
            batch_ids = order[start : start + BATCH_SIZE]
            features = matrix_to_tensor(x_train[batch_ids], device)
            targets = torch.as_tensor(y_train[batch_ids], dtype=torch.long, device=device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(features)
            loss = loss_fn(logits, targets)
            loss.backward()
            optimizer.step()

            batch_size = len(batch_ids)
            epoch_loss += float(loss.item()) * batch_size
            seen += batch_size
            if hasattr(iterator, "set_postfix"):
                iterator.set_postfix(loss=f"{loss.item():.4f}")

        if seen:
            print(f"{experiment_id} epoch {epoch + 1}: loss={epoch_loss / seen:.4f}")

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return model


def predict_ffnn(model, x_test, labels, experiment_id, device):
    model.eval()
    predictions = []
    probabilities = []

    batches = range(0, x_test.shape[0], BATCH_SIZE)
    iterator = progress_bar(batches, f"{experiment_id} predict")

    with torch.no_grad():
        for start in iterator:
            end = min(start + BATCH_SIZE, x_test.shape[0])
            features = matrix_to_tensor(x_test[start:end], device)
            probs = torch.softmax(model(features), dim=1).detach().cpu().numpy()
            pred_ids = np.argmax(probs, axis=1)

            predictions.extend(labels[i] for i in pred_ids)
            probabilities.append(probs)

    if device.type == "cuda":
        torch.cuda.synchronize(device)

    probabilities = np.vstack(probabilities) if probabilities else np.empty((0, len(labels)))
    return np.asarray(predictions), probabilities, np.asarray(labels)


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


def make_prediction_table(test_df, y_pred, probabilities, classes, labels):
    table = pd.DataFrame(
        {
            "row_id": test_df["source_row_id"].tolist(),
            "true_label": test_df[LABEL_COL].tolist(),
            "predicted_label": y_pred,
            "raw_text": test_df[RAW_TEXT_COL].tolist(),
            "model_text": test_df[TEXT_COL].tolist(),
        }
    )
    table["correct"] = table["true_label"] == table["predicted_label"]

    if probabilities is not None:
        class_to_index = {label: i for i, label in enumerate(classes)}
        table["confidence"] = [
            float(probabilities[i, class_to_index[pred]]) for i, pred in enumerate(y_pred)
        ]
        for label in labels:
            if label in class_to_index:
                table[f"proba_{label}"] = probabilities[:, class_to_index[label]]

    return table


def save_error_analysis(predictions, experiment_id, dirs):
    errors = predictions[~predictions["correct"]].copy()
    if "confidence" in errors.columns:
        errors = errors.sort_values("confidence", ascending=False)
    save_csv(errors.head(ERROR_EXAMPLES), dirs["errors"] / f"{experiment_id}_examples.csv")

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


def run_one_experiment(setting, train_df, test_df, labels, vectorizer_name, model_name, dirs, device):
    experiment_id = safe_name(f"{setting}_{vectorizer_name}_{model_name}")
    print(f"\nRunning {experiment_id}")

    vectorizer = build_vectorizer(vectorizer_name)
    x_train = vectorizer.fit_transform(train_df[TEXT_COL].astype(str).tolist())
    x_test = vectorizer.transform(test_df[TEXT_COL].astype(str).tolist())

    train_labels = train_df[LABEL_COL].astype(str).tolist()
    if model_name == "naive_bayes":
        model = MultinomialNB(alpha=1.0, fit_prior=True)

        model.fit(x_train, train_labels)

        y_pred = model.predict(x_test)
        probabilities = model.predict_proba(x_test)
        model_classes = model.classes_
        framework = "sklearn"
        device_name = "cpu"
        training_mode = "fit"
    else:
        model = train_ffnn(x_train, train_labels, labels, experiment_id, device)
        y_pred, probabilities, model_classes = predict_ffnn(
            model, x_test, labels, experiment_id, device
        )
        framework = "pytorch"
        device_name = str(device)
        training_mode = "pytorch_cuda_batches" if device.type == "cuda" else "pytorch_cpu_batches"

    y_true = test_df[LABEL_COL].astype(str).tolist()
    metrics = calculate_metrics(y_true, y_pred, labels)
    metrics.update(
        {
            "experiment_id": experiment_id,
            "setting": setting,
            "representation": vectorizer_name,
            "model": model_name,
            "train_rows": len(train_df),
            "test_rows": len(test_df),
            "features": int(x_train.shape[1]),
            "framework": framework,
            "device": device_name,
            "training_mode": training_mode,
        }
    )

    save_classification_report(y_true, y_pred, labels, dirs["reports"] / f"{experiment_id}.csv")
    save_confusion_matrix(y_true, y_pred, labels, experiment_id, dirs)

    predictions = make_prediction_table(test_df, y_pred, probabilities, model_classes, labels)
    save_csv(predictions, dirs["predictions"] / f"{experiment_id}.csv")
    metrics.update(save_error_analysis(predictions, experiment_id, dirs))

    print(
        f"{experiment_id}: accuracy={metrics['accuracy']:.4f}, macro_f1={metrics['macro_f1']:.4f} "
        f"| confusion/predictions/errors saved"
    )
    return metrics


def run_all_experiments(df, dirs, device):
    all_metrics = []
    experiments = []

    for setting in ["multiclass", "binary"]:
        train_df, test_df, labels = split_for_setting(df, setting)
        for vectorizer_name in ["bow", "tfidf"]:
            for model_name in ["naive_bayes", "ffnn"]:
                experiments.append((setting, train_df, test_df, labels, vectorizer_name, model_name))

    for setting, train_df, test_df, labels, vectorizer_name, model_name in progress_bar(
        experiments, "Task 3 experiments"
    ):
        metrics = run_one_experiment(
            setting,
            train_df,
            test_df,
            labels,
            vectorizer_name,
            model_name,
            dirs,
            device,
        )
        all_metrics.append(metrics)

    metrics_df = pd.DataFrame(all_metrics)
    columns = [
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
        "framework",
        "device",
        "training_mode",
        "top_confusions",
    ]
    metrics_df = metrics_df[columns]
    save_csv(metrics_df, dirs["root"] / "metrics_summary.csv")
    return metrics_df


def main():
    sns.set_theme(style="whitegrid")

    config = load_config()
    dirs = make_output_dirs(config["output_dir"])

    device = resolve_device()
    print(f"FFNN device: {device}")

    df = load_data(config["input_csv"])
    run_all_experiments(df, dirs, device)
    print(f"Done. Outputs saved to {Path(config['output_dir']).resolve()}")


if __name__ == "__main__":
    main()
