"""
Task 4: Textual similarity between negative tweets.

This script selects 15 negative tweets, builds word vectors from the local tweet
corpus, represents each selected tweet by averaging its word vectors, and then
computes pairwise cosine similarities manually.

Usage:
    python task4.py
    python task4.py --config task4_config.json
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer


DEFAULT_CONFIG_FILE = "task4_config.json"
SCRIPT_DIR = Path(__file__).resolve().parent
LOGGER = logging.getLogger("task4")

DEFAULT_CONFIG: Dict[str, Any] = {
    "data": {
        "csv_path": "outputs/tasks1_2/preprocessed_tweets.csv",
        "text_column": "processed_text",
        "raw_text_column": "text",
        "label_column": "sentiment",
        "negative_label": "negative",
        "encoding": "utf-8-sig",
        "random_seed": 42,
    },
    "selection": {
        "num_examples": 15,
        "min_tokens": 4,
        "strategy": "diverse_length",
    },
    "vectors": {
        "max_features": 5000,
        "min_df": 2,
        "max_df": 0.95,
        "ngram_range": [1, 1],
        "vector_size": 50,
        "token_pattern": r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+(?:\.\d+)?|[!?]+",
    },
    "output": {
        "output_dir": "outputs/task4",
        "plot_format": "png",
        "figure_dpi": 160,
        "top_pairs": 10,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Task 4 textual similarity.")
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG_FILE,
        help=f"JSON config that overrides defaults. Defaults to {DEFAULT_CONFIG_FILE}.",
    )
    parser.add_argument("--output-dir", default=None, help="Optional output directory override.")
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
    return config


def save_json(data: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)


def save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def tokenize(text: str, token_pattern: str) -> List[str]:
    return re.findall(token_pattern, str(text).lower())


def load_dataset(config: Dict[str, Any]) -> pd.DataFrame:
    data_config = config["data"]
    csv_path = resolve_project_path(data_config["csv_path"])
    LOGGER.info("Loading preprocessed tweets from %s", csv_path)

    df = pd.read_csv(csv_path, encoding=data_config["encoding"])
    required = [
        data_config["text_column"],
        data_config["raw_text_column"],
        data_config["label_column"],
    ]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {csv_path}: {missing}")

    df = df.dropna(subset=[data_config["text_column"], data_config["label_column"]]).copy()
    df[data_config["text_column"]] = df[data_config["text_column"]].astype(str)
    df[data_config["raw_text_column"]] = df[data_config["raw_text_column"]].astype(str)
    df[data_config["label_column"]] = df[data_config["label_column"]].astype(str).str.lower()
    df = df[df[data_config["text_column"]].str.strip().astype(bool)].reset_index(drop=True)
    return df


def select_negative_examples(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    data_config = config["data"]
    selection_config = config["selection"]
    token_pattern = config["vectors"]["token_pattern"]
    text_col = data_config["text_column"]
    label_col = data_config["label_column"]

    negative = df[df[label_col] == data_config["negative_label"]].copy()
    negative["tokens"] = negative[text_col].apply(lambda text: tokenize(text, token_pattern))
    negative["token_count"] = negative["tokens"].apply(len)
    negative = negative[negative["token_count"] >= selection_config["min_tokens"]].copy()

    n = int(selection_config["num_examples"])
    if len(negative) < n:
        raise ValueError(f"Need {n} negative examples, but only found {len(negative)}.")

    if selection_config.get("strategy") == "random":
        selected = negative.sample(n=n, random_state=data_config["random_seed"])
    else:
        negative = negative.sort_values("token_count").reset_index()
        positions = np.linspace(0, len(negative) - 1, n).round().astype(int)
        selected = negative.iloc[positions]

    selected = selected.reset_index(drop=True)
    selected.insert(0, "tweet_id", [f"tweet_{index + 1:02d}" for index in range(len(selected))])
    return selected


def build_word_vectors(df: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, np.ndarray]:
    text_col = config["data"]["text_column"]
    vector_config = config["vectors"]

    vectorizer = TfidfVectorizer(
        token_pattern=vector_config["token_pattern"],
        lowercase=True,
        max_features=vector_config["max_features"],
        min_df=vector_config["min_df"],
        max_df=vector_config["max_df"],
        ngram_range=tuple(vector_config["ngram_range"]),
    )
    matrix = vectorizer.fit_transform(df[text_col].astype(str))
    feature_names = vectorizer.get_feature_names_out()

    vector_size = min(int(vector_config["vector_size"]), max(1, matrix.shape[1] - 1))
    LOGGER.info("Learning %d-dimensional word vectors for %d tokens", vector_size, len(feature_names))
    svd = TruncatedSVD(n_components=vector_size, random_state=config["data"]["random_seed"])
    svd.fit(matrix)

    word_matrix = svd.components_.T
    return {
        word: word_matrix[index].astype(float)
        for index, word in enumerate(feature_names)
    }


def average_word_vectors(tokens: Sequence[str], word_vectors: Dict[str, np.ndarray], vector_size: int) -> np.ndarray:
    vectors = [word_vectors[token] for token in tokens if token in word_vectors]
    if not vectors:
        return np.zeros(vector_size, dtype=float)
    return np.mean(np.vstack(vectors), axis=0)


def manual_cosine_similarity(vector_a: np.ndarray, vector_b: np.ndarray) -> float:
    dot_product = float(np.dot(vector_a, vector_b))
    norm_a = float(np.sqrt(np.dot(vector_a, vector_a)))
    norm_b = float(np.sqrt(np.dot(vector_b, vector_b)))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot_product / (norm_a * norm_b)


def compute_tweet_vectors(
    selected: pd.DataFrame,
    word_vectors: Dict[str, np.ndarray],
    config: Dict[str, Any],
) -> pd.DataFrame:
    vector_size = len(next(iter(word_vectors.values())))
    rows = []
    for _, row in selected.iterrows():
        tokens = row["tokens"]
        in_vocab_tokens = [token for token in tokens if token in word_vectors]
        vector = average_word_vectors(tokens, word_vectors, vector_size)
        rows.append(
            {
                "tweet_id": row["tweet_id"],
                "tokens_used": " ".join(in_vocab_tokens),
                "num_tokens": len(tokens),
                "num_tokens_in_vocab": len(in_vocab_tokens),
                **{f"dim_{index + 1:02d}": value for index, value in enumerate(vector)},
            }
        )
    return pd.DataFrame(rows)


def compute_similarity_outputs(
    selected: pd.DataFrame,
    tweet_vectors: pd.DataFrame,
    config: Dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    vector_columns = [column for column in tweet_vectors.columns if column.startswith("dim_")]
    ids = tweet_vectors["tweet_id"].tolist()
    vectors = tweet_vectors[vector_columns].to_numpy(dtype=float)

    matrix = np.zeros((len(ids), len(ids)), dtype=float)
    pair_rows = []
    raw_lookup = dict(zip(selected["tweet_id"], selected[config["data"]["raw_text_column"]]))

    for i, id_a in enumerate(ids):
        for j, id_b in enumerate(ids):
            matrix[i, j] = manual_cosine_similarity(vectors[i], vectors[j])
            if i < j:
                pair_rows.append(
                    {
                        "tweet_a": id_a,
                        "tweet_b": id_b,
                        "similarity": matrix[i, j],
                        "text_a": raw_lookup[id_a],
                        "text_b": raw_lookup[id_b],
                    }
                )

    matrix_df = pd.DataFrame(matrix, index=ids, columns=ids).reset_index().rename(columns={"index": "tweet_id"})
    pairs_df = pd.DataFrame(pair_rows).sort_values("similarity", ascending=False).reset_index(drop=True)
    return matrix_df, pairs_df


def save_similarity_heatmap(matrix_df: pd.DataFrame, output_dir: Path, config: Dict[str, Any]) -> None:
    values = matrix_df.drop(columns=["tweet_id"]).to_numpy(dtype=float)
    labels = matrix_df["tweet_id"].tolist()

    plt.figure(figsize=(9, 7.5))
    sns.heatmap(
        values,
        xticklabels=labels,
        yticklabels=labels,
        cmap="viridis",
        vmin=-1,
        vmax=1,
        square=True,
        cbar_kws={"label": "Manual cosine similarity"},
    )
    plt.title("Pairwise similarity between selected negative tweets")
    plt.xlabel("")
    plt.ylabel("")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(
        output_dir / f"similarity_heatmap.{config['output']['plot_format']}",
        dpi=config["output"]["figure_dpi"],
    )
    plt.close()


def generate_report_notes(pairs_df: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
    top_n = int(config["output"]["top_pairs"])
    most_similar = pairs_df.head(3)
    least_similar = pairs_df.tail(3).sort_values("similarity")

    notes = [
        "Tweet vectors were created by averaging word vectors learned from the local tweet corpus.",
        "Cosine similarity was computed manually as dot(a, b) / (||a|| * ||b||), without a pre-built similarity function.",
        "Use the most and least similar pair tables to discuss whether lexical overlap, shared topics, or general negative wording explain the scores.",
    ]
    return {
        "notes": notes,
        "most_similar_preview": most_similar[["tweet_a", "tweet_b", "similarity"]].to_dict(orient="records"),
        "least_similar_preview": least_similar[["tweet_a", "tweet_b", "similarity"]].to_dict(orient="records"),
        "top_pairs_saved": top_n,
    }


def main() -> None:
    setup_logging()
    args = parse_args()
    config = load_config(args)
    output_dir = resolve_project_path(config["output"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(config, output_dir / "config_used.json")

    df = load_dataset(config)
    selected = select_negative_examples(df, config)
    word_vectors = build_word_vectors(df, config)
    tweet_vectors = compute_tweet_vectors(selected, word_vectors, config)
    matrix_df, pairs_df = compute_similarity_outputs(selected, tweet_vectors, config)

    top_n = int(config["output"]["top_pairs"])
    selected_output = selected[
        ["tweet_id", config["data"]["raw_text_column"], config["data"]["text_column"], "token_count"]
    ].rename(
        columns={
            config["data"]["raw_text_column"]: "raw_text",
            config["data"]["text_column"]: "processed_text",
        }
    )

    save_csv(selected_output, output_dir / "selected_negative_tweets.csv")
    save_csv(tweet_vectors, output_dir / "tweet_vectors.csv")
    save_csv(matrix_df, output_dir / "similarity_matrix.csv")
    save_csv(pairs_df, output_dir / "all_similarity_pairs.csv")
    save_csv(pairs_df.head(top_n), output_dir / "most_similar_pairs.csv")
    save_csv(pairs_df.tail(top_n).sort_values("similarity"), output_dir / "least_similar_pairs.csv")
    save_similarity_heatmap(matrix_df, output_dir, config)
    save_json(generate_report_notes(pairs_df, config), output_dir / "report_notes_task4.json")

    LOGGER.info("Task 4 completed. Outputs saved to %s", output_dir.resolve())


if __name__ == "__main__":
    main()
