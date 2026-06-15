import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer


CONFIG_FILE = "task4_config.json"

TEXT_COL = "processed_text"
RAW_TEXT_COL = "text"
LABEL_COL = "sentiment"
NEGATIVE_LABEL = "negative"

RANDOM_SEED = 42
NUM_EXAMPLES = 15
MIN_TOKENS = 4
VECTOR_SIZE = 50
MAX_FEATURES = 5000
MIN_DF = 2
MAX_DF = 0.95
TOKEN_PATTERN = r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+(?:\.\d+)?|[!?]+"
TOP_PAIRS = 10
PLOT_FORMAT = "png"
DPI = 160


def load_config():
    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def save_csv(df, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def tokenize(text):
    return re.findall(TOKEN_PATTERN, str(text).lower())


def load_data(csv_path):
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    df = df[[RAW_TEXT_COL, TEXT_COL, LABEL_COL]].copy()
    df[RAW_TEXT_COL] = df[RAW_TEXT_COL].fillna("").astype(str)
    df[TEXT_COL] = df[TEXT_COL].fillna("").astype(str)
    df[LABEL_COL] = df[LABEL_COL].fillna("").astype(str).str.strip().str.lower()
    df = df[df[TEXT_COL].str.strip().astype(bool)]
    return df.reset_index(drop=True)


def select_negative_tweets(df):
    negative = df[df[LABEL_COL] == NEGATIVE_LABEL].copy()
    negative["tokens"] = negative[TEXT_COL].apply(tokenize)
    negative["token_count"] = negative["tokens"].apply(len)
    negative = negative[negative["token_count"] >= MIN_TOKENS].copy()

    negative = negative.sort_values("token_count").reset_index(drop=True)
    positions = np.linspace(0, len(negative) - 1, NUM_EXAMPLES).round().astype(int)
    selected = negative.iloc[positions].reset_index(drop=True)
    selected.insert(0, "tweet_id", [f"tweet_{i + 1:02d}" for i in range(len(selected))])
    return selected


def build_word_vectors(df):
    vectorizer = TfidfVectorizer(
        lowercase=True,
        token_pattern=TOKEN_PATTERN,
        ngram_range=(1, 1),
        min_df=MIN_DF,
        max_df=MAX_DF,
        max_features=MAX_FEATURES,
    )
    matrix = vectorizer.fit_transform(df[TEXT_COL].astype(str).tolist())
    words = vectorizer.get_feature_names_out()

    vector_size = min(VECTOR_SIZE, max(1, matrix.shape[1] - 1))
    svd = TruncatedSVD(n_components=vector_size, random_state=RANDOM_SEED)
    svd.fit(matrix)

    word_matrix = svd.components_.T
    return {word: word_matrix[i].astype(float) for i, word in enumerate(words)}


def average_vectors(tokens, word_vectors):
    vectors = [word_vectors[token] for token in tokens if token in word_vectors]
    vector_size = len(next(iter(word_vectors.values())))
    if not vectors:
        return np.zeros(vector_size, dtype=float)
    return np.mean(np.vstack(vectors), axis=0)


def cosine_similarity(a, b):
    dot = float(np.dot(a, b))
    norm_a = float(np.sqrt(np.dot(a, a)))
    norm_b = float(np.sqrt(np.dot(b, b)))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def make_tweet_vectors(selected, word_vectors):
    rows = []
    for _, row in selected.iterrows():
        tokens = row["tokens"]
        used_tokens = [token for token in tokens if token in word_vectors]
        vector = average_vectors(tokens, word_vectors)
        vector_values = {f"dim_{i + 1:02d}": value for i, value in enumerate(vector)}
        rows.append(
            {
                "tweet_id": row["tweet_id"],
                "tokens_used": " ".join(used_tokens),
                "num_tokens": len(tokens),
                "num_tokens_in_vocab": len(used_tokens),
                **vector_values,
            }
        )
    return pd.DataFrame(rows)


def make_similarity_tables(selected, tweet_vectors):
    vector_cols = [col for col in tweet_vectors.columns if col.startswith("dim_")]
    ids = tweet_vectors["tweet_id"].tolist()
    vectors = tweet_vectors[vector_cols].to_numpy(dtype=float)
    raw_text = dict(zip(selected["tweet_id"], selected[RAW_TEXT_COL]))

    matrix = np.zeros((len(ids), len(ids)), dtype=float)
    pairs = []

    for i, id_a in enumerate(ids):
        for j, id_b in enumerate(ids):
            score = cosine_similarity(vectors[i], vectors[j])
            matrix[i, j] = score
            if i < j:
                pairs.append(
                    {
                        "tweet_a": id_a,
                        "tweet_b": id_b,
                        "similarity": score,
                        "text_a": raw_text[id_a],
                        "text_b": raw_text[id_b],
                    }
                )

    matrix_df = pd.DataFrame(matrix, index=ids, columns=ids)
    matrix_df = matrix_df.reset_index().rename(columns={"index": "tweet_id"})
    pairs_df = pd.DataFrame(pairs).sort_values("similarity", ascending=False).reset_index(drop=True)
    return matrix_df, pairs_df


def save_heatmap(matrix_df, output_dir):
    labels = matrix_df["tweet_id"].tolist()
    values = matrix_df.drop(columns=["tweet_id"]).to_numpy(dtype=float)

    plt.figure(figsize=(9, 7.5))
    sns.heatmap(
        values,
        xticklabels=labels,
        yticklabels=labels,
        cmap="viridis",
        vmin=-1,
        vmax=1,
        square=True,
        cbar_kws={"label": "Cosine similarity"},
    )
    plt.title("Pairwise similarity between selected negative tweets")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_dir / f"similarity_heatmap.{PLOT_FORMAT}", dpi=DPI)
    plt.close()


def main():
    config = load_config()
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_data(config["input_csv"])
    selected = select_negative_tweets(df)
    word_vectors = build_word_vectors(df)
    tweet_vectors = make_tweet_vectors(selected, word_vectors)
    matrix_df, pairs_df = make_similarity_tables(selected, tweet_vectors)

    selected_out = selected[["tweet_id", RAW_TEXT_COL, TEXT_COL, "token_count"]].rename(
        columns={RAW_TEXT_COL: "raw_text", TEXT_COL: "processed_text"}
    )

    save_csv(selected_out, output_dir / "selected_negative_tweets.csv")
    save_csv(matrix_df, output_dir / "similarity_matrix.csv")
    save_csv(pairs_df.head(TOP_PAIRS), output_dir / "most_similar_pairs.csv")
    save_csv(pairs_df.tail(TOP_PAIRS).sort_values("similarity"), output_dir / "least_similar_pairs.csv")
    save_heatmap(matrix_df, output_dir)

    print(f"Done. Outputs saved to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
