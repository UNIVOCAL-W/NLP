import html
import json
import re
import string
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer


CONFIG_FILE = "tasks1_2_config.json"

TEXT_COL = "text"
LABEL_COL = "sentiment"
CLASS_ORDER = ["negative", "neutral", "positive"]
TOKEN_PATTERN = r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+(?:\.\d+)?|[!?]+"
STOPWORDS = {
    "i",
    "to",
    "the",
    "a",
    "my",
    "it",
    "you",
    "and",
    "is",
    "in",
    "s",
    "for",
    "of",
    "that",
    "me",
    "on",
    "so",
    "have",
    "but",
    "m",
    "just",
    "with",
    "be",
    "at",
    "was",
}


TOP_N_WORDS = 30
TOP_N_NGRAMS = 25
MIN_DF = 2
MAX_FEATURES = 50000
PLOT_FORMAT = "png"
DPI = 160

URL_RE = re.compile(r"https?://\S+|www\.\S+", flags=re.IGNORECASE)
MENTION_RE = re.compile(r"@\w+")
HASHTAG_RE = re.compile(r"#(\w+)")
SPACE_RE = re.compile(r"\s+")


def load_config():
    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def save_csv(df, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def tokenize(text):
    return re.findall(TOKEN_PATTERN, str(text))


def remove_custom_stopwords(tokens):
    return [token for token in tokens if token not in STOPWORDS]


def load_data(csv_path):
    df = pd.read_csv(csv_path, encoding="latin1")
    df = df[[TEXT_COL, LABEL_COL]].copy()
    df[TEXT_COL] = df[TEXT_COL].astype("string")
    df[LABEL_COL] = df[LABEL_COL].astype("string").str.strip().str.lower()
    df = df.dropna(subset=[TEXT_COL, LABEL_COL])
    df = df[df[TEXT_COL].str.strip().astype(bool)]
    return df.reset_index(drop=True)


def add_text_features(df):
    result = df.copy()
    result["char_len"] = result[TEXT_COL].astype(str).str.len()
    result["word_count"] = result[TEXT_COL].astype(str).apply(lambda x: len(tokenize(x)))
    result["unique_word_count"] = result[TEXT_COL].astype(str).apply(
        lambda x: len(set(token.lower() for token in tokenize(x)))
    )
    return result


def class_distribution(df):
    counts = df[LABEL_COL].value_counts().rename_axis(LABEL_COL).reset_index(name="count")
    counts["percentage"] = counts["count"] / counts["count"].sum() * 100
    counts[LABEL_COL] = pd.Categorical(counts[LABEL_COL], categories=CLASS_ORDER, ordered=True)
    return counts.sort_values(LABEL_COL).reset_index(drop=True)


def length_stats(df):
    stats = (
        df.groupby(LABEL_COL)
        .agg(
            rows=("char_len", "size"),
            char_mean=("char_len", "mean"),
            char_median=("char_len", "median"),
            char_min=("char_len", "min"),
            char_max=("char_len", "max"),
            word_mean=("word_count", "mean"),
            word_median=("word_count", "median"),
            word_min=("word_count", "min"),
            word_max=("word_count", "max"),
            unique_word_mean=("unique_word_count", "mean"),
        )
        .reset_index()
    )
    return stats.round(3)


def vocabulary_stats(df):
    def vocab_size(texts):
        words = set()
        for text in texts.astype(str):
            words.update(token.lower() for token in tokenize(text))
        return len(words)

    return {
        "overall_vocabulary_size": vocab_size(df[TEXT_COL]),
        "vocabulary_size_by_class": {
            label: vocab_size(group[TEXT_COL]) for label, group in df.groupby(LABEL_COL)
        },
    }


def top_terms(texts, ngram_range, top_n):
    vectorizer = CountVectorizer(
        lowercase=True,
        token_pattern=TOKEN_PATTERN,
        ngram_range=ngram_range,
        min_df=MIN_DF,
        max_features=MAX_FEATURES,
    )
    matrix = vectorizer.fit_transform(texts)
    counts = np.asarray(matrix.sum(axis=0)).ravel()
    terms = np.asarray(vectorizer.get_feature_names_out())
    order = counts.argsort()[::-1][:top_n]
    return pd.DataFrame({"term": terms[order], "count": counts[order].astype(int)})


def top_terms_by_class(df, ngram_range, top_n):
    tables = []
    for label, group in df.groupby(LABEL_COL):
        table = top_terms(group[TEXT_COL].astype(str).tolist(), ngram_range, top_n)
        table.insert(0, LABEL_COL, label)
        tables.append(table)
    return pd.concat(tables, ignore_index=True)


def save_class_plot(counts, output_dir):
    plt.figure(figsize=(7, 4.5))
    ax = sns.barplot(
        data=counts,
        x=LABEL_COL,
        y="count",
        hue=LABEL_COL,
        palette="Set2",
        legend=False,
    )
    ax.set_title("Sentiment class distribution")
    ax.set_xlabel("Sentiment")
    ax.set_ylabel("Number of tweets")
    for container in ax.containers:
        ax.bar_label(container, fmt="%d", padding=3)
    plt.tight_layout()
    plt.savefig(output_dir / f"class_distribution.{PLOT_FORMAT}", dpi=DPI)
    plt.close()


def save_length_plots(df, output_dir):
    plt.figure(figsize=(9, 5))
    sns.histplot(data=df, x="char_len", hue=LABEL_COL, bins=40, kde=True, element="step")
    plt.title("Tweet character length distribution")
    plt.xlabel("Characters")
    plt.ylabel("Tweets")
    plt.tight_layout()
    plt.savefig(output_dir / f"char_length_distribution.{PLOT_FORMAT}", dpi=DPI)
    plt.close()

    plt.figure(figsize=(8, 5))
    sns.boxplot(
        data=df,
        x=LABEL_COL,
        y="word_count",
        hue=LABEL_COL,
        order=CLASS_ORDER,
        hue_order=CLASS_ORDER,
        palette="Set2",
        legend=False,
    )
    plt.title("Tweet token count by sentiment")
    plt.xlabel("Sentiment")
    plt.ylabel("Token count")
    plt.tight_layout()
    plt.savefig(output_dir / f"token_count_by_sentiment.{PLOT_FORMAT}", dpi=DPI)
    plt.close()


def save_terms_plot(table, title, filename, output_dir):
    plot_df = table.sort_values("count", ascending=True)
    plt.figure(figsize=(9, max(5, len(plot_df) * 0.25)))
    sns.barplot(data=plot_df, x="count", y="term", color="#4C78A8")
    plt.title(title)
    plt.xlabel("Count")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(output_dir / f"{filename}.{PLOT_FORMAT}", dpi=DPI)
    plt.close()


def run_eda(df, output_dir):
    sns.set_theme(style="whitegrid")
    df = add_text_features(df)

    counts = class_distribution(df)
    stats = length_stats(df)
    vocab = vocabulary_stats(df)
    words_overall = top_terms(df[TEXT_COL].astype(str).tolist(), (1, 1), TOP_N_WORDS)
    ngrams_overall = top_terms(df[TEXT_COL].astype(str).tolist(), (2, 2), TOP_N_NGRAMS)
    words_by_class = top_terms_by_class(df, (1, 1), TOP_N_WORDS)
    ngrams_by_class = top_terms_by_class(df, (2, 2), TOP_N_NGRAMS)

    save_csv(counts, output_dir / "class_distribution.csv")
    save_csv(stats, output_dir / "length_stats_by_sentiment.csv")
    save_csv(words_overall, output_dir / "top_words_overall.csv")
    save_csv(words_by_class, output_dir / "top_words_by_sentiment.csv")
    save_csv(ngrams_overall, output_dir / "top_ngrams_overall.csv")
    save_csv(ngrams_by_class, output_dir / "top_ngrams_by_sentiment.csv")
    save_json(vocab, output_dir / "vocabulary_stats.json")

    save_class_plot(counts, output_dir)
    save_length_plots(df, output_dir)
    save_terms_plot(words_overall, "Most frequent words overall", "top_words_overall", output_dir)
    save_terms_plot(ngrams_overall, "Most frequent n-grams overall", "top_ngrams_overall", output_dir)
    return df


def preprocess_text(text):
    text = "" if pd.isna(text) else str(text)
    text = html.unescape(text).replace("`", "'")
    text = URL_RE.sub(" URL ", text)
    text = MENTION_RE.sub(" USER ", text)
    text = HASHTAG_RE.sub(r" \1 ", text)
    text = text.lower()

    punctuation = string.punctuation.replace("!", "").replace("?", "")
    text = text.translate(str.maketrans({char: " " for char in punctuation}))
    text = SPACE_RE.sub(" ", text).strip()
    tokens = remove_custom_stopwords(tokenize(text))
    return " ".join(tokens)


# def preprocessing_description():
#     return {
#         "html_unescape": True,
#         "normalize_backticks": True,
#         "lowercase": True,
#         "strip_whitespace": True,
#         "normalize_whitespace": True,
#         "normalize_elongated_words": False,
#         "max_repeated_chars": 2,
#         "url_mode": "replace",
#         "url_token": "URL",
#     }


def run_preprocessing(df, output_dir):
    processed = df.copy()
    processed["processed_text"] = processed[TEXT_COL].apply(preprocess_text)
    processed["processed_tokens"] = processed["processed_text"]
    processed["processed_char_len"] = processed["processed_text"].str.len()
    processed["processed_word_count"] = processed["processed_text"].apply(lambda x: len(x.split()) if x else 0)

    save_csv(processed, output_dir / "preprocessed_tweets.csv")
    save_csv(sample_examples(processed), output_dir / "preprocessing_examples.csv")

    length_summary = (
        processed.groupby(LABEL_COL)
        .agg(
            rows=("processed_word_count", "size"),
            processed_word_mean=("processed_word_count", "mean"),
            processed_word_median=("processed_word_count", "median"),
            processed_word_min=("processed_word_count", "min"),
            processed_word_max=("processed_word_count", "max"),
            processed_char_mean=("processed_char_len", "mean"),
            processed_char_median=("processed_char_len", "median"),
        )
        .reset_index()
        .round(3)
    )
    save_csv(length_summary, output_dir / "preprocessed_length_stats_by_sentiment.csv")

    change_summary = {
        "raw_mean_word_count": float(df["word_count"].mean()),
        "processed_mean_word_count": float(processed["processed_word_count"].mean()),
        "raw_mean_char_len": float(df["char_len"].mean()),
        "processed_mean_char_len": float(processed["processed_char_len"].mean()),
    }
    save_json(change_summary, output_dir / "preprocessing_change_summary.json")
    #save_json(preprocessing_description(), output_dir / "preprocessing_config_used.json")
    return processed


def sample_examples(df):
    examples = []
    for label, group in df.groupby(LABEL_COL):
        examples.append(group.sample(n=min(5, len(group)), random_state=42)[[LABEL_COL, TEXT_COL, "processed_text"]])
    return pd.concat(examples, ignore_index=True)


def main():
    config = load_config()
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    data = load_data(config["input_csv"])
    data_with_features = run_eda(data, output_dir)
    run_preprocessing(data_with_features, output_dir)
    print(f"Done. Outputs saved to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
