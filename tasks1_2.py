"""
Tasks 1 and 2 for the Twitter sentiment project.

This script performs:
    Task 1: Exploratory Data Analysis (EDA)
    Task 2: Configurable text preprocessing

Usage:
    python tasks1_2.py
    python tasks1_2.py --config config.json
    python tasks1_2.py --sample-size 5000 --output-dir outputs/quick_run

All adjustable parameters live in DEFAULT_CONFIG below. You can either edit that
dictionary directly or pass a JSON file that overrides only the values you want
to change.
"""

from __future__ import annotations

import argparse
import html
import json
import logging
import re
import string
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
from sklearn.model_selection import train_test_split

try:
    from wordcloud import WordCloud
except ImportError:  # Word clouds are useful but not essential.
    WordCloud = None


DEFAULT_CONFIG: Dict[str, Any] = {
    "data": {
        "csv_path": "TweetSentiment/TweetSentiment.csv",
        "text_column": "text",
        "label_column": "sentiment",
        "keep_columns": ["text", "sentiment"],
        "sample_size": None,
        "random_seed": 42,
        "encoding": "latin1",
    },
    "eda": {
        "output_dir": "outputs/tasks1_2",
        "plot_format": "png",
        "figure_dpi": 160,
        "class_order": ["negative", "neutral", "positive"],
        "top_n_words": 30,
        "top_n_ngrams": 25,
        "ngram_range": [2, 2],
        "min_df": 2,
        "max_features": 50000,
        "hist_bins": 40,
        "make_wordcloud": True,
        "wordcloud_max_words": 120,
        "wordcloud_width": 1200,
        "wordcloud_height": 700,
        "wordcloud_background": "white",
        "examples_per_class": 5,
    },
    "preprocessing": {
        "html_unescape": True,
        "normalize_backticks": True,
        "lowercase": True,
        "strip_whitespace": True,
        "normalize_whitespace": True,
        "normalize_elongated_words": False,
        "max_repeated_chars": 2,
        "url_mode": "replace",          # keep | remove | replace
        "url_token": "URL",
        "mention_mode": "replace",      # keep | remove | replace
        "mention_token": "USER",
        "hashtag_mode": "remove_hash",  # keep | remove | remove_hash
        "number_mode": "keep",          # keep | remove | replace
        "number_token": "NUMBER",
        "remove_punctuation": True,
        "keep_sentence_emotion": True,   # keeps ! and ? if punctuation is removed
        "remove_non_ascii": False,
        "token_pattern": r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+(?:\.\d+)?|[!?]+",
        "min_token_length": 1,
        "remove_stopwords": False,
        "stopword_source": "sklearn",    # sklearn | nltk
        "keep_negations": True,
        "extra_stopwords": [],
        "stemming": False,
        "lemmatization": False,
        "processed_text_column": "processed_text",
        "tokens_column": "processed_tokens",
    },
}


DEFAULT_CONFIG_FILE = "tasks1_2_config.json"
LOGGER = logging.getLogger("tasks1_2")
URL_RE = re.compile(r"https?://\S+|www\.\S+", flags=re.IGNORECASE)
MENTION_RE = re.compile(r"@\w+")
HASHTAG_RE = re.compile(r"#(\w+)")
NUMBER_RE = re.compile(r"\b\d+(?:[.,]\d+)?\b")
WHITESPACE_RE = re.compile(r"\s+")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Tasks 1 and 2.")
    parser.add_argument(
        "--config",
        type=str,
        default=DEFAULT_CONFIG_FILE,
        help=(
            "JSON config that overrides DEFAULT_CONFIG. "
            f"Defaults to {DEFAULT_CONFIG_FILE} if the file exists."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Optional output directory override.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Optional sample size override for quick runs.",
    )
    parser.add_argument(
        "--skip-wordcloud",
        action="store_true",
        help="Disable word cloud generation.",
    )
    return parser.parse_args()


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def deep_update(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively update a nested config dictionary."""
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_config(args: argparse.Namespace) -> Dict[str, Any]:
    config = deepcopy(DEFAULT_CONFIG)

    if args.config and Path(args.config).exists():
        config_path = Path(args.config)
        with config_path.open("r", encoding="utf-8") as file:
            user_config = json.load(file)
        deep_update(config, user_config)
    elif args.config != DEFAULT_CONFIG_FILE:
        raise FileNotFoundError(f"Config file not found: {args.config}")

    if args.output_dir:
        config["eda"]["output_dir"] = args.output_dir

    if args.sample_size is not None:
        config["data"]["sample_size"] = args.sample_size

    if args.skip_wordcloud:
        config["eda"]["make_wordcloud"] = False

    return config


def ensure_output_dir(config: Dict[str, Any]) -> Path:
    output_dir = Path(config["eda"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def save_json(data: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)


def save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def load_dataset(config: Dict[str, Any]) -> pd.DataFrame:
    data_config = config["data"]
    csv_path = Path(data_config["csv_path"])
    text_col = data_config["text_column"]
    label_col = data_config["label_column"]

    LOGGER.info("Loading dataset from %s", csv_path)
    df = pd.read_csv(csv_path, encoding=data_config["encoding"])

    missing_required = [col for col in [text_col, label_col] if col not in df.columns]
    if missing_required:
        raise ValueError(f"Missing required columns: {missing_required}")

    keep_columns = data_config.get("keep_columns") or [text_col, label_col]
    keep_columns = [col for col in keep_columns if col in df.columns]
    df = df.loc[:, keep_columns].copy()

    raw_rows = len(df)
    df[text_col] = df[text_col].astype("string")
    df[label_col] = df[label_col].astype("string").str.strip().str.lower()

    before_drop = len(df)
    df = df.dropna(subset=[text_col, label_col]).copy()
    df = df[df[text_col].str.strip().astype(bool)].copy()
    dropped_rows = before_drop - len(df)

    sample_size = data_config.get("sample_size")
    if sample_size is not None and sample_size < len(df):
        df, _ = train_test_split(
            df,
            train_size=sample_size,
            random_state=data_config["random_seed"],
            stratify=df[label_col] if df[label_col].nunique() > 1 else None,
        )

    df = df.reset_index(drop=True)
    LOGGER.info(
        "Loaded %d usable rows from %d raw rows (%d removed).",
        len(df),
        raw_rows,
        dropped_rows,
    )
    return df


def simple_tokenize(text: str, token_pattern: str) -> List[str]:
    return re.findall(token_pattern, str(text))


def add_text_features(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    text_col = config["data"]["text_column"]
    token_pattern = config["preprocessing"]["token_pattern"]
    result = df.copy()
    result["char_len"] = result[text_col].astype(str).str.len()
    result["word_count"] = result[text_col].astype(str).apply(
        lambda text: len(simple_tokenize(text, token_pattern))
    )
    result["unique_word_count"] = result[text_col].astype(str).apply(
        lambda text: len(set(token.lower() for token in simple_tokenize(text, token_pattern)))
    )
    return result


def class_order_for_data(df: pd.DataFrame, config: Dict[str, Any]) -> List[str]:
    label_col = config["data"]["label_column"]
    preferred = config["eda"].get("class_order") or []
    labels = list(df[label_col].dropna().unique())
    ordered = [label for label in preferred if label in labels]
    ordered.extend(sorted(label for label in labels if label not in ordered))
    return ordered


def compute_class_distribution(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    label_col = config["data"]["label_column"]
    counts = df[label_col].value_counts().rename_axis(label_col).reset_index(name="count")
    counts["percentage"] = counts["count"] / counts["count"].sum() * 100

    order = class_order_for_data(df, config)
    counts[label_col] = pd.Categorical(counts[label_col], categories=order, ordered=True)
    return counts.sort_values(label_col).reset_index(drop=True)


def compute_length_stats(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    label_col = config["data"]["label_column"]
    stats = (
        df.groupby(label_col)
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


def get_stopwords(config: Dict[str, Any]) -> Optional[Sequence[str]]:
    prep_config = config["preprocessing"]
    if not prep_config.get("remove_stopwords"):
        return None

    source = prep_config.get("stopword_source", "sklearn").lower()
    if source == "nltk":
        try:
            from nltk.corpus import stopwords

            stop_words = set(stopwords.words("english"))
        except Exception as exc:
            LOGGER.warning(
                "Could not load NLTK stopwords (%s). Falling back to sklearn.",
                exc,
            )
            stop_words = set(ENGLISH_STOP_WORDS)
    else:
        stop_words = set(ENGLISH_STOP_WORDS)

    if prep_config.get("keep_negations"):
        stop_words -= {
            "no",
            "nor",
            "not",
            "never",
            "n't",
            "cannot",
            "cant",
            "won",
            "wont",
            "dont",
            "didnt",
            "isnt",
            "wasnt",
        }

    stop_words.update(str(word).lower() for word in prep_config.get("extra_stopwords", []))
    return sorted(stop_words)


def build_vectorizer(
    config: Dict[str, Any],
    ngram_range: Tuple[int, int],
    stop_words: Optional[Sequence[str]] = None,
) -> CountVectorizer:
    eda_config = config["eda"]
    return CountVectorizer(
        lowercase=True,
        token_pattern=config["preprocessing"]["token_pattern"],
        ngram_range=ngram_range,
        min_df=eda_config["min_df"],
        max_features=eda_config["max_features"],
        stop_words=stop_words,
    )


def top_terms_from_texts(
    texts: Sequence[str],
    config: Dict[str, Any],
    ngram_range: Tuple[int, int],
    top_n: int,
    stop_words: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    if len(texts) == 0:
        return pd.DataFrame(columns=["term", "count"])

    vectorizer = build_vectorizer(config, ngram_range=ngram_range, stop_words=stop_words)
    try:
        matrix = vectorizer.fit_transform(texts)
    except ValueError:
        return pd.DataFrame(columns=["term", "count"])

    counts = np.asarray(matrix.sum(axis=0)).ravel()
    terms = np.asarray(vectorizer.get_feature_names_out())
    order = counts.argsort()[::-1][:top_n]
    return pd.DataFrame({"term": terms[order], "count": counts[order].astype(int)})


def compute_vocabulary_stats(
    df: pd.DataFrame,
    config: Dict[str, Any],
    stop_words: Optional[Sequence[str]],
) -> Dict[str, Any]:
    text_col = config["data"]["text_column"]
    label_col = config["data"]["label_column"]
    token_pattern = config["preprocessing"]["token_pattern"]
    stop_set = set(stop_words or [])

    def vocab_size(texts: Iterable[str]) -> int:
        vocab = set()
        for text in texts:
            tokens = [token.lower() for token in simple_tokenize(str(text), token_pattern)]
            vocab.update(token for token in tokens if token not in stop_set)
        return len(vocab)

    by_class = {}
    for label, group in df.groupby(label_col):
        by_class[str(label)] = vocab_size(group[text_col].astype(str))

    return {
        "overall_vocabulary_size": vocab_size(df[text_col].astype(str)),
        "vocabulary_size_by_class": by_class,
    }


def compute_top_term_tables(
    df: pd.DataFrame,
    config: Dict[str, Any],
    stop_words: Optional[Sequence[str]],
) -> Dict[str, pd.DataFrame]:
    text_col = config["data"]["text_column"]
    label_col = config["data"]["label_column"]
    eda_config = config["eda"]

    top_words_overall = top_terms_from_texts(
        df[text_col].astype(str).tolist(),
        config,
        ngram_range=(1, 1),
        top_n=eda_config["top_n_words"],
        stop_words=stop_words,
    )

    ngram_range = tuple(eda_config["ngram_range"])
    top_ngrams_overall = top_terms_from_texts(
        df[text_col].astype(str).tolist(),
        config,
        ngram_range=ngram_range,
        top_n=eda_config["top_n_ngrams"],
        stop_words=stop_words,
    )

    top_words_by_class = []
    top_ngrams_by_class = []
    for label, group in df.groupby(label_col):
        words = top_terms_from_texts(
            group[text_col].astype(str).tolist(),
            config,
            ngram_range=(1, 1),
            top_n=eda_config["top_n_words"],
            stop_words=stop_words,
        )
        words.insert(0, label_col, label)
        top_words_by_class.append(words)

        ngrams = top_terms_from_texts(
            group[text_col].astype(str).tolist(),
            config,
            ngram_range=ngram_range,
            top_n=eda_config["top_n_ngrams"],
            stop_words=stop_words,
        )
        ngrams.insert(0, label_col, label)
        top_ngrams_by_class.append(ngrams)

    return {
        "top_words_overall": top_words_overall,
        "top_words_by_class": pd.concat(top_words_by_class, ignore_index=True),
        "top_ngrams_overall": top_ngrams_overall,
        "top_ngrams_by_class": pd.concat(top_ngrams_by_class, ignore_index=True),
    }


def save_class_distribution_plot(
    class_distribution: pd.DataFrame,
    config: Dict[str, Any],
    output_dir: Path,
) -> None:
    label_col = config["data"]["label_column"]
    plot_format = config["eda"]["plot_format"]
    dpi = config["eda"]["figure_dpi"]

    plt.figure(figsize=(7, 4.5))
    ax = sns.barplot(data=class_distribution, x=label_col, y="count", palette="Set2")
    ax.set_title("Sentiment class distribution")
    ax.set_xlabel("Sentiment")
    ax.set_ylabel("Number of tweets")
    for container in ax.containers:
        ax.bar_label(container, fmt="%d", padding=3)
    plt.tight_layout()
    plt.savefig(output_dir / f"class_distribution.{plot_format}", dpi=dpi)
    plt.close()


def save_length_plots(df: pd.DataFrame, config: Dict[str, Any], output_dir: Path) -> None:
    label_col = config["data"]["label_column"]
    plot_format = config["eda"]["plot_format"]
    dpi = config["eda"]["figure_dpi"]
    bins = config["eda"]["hist_bins"]

    plt.figure(figsize=(9, 5))
    sns.histplot(data=df, x="char_len", hue=label_col, bins=bins, kde=True, element="step")
    plt.title("Tweet character length distribution")
    plt.xlabel("Characters")
    plt.ylabel("Tweets")
    plt.tight_layout()
    plt.savefig(output_dir / f"char_length_distribution.{plot_format}", dpi=dpi)
    plt.close()

    plt.figure(figsize=(8, 5))
    order = class_order_for_data(df, config)
    sns.boxplot(data=df, x=label_col, y="word_count", order=order, palette="Set2")
    plt.title("Tweet token count by sentiment")
    plt.xlabel("Sentiment")
    plt.ylabel("Token count")
    plt.tight_layout()
    plt.savefig(output_dir / f"token_count_by_sentiment.{plot_format}", dpi=dpi)
    plt.close()


def save_top_terms_plot(
    terms: pd.DataFrame,
    title: str,
    filename: str,
    config: Dict[str, Any],
    output_dir: Path,
) -> None:
    if terms.empty:
        return

    plot_format = config["eda"]["plot_format"]
    dpi = config["eda"]["figure_dpi"]
    plot_df = terms.sort_values("count", ascending=True)

    plt.figure(figsize=(9, max(5, len(plot_df) * 0.25)))
    sns.barplot(data=plot_df, x="count", y="term", color="#4C78A8")
    plt.title(title)
    plt.xlabel("Count")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(output_dir / f"{filename}.{plot_format}", dpi=dpi)
    plt.close()


def save_wordclouds(
    df: pd.DataFrame,
    config: Dict[str, Any],
    output_dir: Path,
    stop_words: Optional[Sequence[str]],
) -> None:
    eda_config = config["eda"]
    if not eda_config.get("make_wordcloud"):
        return

    if WordCloud is None:
        LOGGER.warning("wordcloud is not installed; skipping word cloud generation.")
        return

    text_col = config["data"]["text_column"]
    label_col = config["data"]["label_column"]
    plot_format = eda_config["plot_format"]

    def build_cloud(texts: Sequence[str], filename: str) -> None:
        text = " ".join(str(value) for value in texts if str(value).strip())
        if not text.strip():
            return
        cloud = WordCloud(
            width=eda_config["wordcloud_width"],
            height=eda_config["wordcloud_height"],
            max_words=eda_config["wordcloud_max_words"],
            background_color=eda_config["wordcloud_background"],
            stopwords=set(stop_words or []),
            collocations=False,
            random_state=config["data"]["random_seed"],
        ).generate(text)
        cloud.to_file(str(output_dir / f"{filename}.{plot_format}"))

    build_cloud(df[text_col].astype(str).tolist(), "wordcloud_overall")
    for label, group in df.groupby(label_col):
        safe_label = re.sub(r"[^A-Za-z0-9_-]+", "_", str(label)).strip("_")
        build_cloud(group[text_col].astype(str).tolist(), f"wordcloud_{safe_label}")


def generate_report_notes(
    class_distribution: pd.DataFrame,
    length_stats: pd.DataFrame,
    vocabulary_stats: Dict[str, Any],
    top_words: pd.DataFrame,
    config: Dict[str, Any],
) -> List[str]:
    label_col = config["data"]["label_column"]
    notes = []

    if not class_distribution.empty:
        max_row = class_distribution.loc[class_distribution["count"].idxmax()]
        min_row = class_distribution.loc[class_distribution["count"].idxmin()]
        notes.append(
            "Class balance: the largest class is "
            f"{max_row[label_col]} ({int(max_row['count'])} tweets), while the smallest is "
            f"{min_row[label_col]} ({int(min_row['count'])} tweets)."
        )

    if not length_stats.empty:
        longest = length_stats.sort_values("word_mean", ascending=False).iloc[0]
        notes.append(
            "Text length: the class with the highest average token count is "
            f"{longest[label_col]} ({longest['word_mean']:.2f} tokens on average)."
        )

    vocab_size = vocabulary_stats.get("overall_vocabulary_size")
    if vocab_size is not None:
        notes.append(
            f"Vocabulary: the corpus contains about {vocab_size} unique tokens under the current tokenizer."
        )

    if not top_words.empty:
        preview = ", ".join(top_words["term"].head(8).astype(str).tolist())
        notes.append(f"Frequent words to inspect in the report: {preview}.")

    notes.append(
        "Preprocessing reflection: compare whether URLs, mentions, hashtags, casing, punctuation, "
        "and stop-word removal preserve or remove sentiment cues such as negation and emphasis."
    )
    return notes


def run_eda(df: pd.DataFrame, config: Dict[str, Any], output_dir: Path) -> pd.DataFrame:
    LOGGER.info("Running Task 1: exploratory data analysis")
    sns.set_theme(style="whitegrid")

    df_features = add_text_features(df, config)
    stop_words = get_stopwords(config)

    class_distribution = compute_class_distribution(df_features, config)
    length_stats = compute_length_stats(df_features, config)
    vocabulary_stats = compute_vocabulary_stats(df_features, config, stop_words)
    top_tables = compute_top_term_tables(df_features, config, stop_words)

    save_csv(class_distribution, output_dir / "class_distribution.csv")
    save_csv(length_stats, output_dir / "length_stats_by_sentiment.csv")
    save_csv(top_tables["top_words_overall"], output_dir / "top_words_overall.csv")
    save_csv(top_tables["top_words_by_class"], output_dir / "top_words_by_sentiment.csv")
    save_csv(top_tables["top_ngrams_overall"], output_dir / "top_ngrams_overall.csv")
    save_csv(top_tables["top_ngrams_by_class"], output_dir / "top_ngrams_by_sentiment.csv")
    save_json(vocabulary_stats, output_dir / "vocabulary_stats.json")

    overview = {
        "rows": int(len(df_features)),
        "columns": list(df_features.columns),
        "sentiment_labels": class_order_for_data(df_features, config),
        "class_distribution": class_distribution.to_dict(orient="records"),
        "length_stats_by_sentiment": length_stats.to_dict(orient="records"),
    }
    save_json(overview, output_dir / "dataset_overview.json")

    save_class_distribution_plot(class_distribution, config, output_dir)
    save_length_plots(df_features, config, output_dir)
    save_top_terms_plot(
        top_tables["top_words_overall"],
        "Most frequent words overall",
        "top_words_overall",
        config,
        output_dir,
    )
    save_top_terms_plot(
        top_tables["top_ngrams_overall"],
        "Most frequent n-grams overall",
        "top_ngrams_overall",
        config,
        output_dir,
    )
    save_wordclouds(df_features, config, output_dir, stop_words)

    notes = generate_report_notes(
        class_distribution,
        length_stats,
        vocabulary_stats,
        top_tables["top_words_overall"],
        config,
    )
    save_json({"notes": notes}, output_dir / "report_notes_task1.json")

    LOGGER.info("Task 1 outputs saved to %s", output_dir)
    return df_features


def normalize_elongated_words(text: str, max_repeated_chars: int) -> str:
    if max_repeated_chars < 1:
        return text
    pattern = re.compile(r"(.)\1{" + str(max_repeated_chars) + r",}")
    return pattern.sub(lambda match: match.group(1) * max_repeated_chars, text)


def get_text_normalizer(config: Dict[str, Any]):
    prep = config["preprocessing"]

    def normalize(text: Any) -> str:
        value = "" if pd.isna(text) else str(text)

        if prep["html_unescape"]:
            value = html.unescape(value)

        if prep["normalize_backticks"]:
            value = value.replace("`", "'")

        if prep["url_mode"] == "remove":
            value = URL_RE.sub(" ", value)
        elif prep["url_mode"] == "replace":
            value = URL_RE.sub(f" {prep['url_token']} ", value)

        if prep["mention_mode"] == "remove":
            value = MENTION_RE.sub(" ", value)
        elif prep["mention_mode"] == "replace":
            value = MENTION_RE.sub(f" {prep['mention_token']} ", value)

        if prep["hashtag_mode"] == "remove":
            value = HASHTAG_RE.sub(" ", value)
        elif prep["hashtag_mode"] == "remove_hash":
            value = HASHTAG_RE.sub(r" \1 ", value)

        if prep["number_mode"] == "remove":
            value = NUMBER_RE.sub(" ", value)
        elif prep["number_mode"] == "replace":
            value = NUMBER_RE.sub(f" {prep['number_token']} ", value)

        if prep["remove_non_ascii"]:
            value = value.encode("ascii", errors="ignore").decode("ascii")

        if prep["normalize_elongated_words"]:
            value = normalize_elongated_words(value, prep["max_repeated_chars"])

        if prep["lowercase"]:
            value = value.lower()

        if prep["remove_punctuation"]:
            punctuation = string.punctuation
            if prep["keep_sentence_emotion"]:
                punctuation = punctuation.replace("!", "").replace("?", "")
            translator = str.maketrans({char: " " for char in punctuation})
            value = value.translate(translator)

        if prep["normalize_whitespace"]:
            value = WHITESPACE_RE.sub(" ", value)

        if prep["strip_whitespace"]:
            value = value.strip()

        return value

    return normalize


def build_token_transformers(config: Dict[str, Any]):
    prep = config["preprocessing"]
    stop_words = set(get_stopwords(config) or [])
    stemmer = None
    lemmatizer = None

    if prep["stemming"]:
        try:
            from nltk.stem import PorterStemmer

            stemmer = PorterStemmer()
        except Exception as exc:
            LOGGER.warning("Could not initialize NLTK PorterStemmer: %s", exc)

    if prep["lemmatization"]:
        try:
            from nltk.stem import WordNetLemmatizer

            lemmatizer = WordNetLemmatizer()
        except Exception as exc:
            LOGGER.warning("Could not initialize NLTK WordNetLemmatizer: %s", exc)

    warned_lemma = {"value": False}

    def transform(tokens: Sequence[str]) -> List[str]:
        result = []
        for token in tokens:
            if len(token) < prep["min_token_length"]:
                continue
            if token.lower() in stop_words:
                continue

            value = token
            if stemmer is not None:
                value = stemmer.stem(value)
            if lemmatizer is not None:
                try:
                    value = lemmatizer.lemmatize(value)
                except LookupError as exc:
                    if not warned_lemma["value"]:
                        LOGGER.warning(
                            "WordNet resource is unavailable; skipping lemmatization. %s",
                            exc,
                        )
                        warned_lemma["value"] = True

            result.append(value)
        return result

    return transform


def preprocess_text(text: Any, config: Dict[str, Any]) -> Tuple[str, List[str]]:
    prep = config["preprocessing"]
    normalize = get_text_normalizer(config)
    transform_tokens = build_token_transformers(config)

    normalized = normalize(text)
    tokens = simple_tokenize(normalized, prep["token_pattern"])
    tokens = transform_tokens(tokens)
    processed_text = " ".join(tokens)
    return processed_text, tokens


def apply_preprocessing(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    LOGGER.info("Running Task 2: text preprocessing")
    text_col = config["data"]["text_column"]
    processed_col = config["preprocessing"]["processed_text_column"]
    tokens_col = config["preprocessing"]["tokens_column"]

    normalize = get_text_normalizer(config)
    transform_tokens = build_token_transformers(config)
    token_pattern = config["preprocessing"]["token_pattern"]

    processed_texts = []
    processed_tokens = []
    for text in df[text_col].tolist():
        normalized = normalize(text)
        tokens = simple_tokenize(normalized, token_pattern)
        tokens = transform_tokens(tokens)
        processed_tokens.append(tokens)
        processed_texts.append(" ".join(tokens))

    result = df.copy()
    result[processed_col] = processed_texts
    result[tokens_col] = [" ".join(tokens) for tokens in processed_tokens]
    result["processed_char_len"] = result[processed_col].astype(str).str.len()
    result["processed_word_count"] = result[processed_col].astype(str).apply(
        lambda text: len(text.split()) if text else 0
    )
    return result


def save_preprocessing_examples(
    df: pd.DataFrame,
    config: Dict[str, Any],
    output_dir: Path,
) -> None:
    text_col = config["data"]["text_column"]
    label_col = config["data"]["label_column"]
    processed_col = config["preprocessing"]["processed_text_column"]
    examples_per_class = config["eda"]["examples_per_class"]
    random_seed = config["data"]["random_seed"]

    examples = []
    for label, group in df.groupby(label_col):
        sample_n = min(examples_per_class, len(group))
        sampled = group.sample(n=sample_n, random_state=random_seed) if sample_n else group
        examples.append(sampled[[label_col, text_col, processed_col]])

    if examples:
        save_csv(pd.concat(examples, ignore_index=True), output_dir / "preprocessing_examples.csv")


def summarize_preprocessing_changes(
    before_df: pd.DataFrame,
    after_df: pd.DataFrame,
    config: Dict[str, Any],
    output_dir: Path,
) -> None:
    label_col = config["data"]["label_column"]
    summary = (
        after_df.groupby(label_col)
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
    save_csv(summary, output_dir / "preprocessed_length_stats_by_sentiment.csv")

    comparison = {
        "raw_mean_word_count": float(before_df["word_count"].mean()),
        "processed_mean_word_count": float(after_df["processed_word_count"].mean()),
        "raw_mean_char_len": float(before_df["char_len"].mean()),
        "processed_mean_char_len": float(after_df["processed_char_len"].mean()),
    }
    save_json(comparison, output_dir / "preprocessing_change_summary.json")


def run_preprocessing(
    df_features: pd.DataFrame,
    config: Dict[str, Any],
    output_dir: Path,
) -> pd.DataFrame:
    processed = apply_preprocessing(df_features, config)
    save_csv(processed, output_dir / "preprocessed_tweets.csv")
    save_preprocessing_examples(processed, config, output_dir)
    summarize_preprocessing_changes(df_features, processed, config, output_dir)
    save_json(config["preprocessing"], output_dir / "preprocessing_config_used.json")
    LOGGER.info("Task 2 outputs saved to %s", output_dir)
    return processed


def main() -> None:
    setup_logging()
    args = parse_args()
    config = load_config(args)
    output_dir = ensure_output_dir(config)
    save_json(config, output_dir / "config_used.json")

    df = load_dataset(config)
    df_features = run_eda(df, config, output_dir)
    run_preprocessing(df_features, config, output_dir)

    LOGGER.info("Done. Main outputs are in: %s", output_dir.resolve())


if __name__ == "__main__":
    main()
