"""
Learning scaffold for NLP Project 1.4, Tasks 1 and 2.

Important:
    The assignment states that generative AI tools must not be used to generate
    implementation code for the submitted project. This file is therefore a
    non-submittable scaffold: use it as a checklist and structure guide, then
    write your own implementation.

Goal:
    Task 1: Exploratory Data Analysis
    Task 2: Text Preprocessing

Suggested workflow:
    1. Copy the configuration shape into your own script.
    2. Implement each TODO yourself.
    3. Run the script and save tables/plots for the report.
    4. Fill the experiment summary sheet with the preprocessing choices.
"""


# ---------------------------------------------------------------------------
# Configuration template
# ---------------------------------------------------------------------------
#
# Keep all adjustable parameters in one place. You can replace this commented
# template with your own dataclass, argparse setup, YAML loader, or plain dict.
#
# CONFIG = {
#     "data": {
#         "csv_path": "TweetSentiment/TweetSentiment.csv",
#         "text_column": "text",
#         "label_column": "sentiment",
#         "sample_size": None,       # e.g. 5000 for quick experiments
#         "random_seed": 42,
#     },
#     "eda": {
#         "output_dir": "outputs/task1_eda",
#         "top_n_words": 30,
#         "top_n_ngrams": 20,
#         "ngram_range": (1, 2),
#         "make_plots": True,
#         "make_wordcloud": False,
#     },
#     "preprocessing": {
#         "lowercase": True,
#         "strip_whitespace": True,
#         "remove_urls": True,
#         "remove_user_mentions": True,
#         "remove_hashtag_symbol": True,
#         "remove_punctuation": False,
#         "remove_numbers": False,
#         "remove_stopwords": False,
#         "stemming": False,
#         "lemmatization": False,
#         "min_token_length": 1,
#     },
# }


# ---------------------------------------------------------------------------
# Task 1 checklist: Exploratory Data Analysis
# ---------------------------------------------------------------------------
#
# TODO: Load the CSV file.
#       - Keep only the text and sentiment columns.
#       - Drop or inspect rows with missing text/labels.
#       - Optionally sample rows if training later is too slow.
#
# TODO: Basic dataset statistics.
#       Suggested values to report:
#       - number of examples
#       - number of missing texts
#       - number of unique sentiment labels
#       - examples per sentiment class
#       - class percentages
#
# TODO: Text length analysis.
#       Suggested values:
#       - character length per tweet
#       - token count per tweet
#       - mean, median, min, max by class
#       Suggested plots:
#       - histogram of tweet lengths
#       - boxplot of token counts by sentiment class
#
# TODO: Vocabulary analysis.
#       Suggested values:
#       - total vocabulary size
#       - vocabulary size by sentiment class
#       - most frequent words overall
#       - most frequent words per class
#       - frequent bigrams if useful
#
# TODO: Interpretation notes for the report.
#       Write down observations such as:
#       - Is the dataset balanced?
#       - Are neutral tweets harder because they are vague or short?
#       - Are positive/negative tweets associated with obvious sentiment words?
#       - Could URLs, mentions, hashtags, casing, punctuation, or repeated letters
#         influence the classifier?
#       - Which preprocessing choices does the EDA suggest?


# ---------------------------------------------------------------------------
# Task 2 checklist: Text Preprocessing
# ---------------------------------------------------------------------------
#
# TODO: Implement a preprocessing function controlled by CONFIG.
#       Suggested function shape:
#
#       def preprocess_text(text, config):
#           ...
#           return processed_text
#
# TODO: Consider these operations.
#       - lowercase
#       - trim extra whitespace
#       - remove or replace URLs
#       - remove or replace @mentions
#       - decide how to handle hashtags
#       - decide whether punctuation should be removed
#       - decide whether numbers should be removed
#       - tokenize
#       - optionally remove stop words
#       - optionally apply stemming or lemmatization
#
# TODO: Be careful with sentiment-bearing text.
#       Examples:
#       - Do not blindly remove all punctuation if "!!!" or ":(" may signal emotion.
#       - Stop-word removal can remove useful words such as "not".
#       - Stemming/lemmatization may help sparsity but can also distort slang.
#       - Twitter text often contains emojis, elongated words, hashtags, and mentions.
#
# TODO: Save or print examples before/after preprocessing.
#       Include examples from positive, neutral, and negative classes.
#       These examples are useful for the report and for debugging.
#
# TODO: Record your final preprocessing design.
#       In the report, briefly explain:
#       - which steps you used
#       - why they are suitable for this dataset
#       - which steps you skipped
#       - possible risks introduced by preprocessing


# ---------------------------------------------------------------------------
# Suggested script structure
# ---------------------------------------------------------------------------
#
# def load_dataset(config):
#     """Load and validate the dataset."""
#     raise NotImplementedError("Write this function yourself.")
#
#
# def run_eda(dataframe, config):
#     """Compute Task 1 statistics and create optional plots."""
#     raise NotImplementedError("Write this function yourself.")
#
#
# def preprocess_text(text, config):
#     """Apply Task 2 preprocessing choices to one text."""
#     raise NotImplementedError("Write this function yourself.")
#
#
# def apply_preprocessing(dataframe, config):
#     """Create a processed_text column using preprocess_text."""
#     raise NotImplementedError("Write this function yourself.")
#
#
# def main():
#     """Run Task 1 and Task 2 with the selected configuration."""
#     raise NotImplementedError("Write this function yourself.")
#
#
# if __name__ == "__main__":
#     main()

