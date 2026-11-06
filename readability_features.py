
import re
import numpy as np

from scipy.sparse import hstack, csr_matrix

def count_syllables(word):
    word = word.lower()
    groups = re.findall(r"[aeiouy]+", word)
    count = max(1, len(groups))

    if (
        len(word) > 3
        and word.endswith("e")
        and not word.endswith("le")
        and count > 1
    ):
        count -= 1

    return count


def extract_stats(texts):
    records = []

    for text in texts:
        text = str(text)

        words = re.findall(
            r"[A-Za-z]+(?:'[A-Za-z]+)?", text
        )

        n_words = max(len(words), 1)

        sentences = max(
            len(re.findall(r"[.!?]+", text)), 1
        )

        lengths = [len(w) for w in words]
        syllables = [
            count_syllables(w) for w in words
        ]

        n_syllables = sum(syllables)
        avg_syllables = n_syllables / n_words
        words_per_sentence = n_words / sentences

        flesch = (
            206.835
            - 1.015 * words_per_sentence
            - 84.6 * avg_syllables
        )

        fk_grade = (
            0.39 * words_per_sentence
            + 11.8 * avg_syllables
            - 15.59
        )

        records.append([
            len(words),
            sentences,
            len(text),
            np.mean(lengths) if lengths else 0,
            np.median(lengths) if lengths else 0,
            words_per_sentence,
            len(set(w.lower() for w in words)) / n_words,
            sum(len(w) >= 7 for w in words) / n_words,
            sum(c.isupper() for c in text)
            / max(len(text), 1),
            sum(c.isdigit() for c in text),
            text.count(","),
            text.count(";"),
            text.count("?"),
            text.count("!"),
            sum(c in ".,;:!?-" for c in text),
            avg_syllables,
            sum(s >= 3 for s in syllables) / n_words,
            flesch,
            fk_grade
        ])

    return np.asarray(records, dtype=np.float32)


def predict_readability(bundle, texts):
    texts = [str(t) for t in texts]

    word = bundle["word_vectorizer"].transform(texts)
    char = bundle["char_vectorizer"].transform(texts)

    stats = extract_stats(texts)

    scaled = bundle["stats_scaler"].transform(stats)

    weight = float(bundle["stats_weight"])

    matrix = hstack([
        word,
        char,
        csr_matrix(scaled * weight)
    ], format="csr")

    predictions = bundle["model"].predict(matrix)

    return np.asarray(predictions, dtype=np.float64)
