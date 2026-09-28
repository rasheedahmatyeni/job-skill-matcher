from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from matcher import lexical_similarity, normalize_aliases


def char_ngram_similarity(text_a, text_b):
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5))
    try:
        matrix = vectorizer.fit_transform([text_a.lower(), text_b.lower()])
    except ValueError:
        return 0.0
    return float(cosine_similarity(matrix[0:1], matrix[1:2])[0, 0])


def word_similarity(text_a, text_b):
    return lexical_similarity(text_a, text_b)


def alias_similarity(text_a, text_b):
    return lexical_similarity(normalize_aliases(text_a), normalize_aliases(text_b))


CASES = {
    "Abbreviations": (
        "ML, NLP, AWS",
        "machine learning, natural language processing, amazon web services",
    ),
    "Typos": (
        "machne lerning, pandaz, pythn, streamlt",
        "machine learning, pandas, python, streamlit",
    ),
}

METHODS = {
    "Word TF-IDF": word_similarity,
    "Alias-aware TF-IDF": alias_similarity,
    "Character n-grams": char_ngram_similarity,
}

print(f"{'Case':<15}{'Method':<22}{'Similarity':>10}")
print("-" * 47)
for case_name, (resume_side, job_side) in CASES.items():
    for method_name, method in METHODS.items():
        score = method(resume_side, job_side)
        print(f"{case_name:<15}{method_name:<22}{score:>10.1%}")
    print()