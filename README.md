# Job Application Skill Matcher

A Streamlit app that compares a resume with a job description and explains the match, with optional AI coaching that never sees your resume text.

**Live demo:** https://jobskill-matcher.streamlit.app

## What it does

- Accepts pasted resume text or a text-based PDF (processed in memory, never saved)
- Scores the match three ways: lexical TF-IDF similarity, alias-aware similarity, and skill coverage
- Shows matched skills, missing skills, and the resume sentences that support each match
- Optionally requests three improvement suggestions from OpenRouter

## Why alias-aware matching?

A plain keyword check treats "ML" and "machine learning" as different terms, so the same skill can look missing. A small skill taxonomy normalizes known aliases before scoring, which closes that gap and keeps the logic inspectable.

## Privacy design

- Uploaded files are held in memory only
- The AI coach receives only scores and skill names, never resume or job-description text
- The overall score is a project-defined blend, not a hiring score

## Robustness test

`robustness_test.py` compares word TF-IDF, alias-aware TF-IDF, and character n-grams on abbreviations and typos.

| Case          | Word TF-IDF | Alias-aware | Character n-grams |
|---------------|-------------|-------------|-------------------|
| Abbreviations | 0.0%        | 100.0%      | 0.0%              |
| Typos         | 0.0%        | 0.0%        | 36.6%             |

Alias normalization fixes abbreviations but not misspellings; character n-grams partly recover misspellings but not abbreviations. These are two small synthetic cases meant to show the pattern, not a benchmark.

## Run locally

    python -m venv .venv
    .venv\Scripts\Activate.ps1
    python -m pip install -r requirements.txt
    python -m streamlit run app.py

To enable AI suggestions, copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` and add your OpenRouter API key.

## Limitations

- Skill detection only covers the skills in the taxonomy
- Scanned or image-only PDFs are not supported (no OCR)
- Free-tier AI availability and quality can vary

## Tech

Python, Streamlit, scikit-learn, pandas, pypdf, OpenRouter