import json
import re

import requests
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
FREE_MODEL = "openrouter/free"

SKILL_CATEGORIES = {
    "Programming": {
        "Python": ["python"],
        "SQL": ["sql"],
        "JavaScript": ["javascript", "js"],
    },
    "Data and ML": {
        "pandas": ["pandas"],
        "scikit-learn": ["scikit-learn", "scikit learn", "sklearn"],
        "Machine Learning": ["machine learning", "ml"],
        "Natural Language Processing": ["natural language processing", "nlp"],
        "Statistics": ["statistics", "statistical"],
    },
    "Cloud and DevOps": {
        "AWS": ["aws", "amazon web services"],
        "Azure": ["azure", "microsoft azure"],
        "Docker": ["docker"],
        "Git": ["git"],
    },
    "Analytics": {
        "Power BI": ["power bi", "powerbi"],
        "Tableau": ["tableau"],
        "Excel": ["excel"],
    },
    "Apps and Databases": {
        "Streamlit": ["streamlit"],
        "PostgreSQL": ["postgresql", "postgres"],
        "MySQL": ["mysql"],
    },
}


def has_phrase(text, phrase):
    pattern = rf"(?<!\w){re.escape(phrase.lower())}(?!\w)"
    return re.search(pattern, text.lower()) is not None


def extract_pdf_text(uploaded_file):
    uploaded_file.seek(0)
    reader = PdfReader(uploaded_file)
    text = "\n".join((page.extract_text() or "") for page in reader.pages)
    if not text.strip():
        raise ValueError(
            "No extractable text was found. Use a text-based PDF or paste the resume text."
        )
    return text


def normalize_aliases(text):
    normalized = text.lower()
    for skills in SKILL_CATEGORIES.values():
        for aliases in skills.values():
            canonical = aliases[0]
            for alias in sorted(aliases[1:], key=len, reverse=True):
                pattern = rf"(?<!\w){re.escape(alias)}(?!\w)"
                normalized = re.sub(pattern, canonical, normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def lexical_similarity(resume_text, job_description):
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    try:
        matrix = vectorizer.fit_transform([resume_text, job_description])
    except ValueError:
        return 0.0
    return float(cosine_similarity(matrix[0:1], matrix[1:2])[0, 0])


def skills_in_text(text):
    found = {}
    for category, skills in SKILL_CATEGORIES.items():
        found[category] = [
            label
            for label, aliases in skills.items()
            if any(has_phrase(text, alias) for alias in aliases)
        ]
    return found


def aliases_for_skill(skill_label):
    for skills in SKILL_CATEGORIES.values():
        if skill_label in skills:
            return skills[skill_label]
    return []


def relevant_evidence(resume_text, matched_skills, limit=5):
    segments = re.split(r"(?<=[.!?])\s+|\n+", resume_text)
    evidence = []
    for segment in segments:
        cleaned = segment.strip()
        if len(cleaned) < 12:
            continue
        if any(
            has_phrase(cleaned, alias)
            for skill in matched_skills
            for alias in aliases_for_skill(skill)
        ):
            evidence.append(cleaned)
        if len(evidence) == limit:
            break
    return evidence


def analyze_match(resume_text, job_description):
    lexical = lexical_similarity(resume_text, job_description)
    alias_aware = lexical_similarity(
        normalize_aliases(resume_text), normalize_aliases(job_description)
    )

    resume_skills = skills_in_text(resume_text)
    required_skills = skills_in_text(job_description)
    rows = []
    matched_all = []
    missing_all = []

    for category, required in required_skills.items():
        if not required:
            continue
        matched = [skill for skill in required if skill in resume_skills[category]]
        missing = [skill for skill in required if skill not in resume_skills[category]]
        matched_all.extend(matched)
        missing_all.extend(missing)
        rows.append(
            {
                "Skill category": category,
                "Matched skills": ", ".join(matched) or "None",
                "Missing skills": ", ".join(missing) or "None",
            }
        )

    required_count = len(matched_all) + len(missing_all)
    coverage = len(matched_all) / required_count if required_count else 0.0
    overall = 0.5 * alias_aware + 0.5 * coverage

    return {
        "lexical_similarity": round(lexical * 100, 1),
        "alias_similarity": round(alias_aware * 100, 1),
        "skill_coverage": round(coverage * 100, 1),
        "overall_score": round(overall * 100, 1),
        "matched_skills": matched_all,
        "missing_skills": missing_all,
        "category_rows": rows,
        "evidence": relevant_evidence(resume_text, matched_all),
    }
BLOCKED_MODEL_WORDS = ("safety", "guard", "moderation", "classifier")


def looks_like_coaching(content):
    bullets = [
        line for line in content.splitlines()
        if line.strip().startswith(("-", "*", "•")) or line.strip()[:2].rstrip(".").isdigit()
    ]
    return len(bullets) >= 3


def get_ai_recommendations(api_key, result, attempts=4):
    signals = {
        "overall_match_percent": result["overall_score"],
        "alias_similarity_percent": result["alias_similarity"],
        "skill_coverage_percent": result["skill_coverage"],
        "matched_skills": result["matched_skills"],
        "missing_skills": result["missing_skills"],
    }

    for _ in range(attempts):
        response = requests.post(
            url=OPENROUTER_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": FREE_MODEL,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are a careful job application coach. Use only the supplied "
                            "scores and skill names. Do not invent experience, employers, "
                            "qualifications, or achievements. Return exactly three concise "
                            "Markdown bullet points with practical improvement suggestions."
                        ),
                    },
                    {"role": "user", "content": json.dumps(signals, indent=2)},
                ],
            },
            timeout=45,
        )
        response.raise_for_status()
        data = response.json()
        model = data.get("model", "")
        content = data["choices"][0]["message"]["content"] or ""

        if any(word in model.lower() for word in BLOCKED_MODEL_WORDS):
            continue
        if looks_like_coaching(content):
            return content, model

    raise ValueError(
        "The free model router did not return usable coaching after several tries. "
        "Click Analyze Match again."
    )