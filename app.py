import pandas as pd
import requests
import streamlit as st

from matcher import analyze_match, extract_pdf_text, get_ai_recommendations

SAMPLE_RESUME = """Data Analyst
Built ML pipelines in Python and pandas to forecast product demand.
Created SQL reports and automated weekly data-quality checks.
Deployed an internal Streamlit dashboard for operations teams.
Used Git and Docker to package and review analytics work.
"""

SAMPLE_JOB_DESCRIPTION = """We are hiring a Junior Machine Learning Engineer.
The role requires Python, machine learning, pandas, scikit-learn, SQL,
Streamlit, and AWS. The candidate should communicate analytical findings
and improve repeatable data workflows.
"""

st.title("Job Application Skill Matcher")
st.write("Compare a resume with a job description using explainable NLP plus optional AI coaching.")
st.caption(
    "Privacy: uploaded files are processed in memory and are not saved by this app. "
    "Only scores and skill names are sent to the AI coach, never your resume or job text. "
    "Please use synthetic or non-sensitive text when trying the demo."
)

uploaded_file = st.file_uploader(
    "Upload a text-based PDF resume (optional)", type="pdf", max_upload_size=5
)
resume_fallback = st.text_area("Resume text used when no PDF is uploaded", value=SAMPLE_RESUME, height=180)
job_description = st.text_area("Job description", value=SAMPLE_JOB_DESCRIPTION, height=220)

pdf_text = None
if uploaded_file is not None:
    try:
        pdf_text = extract_pdf_text(uploaded_file)
    except (ValueError, OSError) as exc:
        st.error(str(exc))
    else:
        st.success(f"Extracted {len(pdf_text):,} characters from the PDF.")

if st.button("Analyze Match", type="primary", width="stretch"):
    resume_text = pdf_text if uploaded_file is not None else resume_fallback

    if not resume_text or not resume_text.strip():
        st.error("Provide extractable resume text before running the analysis.")
    elif not job_description.strip():
        st.error("Paste a job description before running the analysis.")
    else:
        result = analyze_match(resume_text, job_description)

        st.subheader("Match report")
        lexical_col, alias_col, coverage_col, overall_col = st.columns(4)
        lexical_col.metric("Lexical similarity", f"{result['lexical_similarity']}%")
        alias_col.metric("Alias-aware similarity", f"{result['alias_similarity']}%")
        coverage_col.metric("Skill coverage", f"{result['skill_coverage']}%")
        overall_col.metric("Overall match", f"{result['overall_score']}%")

        st.warning(
            "The overall score is a project-defined blend, not a standardized "
            "recruiting or hiring score."
        )

        st.subheader("Skills by category")
        if result["category_rows"]:
            st.dataframe(pd.DataFrame(result["category_rows"]), hide_index=True)
        else:
            st.write("No skills from the current taxonomy were found in the job description.")

        st.subheader("Matched and missing skills")
        st.write("Matched:", ", ".join(result["matched_skills"]) or "None")
        st.write("Missing:", ", ".join(result["missing_skills"]) or "None")

        with st.expander("Relevant resume evidence", expanded=True):
            if result["evidence"]:
                for sentence in result["evidence"]:
                    st.write(f"- {sentence}")
            else:
                st.write("No evidence sentence matched the current skill taxonomy.")

        st.subheader("AI improvement suggestions")
        try:
            api_key = st.secrets["OPENROUTER_API_KEY"]
        except (FileNotFoundError, KeyError):
            api_key = ""

        if not api_key or api_key == "your-api-key-here":
            st.warning(
                "Add OPENROUTER_API_KEY to .streamlit/secrets.toml to enable "
                "optional AI suggestions."
            )
        else:
            try:
                recommendations, model_used = get_ai_recommendations(api_key, result)
            except requests.RequestException as exc:
                st.error(f"OpenRouter request failed: {exc}")
            except (KeyError, TypeError, ValueError) as exc:
                st.error(f"OpenRouter returned an unexpected response: {exc}")
            else:
                st.markdown(recommendations)
                st.write("Free model selected:", model_used)
                st.warning(
                    "Verify every suggestion before editing a resume. The model has "
                    "not seen your raw experience and must not be used to invent claims."
                )