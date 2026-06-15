import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from graph.graph import graph  # noqa: E402
from tools.pdf_loader import extract_text_from_pdf  # noqa: E402

st.set_page_config(page_title="CV Enhancer", page_icon="📄", layout="wide")
st.title("CV Enhancement Agent")
st.caption("Powered by LangChain · LangGraph · Claude")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Your CV")
    uploaded_pdf = st.file_uploader("Upload CV (PDF)", type=["pdf"])
    cv_text = st.text_area("Or paste CV text", height=300, placeholder="Paste your CV here...")
    if uploaded_pdf:
        import tempfile, os
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_pdf.read())
            tmp_path = tmp.name
        cv_text = extract_text_from_pdf(tmp_path)
        os.unlink(tmp_path)
        st.success("PDF loaded.")

with col2:
    st.subheader("Job Description")
    jd_text = st.text_area("Paste the job description", height=300, placeholder="Paste the job posting here...")

run = st.button("Analyze & Enhance", type="primary", disabled=not (cv_text and jd_text))

if run:
    with st.spinner("Running agent pipeline..."):
        result = graph.invoke({"cv_raw": cv_text, "jd_raw": jd_text})

    tab1, tab2, tab3 = st.tabs(["Gaps", "Enhanced Bullets", "ATS Score"])

    with tab1:
        st.subheader("Identified Gaps")
        for gap in result["gaps"]:
            st.markdown(f"- {gap}")

    with tab2:
        st.subheader("Enhanced Bullet Points")
        for bullet in result["enhanced_bullets"]:
            st.markdown(f"- {bullet}")

    with tab3:
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Before", f"{result['ats_score_before']} / 100")
        with col_b:
            st.metric("After", f"{result['ats_score_after']} / 100",
                      delta=result['ats_score_after'] - result['ats_score_before'])

    with st.expander("Full Report"):
        st.markdown(result["final_report"])
