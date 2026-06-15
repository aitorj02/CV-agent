"""CLI entry point — useful for quick testing without the Streamlit UI."""
import sys
from dotenv import load_dotenv

load_dotenv()

from graph.graph import graph  # noqa: E402 — load after dotenv


def run(cv_text: str, jd_text: str) -> None:
    print("Running CV enhancement pipeline...\n")
    result = graph.invoke({"cv_raw": cv_text, "jd_raw": jd_text})
    print(result["final_report"])


if __name__ == "__main__":
    # Quick smoke test with placeholder text
    sample_cv = """
    John Doe | john@example.com

    EXPERIENCE
    Software Engineer at Acme Corp (2021-2024)
    - Built REST APIs using Flask
    - Worked with SQL databases
    - Collaborated with product team

    SKILLS: Python, Flask, SQL, Git

    EDUCATION
    BSc Computer Science, University of Example, 2021
    """

    sample_jd = """
    Senior Backend Engineer

    We are looking for a Senior Backend Engineer with:
    - 3+ years Python experience
    - FastAPI or Django REST framework
    - Experience with LangChain or LLM integrations
    - PostgreSQL and Redis
    - Docker and Kubernetes
    - CI/CD pipelines (GitHub Actions)

    Nice to have: LangGraph, RAG systems, vector databases
    """

    run(sample_cv, sample_jd)
