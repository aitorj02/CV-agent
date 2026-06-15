import json
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

from graph.state import CVState
from tools.config import GEMINI_MODEL
from tools.llm_utils import parse_json_response

_llm = ChatGoogleGenerativeAI(model=GEMINI_MODEL, temperature=0)

_prompt = ChatPromptTemplate.from_messages([
    ("system", (
        "You are an ATS (Applicant Tracking System) scoring engine. "
        "Score how well a CV matches a job description on a scale of 0–100. "
        "Consider keyword coverage, skill match, and relevance. "
        "Return ONLY valid JSON with keys: score (integer 0-100), reasoning (string)."
    )),
    ("human", (
        "Job description keywords and requirements:\n{jd_parsed}\n\n"
        "CV content to score:\n{cv_content}\n\n"
        "Provide a match score."
    )),
])

_chain = _prompt | _llm


def _score(jd_parsed: dict, cv_content: str) -> tuple[int, str]:
    response = _chain.invoke({
        "jd_parsed": json.dumps(jd_parsed, indent=2),
        "cv_content": cv_content,
    })
    result = parse_json_response(response, "ats_scorer")
    return result.get("score", 0), result.get("reasoning", "")


def ats_scorer_node(state: CVState) -> CVState:
    jd_parsed = state["jd_parsed"]
    original_cv = json.dumps(state["cv_parsed"], indent=2)
    enhanced_cv = "\n".join(state["enhanced_bullets"])

    score_before, _ = _score(jd_parsed, original_cv)
    score_after, reasoning = _score(jd_parsed, enhanced_cv)

    final_report = _build_report(state, score_before, score_after, reasoning)
    return {
        "ats_score_before": score_before,
        "ats_score_after": score_after,
        "final_report": final_report,
    }


def _build_report(state: CVState, score_before: int, score_after: int, reasoning: str) -> str:
    lines = [
        "# CV Enhancement Report",
        "",
        f"**ATS Score: {score_before} → {score_after}**",
        f"_{reasoning}_",
        "",
        "## Gaps Identified",
        *[f"- {g}" for g in state["gaps"]],
        "",
        "## Enhanced Bullet Points",
        *[f"- {b}" for b in state["enhanced_bullets"]],
    ]
    return "\n".join(lines)
