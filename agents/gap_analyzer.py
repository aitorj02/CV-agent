import json
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

from graph.state import CVState
from tools.config import GEMINI_MODEL
from tools.llm_utils import parse_json_response

_llm = ChatGoogleGenerativeAI(model=GEMINI_MODEL, temperature=0)

_prompt = ChatPromptTemplate.from_messages([
    ("system", (
        "You are a career coach. Compare a candidate's CV against a job description "
        "and identify gaps. Return ONLY valid JSON with these keys: "
        "missing_skills (list of strings), "
        "weak_areas (list of strings describing bullet points or sections that need strengthening), "
        "missing_keywords (list of ATS keywords from the JD absent in the CV)."
    )),
    ("human", (
        "CV data:\n{cv_parsed}\n\n"
        "Job description data:\n{jd_parsed}\n\n"
        "Identify all gaps."
    )),
])

_chain = _prompt | _llm


def gap_analyzer_node(state: CVState) -> CVState:
    response = _chain.invoke({
        "cv_parsed": json.dumps(state["cv_parsed"], indent=2),
        "jd_parsed": json.dumps(state["jd_parsed"], indent=2),
    })
    gaps_data = parse_json_response(response, "gap_analyzer")
    gaps = (
        [f"Missing skill: {s}" for s in gaps_data.get("missing_skills", [])]
        + [f"Weak area: {w}" for w in gaps_data.get("weak_areas", [])]
        + [f"Missing keyword: {k}" for k in gaps_data.get("missing_keywords", [])]
    )
    return {"gaps": gaps}
