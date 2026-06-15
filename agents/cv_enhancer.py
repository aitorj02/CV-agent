import json
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

from graph.state import CVState
from tools.config import GEMINI_MODEL
from tools.llm_utils import parse_json_response

_llm = ChatGoogleGenerativeAI(model=GEMINI_MODEL, temperature=0.3)

_prompt = ChatPromptTemplate.from_messages([
    ("system", (
        "You are an expert CV writer. Rewrite the candidate's experience bullet points "
        "to better match the job requirements, incorporating missing keywords naturally. "
        "Use strong action verbs, quantify impact where possible, and keep each bullet "
        "under 20 words. Return ONLY valid JSON with key: "
        "enhanced_bullets (list of strings, one per rewritten bullet)."
    )),
    ("human", (
        "Original CV experience bullets:\n{original_bullets}\n\n"
        "Gaps to address:\n{gaps}\n\n"
        "Rewrite and improve the bullets."
    )),
])

_chain = _prompt | _llm


def cv_enhancer_node(state: CVState) -> CVState:
    experience = state["cv_parsed"].get("experience", [])
    original_bullets = [
        bullet
        for job in experience
        for bullet in job.get("bullets", [])
    ]
    response = _chain.invoke({
        "original_bullets": json.dumps(original_bullets, indent=2),
        "gaps": "\n".join(state["gaps"]),
    })
    result = parse_json_response(response, "cv_enhancer")
    return {"enhanced_bullets": result.get("enhanced_bullets", [])}
