from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

from graph.state import CVState
from tools.config import GEMINI_MODEL
from tools.llm_utils import parse_json_response

_llm = ChatGoogleGenerativeAI(model=GEMINI_MODEL, temperature=0)

_prompt = ChatPromptTemplate.from_messages([
    ("system", (
        "You are a job description analyst. Extract structured requirements from the job posting. "
        "Return ONLY valid JSON with these keys: "
        "required_skills (list of strings), "
        "nice_to_have_skills (list of strings), "
        "tech_stack (list of strings), "
        "keywords (list of strings important for ATS), "
        "role_summary (string)."
    )),
    ("human", "Analyze this job description:\n\n{jd_raw}"),
])

_chain = _prompt | _llm


def jd_analyzer_node(state: CVState) -> CVState:
    response = _chain.invoke({"jd_raw": state["jd_raw"]})
    jd_parsed = parse_json_response(response, "jd_analyzer")
    return {"jd_parsed": jd_parsed}
