from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

from graph.state import CVState
from tools.config import GEMINI_MODEL
from tools.llm_utils import parse_json_response

_llm = ChatGoogleGenerativeAI(model=GEMINI_MODEL, temperature=0)

_prompt = ChatPromptTemplate.from_messages([
    ("system", (
        "You are a CV parsing expert. Extract structured information from the CV text. "
        "Return ONLY valid JSON with these keys: "
        "skills (list of strings), "
        "experience (list of dicts with keys: title, company, duration, bullets), "
        "education (list of dicts with keys: degree, institution, year), "
        "summary (string or null)."
    )),
    ("human", "Parse this CV:\n\n{cv_raw}"),
])

_chain = _prompt | _llm


def cv_parser_node(state: CVState) -> CVState:
    response = _chain.invoke({"cv_raw": state["cv_raw"]})
    cv_parsed = parse_json_response(response, "cv_parser")
    return {"cv_parsed": cv_parsed}
