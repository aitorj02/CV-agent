from langgraph.graph import StateGraph, END

from graph.state import CVState
from agents.cv_parser import cv_parser_node
from agents.jd_analyzer import jd_analyzer_node
from agents.gap_analyzer import gap_analyzer_node
from agents.cv_enhancer import cv_enhancer_node
from agents.ats_scorer import ats_scorer_node


def build_graph() -> StateGraph:
    builder = StateGraph(CVState)

    builder.add_node("cv_parser", cv_parser_node)
    builder.add_node("jd_analyzer", jd_analyzer_node)
    builder.add_node("gap_analyzer", gap_analyzer_node)
    builder.add_node("cv_enhancer", cv_enhancer_node)
    builder.add_node("ats_scorer", ats_scorer_node)

    # cv_parser and jd_analyzer both run first (sequential for simplicity in MVP)
    builder.set_entry_point("cv_parser")
    builder.add_edge("cv_parser", "jd_analyzer")
    builder.add_edge("jd_analyzer", "gap_analyzer")
    builder.add_edge("gap_analyzer", "cv_enhancer")
    builder.add_edge("cv_enhancer", "ats_scorer")
    builder.add_edge("ats_scorer", END)

    return builder.compile()


graph = build_graph()
