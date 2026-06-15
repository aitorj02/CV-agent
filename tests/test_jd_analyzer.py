import json
import pytest
from unittest.mock import MagicMock
from agents.jd_analyzer import jd_analyzer_node


PARSED_JD = {
    "required_skills": ["Python", "FastAPI"],
    "nice_to_have_skills": ["LangGraph"],
    "tech_stack": ["Python", "PostgreSQL", "Docker"],
    "keywords": ["backend", "REST", "CI/CD"],
    "role_summary": "Senior Backend Engineer",
}


@pytest.fixture
def mock_chain(mocker):
    mock = mocker.patch("agents.jd_analyzer._chain")
    mock.invoke.return_value = MagicMock(content=json.dumps(PARSED_JD))
    return mock


def test_jd_analyzer_returns_jd_parsed_key(mock_chain):
    result = jd_analyzer_node({"jd_raw": "We need a Python engineer..."})
    assert "jd_parsed" in result


def test_jd_analyzer_extracts_required_skills(mock_chain):
    result = jd_analyzer_node({"jd_raw": "We need a Python engineer..."})
    assert "Python" in result["jd_parsed"]["required_skills"]


def test_jd_analyzer_raises_on_invalid_json(mocker):
    mock = mocker.patch("agents.jd_analyzer._chain")
    mock.invoke.return_value = MagicMock(content="{ broken json")
    with pytest.raises(ValueError, match="jd_analyzer"):
        jd_analyzer_node({"jd_raw": "some jd"})
