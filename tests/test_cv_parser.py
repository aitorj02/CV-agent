import json
import pytest
from unittest.mock import MagicMock
from agents.cv_parser import cv_parser_node


PARSED_CV = {
    "skills": ["Python", "FastAPI"],
    "experience": [{"title": "Engineer", "company": "Acme", "duration": "2y", "bullets": ["Built APIs"]}],
    "education": [{"degree": "BSc CS", "institution": "MIT", "year": "2021"}],
    "summary": None,
}


@pytest.fixture
def mock_chain(mocker):
    mock = mocker.patch("agents.cv_parser._chain")
    mock.invoke.return_value = MagicMock(content=json.dumps(PARSED_CV))
    return mock


def test_cv_parser_returns_cv_parsed_key(mock_chain):
    result = cv_parser_node({"cv_raw": "John Doe\nPython engineer"})
    assert "cv_parsed" in result


def test_cv_parser_extracts_skills(mock_chain):
    result = cv_parser_node({"cv_raw": "John Doe\nPython engineer"})
    assert result["cv_parsed"]["skills"] == ["Python", "FastAPI"]


def test_cv_parser_raises_on_invalid_json(mocker):
    mock = mocker.patch("agents.cv_parser._chain")
    mock.invoke.return_value = MagicMock(content="not valid json at all")
    with pytest.raises(ValueError, match="cv_parser"):
        cv_parser_node({"cv_raw": "some cv"})
