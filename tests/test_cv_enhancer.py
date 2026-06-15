import json
import pytest
from unittest.mock import MagicMock
from agents.cv_enhancer import cv_enhancer_node

CV_PARSED = {
    "skills": ["Python"],
    "experience": [{"title": "Dev", "company": "X", "duration": "1y", "bullets": ["Built REST APIs"]}],
    "education": [],
    "summary": None,
}
GAPS = ["Missing skill: Docker", "Missing keyword: CI/CD"]
ENHANCED = {"enhanced_bullets": ["Designed and deployed 5 REST APIs, reducing latency by 30%"]}


@pytest.fixture
def mock_chain(mocker):
    mock = mocker.patch("agents.cv_enhancer._chain")
    mock.invoke.return_value = MagicMock(content=json.dumps(ENHANCED))
    return mock


def test_cv_enhancer_returns_enhanced_bullets_key(mock_chain):
    result = cv_enhancer_node({"cv_parsed": CV_PARSED, "gaps": GAPS})
    assert "enhanced_bullets" in result


def test_cv_enhancer_returns_list_of_strings(mock_chain):
    result = cv_enhancer_node({"cv_parsed": CV_PARSED, "gaps": GAPS})
    assert isinstance(result["enhanced_bullets"], list)
    assert all(isinstance(b, str) for b in result["enhanced_bullets"])


def test_cv_enhancer_raises_on_invalid_json(mocker):
    mock = mocker.patch("agents.cv_enhancer._chain")
    mock.invoke.return_value = MagicMock(content="bad json}")
    with pytest.raises(ValueError, match="cv_enhancer"):
        cv_enhancer_node({"cv_parsed": CV_PARSED, "gaps": GAPS})
