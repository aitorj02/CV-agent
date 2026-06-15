import json
import pytest
from unittest.mock import MagicMock
from agents.gap_analyzer import gap_analyzer_node

CV_PARSED = {"skills": ["Python"], "experience": [], "education": [], "summary": None}
JD_PARSED = {"required_skills": ["Python", "Docker"], "keywords": ["CI/CD"]}

GAPS_RESPONSE = {
    "missing_skills": ["Docker"],
    "weak_areas": ["Bullet points lack quantified impact"],
    "missing_keywords": ["CI/CD"],
}


@pytest.fixture
def mock_chain(mocker):
    mock = mocker.patch("agents.gap_analyzer._chain")
    mock.invoke.return_value = MagicMock(content=json.dumps(GAPS_RESPONSE))
    return mock


def test_gap_analyzer_returns_gaps_key(mock_chain):
    result = gap_analyzer_node({"cv_parsed": CV_PARSED, "jd_parsed": JD_PARSED})
    assert "gaps" in result


def test_gap_analyzer_flattens_all_gap_types(mock_chain):
    result = gap_analyzer_node({"cv_parsed": CV_PARSED, "jd_parsed": JD_PARSED})
    gaps = result["gaps"]
    assert any("Missing skill" in g for g in gaps)
    assert any("Weak area" in g for g in gaps)
    assert any("Missing keyword" in g for g in gaps)


def test_gap_analyzer_raises_on_invalid_json(mocker):
    mock = mocker.patch("agents.gap_analyzer._chain")
    mock.invoke.return_value = MagicMock(content="not json")
    with pytest.raises(ValueError, match="gap_analyzer"):
        gap_analyzer_node({"cv_parsed": CV_PARSED, "jd_parsed": JD_PARSED})
