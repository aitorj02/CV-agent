import json
import pytest
from unittest.mock import MagicMock, call
from agents.ats_scorer import ats_scorer_node

CV_PARSED = {"skills": ["Python"], "experience": [], "education": [], "summary": None}
JD_PARSED = {"required_skills": ["Python", "Docker"], "keywords": ["CI/CD"]}
ENHANCED_BULLETS = ["Deployed Docker containers, cutting release time by 40%"]

SCORE_BEFORE = {"score": 45, "reasoning": "Missing Docker and CI/CD keywords."}
SCORE_AFTER = {"score": 78, "reasoning": "Docker now present, keyword coverage improved."}


@pytest.fixture
def mock_chain(mocker):
    mock = mocker.patch("agents.ats_scorer._chain")
    mock.invoke.side_effect = [
        MagicMock(content=json.dumps(SCORE_BEFORE)),
        MagicMock(content=json.dumps(SCORE_AFTER)),
    ]
    return mock


def test_ats_scorer_returns_all_keys(mock_chain):
    result = ats_scorer_node({
        "cv_parsed": CV_PARSED,
        "jd_parsed": JD_PARSED,
        "enhanced_bullets": ENHANCED_BULLETS,
        "gaps": [],
    })
    assert "ats_score_before" in result
    assert "ats_score_after" in result
    assert "final_report" in result


def test_ats_scorer_score_improves(mock_chain):
    result = ats_scorer_node({
        "cv_parsed": CV_PARSED,
        "jd_parsed": JD_PARSED,
        "enhanced_bullets": ENHANCED_BULLETS,
        "gaps": [],
    })
    assert result["ats_score_before"] == 45
    assert result["ats_score_after"] == 78


def test_ats_scorer_report_contains_scores(mock_chain):
    result = ats_scorer_node({
        "cv_parsed": CV_PARSED,
        "jd_parsed": JD_PARSED,
        "enhanced_bullets": ENHANCED_BULLETS,
        "gaps": ["Missing skill: Docker"],
    })
    assert "45" in result["final_report"]
    assert "78" in result["final_report"]
