from typing import TypedDict


class CVState(TypedDict, total=False):
    cv_raw: str
    jd_raw: str
    cv_parsed: dict
    jd_parsed: dict
    gaps: list[str]
    enhanced_bullets: list[str]
    ats_score_before: int
    ats_score_after: int
    final_report: str
