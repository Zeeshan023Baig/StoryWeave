"""
services/learner_service.py
===========================
Learner profile service for StoryWeave.

Responsibilities:
- Load / save / update learner profiles from the database.
- Compute concept mastery from quiz results (per user, per document).
- Determine estimated learner level from mastery data.
- Identify weak and strong concepts for adaptive prompting.
- Save quiz attempts persistently.
- Build the learner-profile context string injected into LLM prompts.

All operations are scoped to a user_id. Cross-user contamination is prevented
by always passing user_id to every database query.
"""

from database.database import (
    get_learner_profile,
    update_learner_profile,
    upsert_concept_mastery,
    get_concept_mastery,
    get_all_doc_mastery_for_user,
    insert_quiz_attempt,
    get_quiz_history,
    get_recent_quiz_score,
    log_interaction,
    insert_learner_profile,
)


# ── Constants ──────────────────────────────────────────────────────────────────
_MASTERY_THRESHOLDS = {
    "weak":   60.0,   # < 60% → weak concept
    "strong": 80.0,   # >= 80% → strong concept
}

_LEVEL_THRESHOLDS = {
    "Beginner":     40.0,
    "Intermediate": 70.0,
    "Advanced":     85.0,
}


# ── Profile Loading ────────────────────────────────────────────────────────────

def load_profile(user_id: str, db_path: str | None = None) -> dict:
    """
    Load learner profile for user_id from the database.
    If no profile exists yet, create a default one and return it.

    Returns a dict with:
        estimated_level, overall_mastery, preferred_modality, updated_at
    """
    kwargs = {"db_path": db_path} if db_path else {}
    profile = get_learner_profile(user_id, **kwargs)
    if profile is None:
        # Auto-create if somehow missing (defensive)
        insert_learner_profile(user_id, **kwargs)
        profile = get_learner_profile(user_id, **kwargs)
    return profile or {
        "estimated_level": "Intermediate",
        "overall_mastery": 0.0,
        "preferred_modality": "Not enough data",
        "updated_at": None,
    }


# ── Mastery Calculation ────────────────────────────────────────────────────────

def compute_concept_mastery(quiz_questions: list, user_answers: dict) -> dict:
    """
    Derive per-concept mastery from a submitted quiz.

    quiz_questions : list of question dicts (each has 'question', 'options',
                     'correct_answer', optionally 'concept')
    user_answers   : {question_index: selected_answer_str}

    Returns: {concept_name: mastery_percentage}
    The concept name is extracted from the question dict's 'concept' key when
    available; otherwise falls back to the question text (truncated).
    """
    if not quiz_questions:
        return {}

    concept_scores: dict[str, list[bool]] = {}
    for i, q in enumerate(quiz_questions):
        # Use explicit concept tag when available; fall back to short question text
        concept = (
            q.get("concept")
            or q.get("question", f"Concept {i + 1}")[:60]
        )
        is_correct = (user_answers.get(i) == q.get("correct_answer"))
        concept_scores.setdefault(concept, []).append(is_correct)

    return {
        concept: round(sum(results) / len(results) * 100, 1)
        for concept, results in concept_scores.items()
        if results
    }


def estimate_level_from_mastery(overall_mastery: float) -> str:
    """Map overall mastery percentage to a learner level string."""
    if overall_mastery >= _LEVEL_THRESHOLDS["Advanced"]:
        return "Advanced"
    if overall_mastery >= _LEVEL_THRESHOLDS["Intermediate"]:
        return "Intermediate"
    return "Beginner"


def _classify_concepts(concept_mastery_list: list) -> tuple[list, list]:
    """
    Split concept mastery records into (weak_concepts, strong_concepts).

    concept_mastery_list: list of {concept, mastery_score, ...} dicts.
    Returns (weak_list, strong_list) — each is a list of concept name strings.
    """
    weak   = []
    strong = []
    for row in concept_mastery_list:
        score = row.get("mastery_score", 0.0)
        name  = row.get("concept", "")
        if score < _MASTERY_THRESHOLDS["weak"]:
            weak.append(name)
        elif score >= _MASTERY_THRESHOLDS["strong"]:
            strong.append(name)
    return weak, strong


# ── Persistence Helpers ────────────────────────────────────────────────────────

def save_quiz_result(user_id: str, doc_id: str,
                     quiz_questions: list, user_answers: dict,
                     score: int, total: int,
                     db_path: str | None = None) -> None:
    """
    Persist a completed quiz:
    1. Save quiz attempt (score + timestamp).
    2. Compute per-concept mastery from this quiz and upsert to DB.
    3. Recompute overall mastery across ALL concepts for this doc.
    4. Update learner profile (level + overall mastery).
    5. Log interaction event.
    """
    kwargs = {"db_path": db_path} if db_path else {}

    # 1. Record attempt
    insert_quiz_attempt(user_id, doc_id, score, total, **kwargs)

    # 2. Per-concept mastery from this quiz
    concept_scores = compute_concept_mastery(quiz_questions, user_answers)
    for concept, mastery_pct in concept_scores.items():
        upsert_concept_mastery(user_id, doc_id, concept, mastery_pct, **kwargs)

    # 3. Recompute overall mastery for this document
    all_concepts = get_concept_mastery(user_id, doc_id, **kwargs)
    if all_concepts:
        overall = round(
            sum(r["mastery_score"] for r in all_concepts) / len(all_concepts), 1
        )
    else:
        overall = round((score / total * 100) if total > 0 else 0.0, 1)

    # 4. Update learner profile
    new_level = estimate_level_from_mastery(overall)
    current_profile = get_learner_profile(user_id, **kwargs) or {}
    preferred_modality = current_profile.get("preferred_modality", "Not enough data")
    update_learner_profile(user_id, new_level, overall, preferred_modality, **kwargs)

    # 5. Log interaction (non-fatal)
    log_interaction(user_id, "quiz_submitted", doc_id, **kwargs)


def get_doc_concept_summary(user_id: str, doc_id: str,
                             db_path: str | None = None) -> dict:
    """
    Return a summary dict for a specific document:
        {
            "concepts": [...mastery rows...],
            "weak_concepts": [...names...],
            "strong_concepts": [...names...],
            "overall_mastery": float,
        }
    """
    kwargs = {"db_path": db_path} if db_path else {}
    concepts = get_concept_mastery(user_id, doc_id, **kwargs)
    weak, strong = _classify_concepts(concepts)
    overall = (
        round(sum(r["mastery_score"] for r in concepts) / len(concepts), 1)
        if concepts else 0.0
    )
    return {
        "concepts":        concepts,
        "weak_concepts":   weak,
        "strong_concepts": strong,
        "overall_mastery": overall,
    }


# ── Adaptive Prompt Context ────────────────────────────────────────────────────

def build_learner_context(user_id: str, doc_id: str,
                          db_path: str | None = None) -> str:
    """
    Build a concise learner-profile context string to inject into LLM prompts.

    Grounded in the user's actual stored profile — never fabricated.
    Deliberately kept compact to avoid inflating prompt size.

    Returns a multi-line string suitable for insertion into a prompt template.
    """
    kwargs = {"db_path": db_path} if db_path else {}

    profile  = load_profile(user_id, **kwargs)
    doc_summary = get_doc_concept_summary(user_id, doc_id, **kwargs)
    recent_score = get_recent_quiz_score(user_id, **kwargs)

    weak_str   = ", ".join(doc_summary["weak_concepts"][:5])   or "None identified yet"
    strong_str = ", ".join(doc_summary["strong_concepts"][:5]) or "None identified yet"
    recent_str = f"{recent_score:.0f}%" if recent_score is not None else "No quiz taken yet"
    mastery_str = f"{doc_summary['overall_mastery']:.0f}%" if doc_summary["overall_mastery"] else "0%"

    return (
        f"LEARNER PROFILE:\n"
        f"  Estimated Level:          {profile.get('estimated_level', 'Intermediate')}\n"
        f"  Overall Mastery (this chapter): {mastery_str}\n"
        f"  Weak Concepts:            {weak_str}\n"
        f"  Strong Concepts:          {strong_str}\n"
        f"  Preferred Learning Format:{profile.get('preferred_modality', 'Not enough data')}\n"
        f"  Recent Quiz Performance:  {recent_str}\n"
    )


# ── Dashboard Data ─────────────────────────────────────────────────────────────

def get_dashboard_data(user_id: str, doc_id: str | None = None,
                       db_path: str | None = None) -> dict:
    """
    Return all data needed to render the learner dashboard.

    Returns:
        {
            "profile":        {...},
            "doc_summary":    {...},   # only if doc_id provided
            "quiz_history":   [...],   # only if doc_id provided
            "recent_score":   float | None,
            "all_doc_mastery": [...],
        }
    """
    kwargs = {"db_path": db_path} if db_path else {}
    profile      = load_profile(user_id, **kwargs)
    recent_score = get_recent_quiz_score(user_id, **kwargs)
    all_docs     = get_all_doc_mastery_for_user(user_id, **kwargs)

    data = {
        "profile":         profile,
        "recent_score":    recent_score,
        "all_doc_mastery": all_docs,
    }

    if doc_id:
        data["doc_summary"]  = get_doc_concept_summary(user_id, doc_id, **kwargs)
        data["quiz_history"] = get_quiz_history(user_id, doc_id, **kwargs)

    return data
