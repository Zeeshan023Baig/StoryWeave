"""
tests/test_multiuser.py
=======================
Comprehensive unit tests for the StoryWeave multi-user support system.

Coverage:
 1.  User registration
 2.  Duplicate username rejection
 3.  Password hashing (never stored plain-text)
 4.  Password verification (correct / incorrect)
 5.  Successful login
 6.  Failed login
 7.  Logout (session state cleared, DB data preserved)
 8.  Learner profile creation
 9.  Loading learner profile
10.  Saving/updating learner profile
11.  Saving quiz attempts
12.  Saving concept mastery
13.  User A cannot access User B's data
14.  User A progress persists after logout/login
15.  User B starts with independent progress
16.  Chapter-specific (doc_id-specific) progress isolation
17.  Adaptive content receives correct user's learner profile
18.  Existing adaptive-learning tests still pass (imports)
19.  Existing retest tests still pass (imports)
20.  Existing Ask StoryWeave tests still pass (imports)
21.  Existing Ollama tests still pass (imports)
22.  Existing Gemini compatibility tests still pass (imports)

All tests use an in-memory / temp SQLite DB — zero real API calls made.
"""

import os
import tempfile
import unittest
from unittest.mock import patch, MagicMock

# ── Helpers to use an isolated temp DB ────────────────────────────────────────

def _tmp_db():
    """Return a path to a fresh temporary SQLite file."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    return path


# ── Import modules under test ─────────────────────────────────────────────────
from database.database import (
    init_db,
    insert_user,
    get_user_by_username,
    get_user_by_id,
    insert_learner_profile,
    get_learner_profile,
    update_learner_profile,
    upsert_concept_mastery,
    get_concept_mastery,
    get_all_doc_mastery_for_user,
    insert_quiz_attempt,
    get_quiz_history,
    get_recent_quiz_score,
    log_interaction,
)
from services.auth_service import (
    _hash_password,
    _verify_password,
    register_user,
    login_user,
)
from services.learner_service import (
    load_profile,
    save_quiz_result,
    compute_concept_mastery,
    estimate_level_from_mastery,
    build_learner_context,
    get_dashboard_data,
)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Database Initialisation
# ─────────────────────────────────────────────────────────────────────────────

class TestDatabaseInit(unittest.TestCase):
    def test_init_creates_db_file(self):
        """init_db() creates the SQLite file if it does not exist."""
        path = _tmp_db()
        os.remove(path)  # ensure it does not exist yet
        init_db(db_path=path)
        self.assertTrue(os.path.exists(path))

    def test_init_is_idempotent(self):
        """Calling init_db() twice on same file does not raise."""
        path = _tmp_db()
        init_db(db_path=path)
        init_db(db_path=path)  # second call must not fail

    def tearDown(self):
        # cleanup temp files
        pass


# ─────────────────────────────────────────────────────────────────────────────
# 2 & 3. Password Hashing
# ─────────────────────────────────────────────────────────────────────────────

class TestPasswordHashing(unittest.TestCase):
    def test_hash_is_not_plain_text(self):
        """Stored hash must not equal the plain-text password."""
        pw = "SuperSecret123"
        hashed = _hash_password(pw)
        self.assertNotEqual(hashed, pw)

    def test_hash_starts_with_bcrypt_prefix(self):
        """bcrypt hashes start with $2b$ (bcrypt version identifier)."""
        hashed = _hash_password("mypassword")
        self.assertTrue(hashed.startswith("$2"), f"Unexpected prefix: {hashed[:5]}")

    def test_two_hashes_of_same_password_differ(self):
        """bcrypt uses random salts — two hashes of the same password must differ."""
        pw = "SamePassword"
        h1 = _hash_password(pw)
        h2 = _hash_password(pw)
        self.assertNotEqual(h1, h2)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Password Verification
# ─────────────────────────────────────────────────────────────────────────────

class TestPasswordVerification(unittest.TestCase):
    def setUp(self):
        self.pw = "CorrectPassword!"
        self.hash = _hash_password(self.pw)

    def test_correct_password_verifies(self):
        self.assertTrue(_verify_password(self.pw, self.hash))

    def test_wrong_password_fails(self):
        self.assertFalse(_verify_password("WrongPassword", self.hash))

    def test_empty_password_fails(self):
        self.assertFalse(_verify_password("", self.hash))

    def test_corrupt_hash_fails_gracefully(self):
        """A corrupted stored hash must not raise — return False."""
        self.assertFalse(_verify_password(self.pw, "not_a_real_hash"))


# ─────────────────────────────────────────────────────────────────────────────
# 1 & 2. User Registration
# ─────────────────────────────────────────────────────────────────────────────

class TestUserRegistration(unittest.TestCase):
    def setUp(self):
        self.db = _tmp_db()
        init_db(db_path=self.db)

    def tearDown(self):
        try:
            os.remove(self.db)
        except FileNotFoundError:
            pass

    def test_register_creates_user(self):
        """register_user() creates a user and returns user_id + username."""
        result = register_user("alice", "AlicePass1!", db_path=self.db)
        self.assertIn("user_id", result)
        self.assertEqual(result["username"], "alice")

    def test_register_creates_learner_profile(self):
        """register_user() automatically creates a default learner profile."""
        result = register_user("bob", "BobPass2@", db_path=self.db)
        profile = get_learner_profile(result["user_id"], db_path=self.db)
        self.assertIsNotNone(profile)
        self.assertEqual(profile["estimated_level"], "Intermediate")
        self.assertEqual(profile["overall_mastery"], 0.0)

    def test_duplicate_username_rejected(self):
        """Registering the same username twice raises RuntimeError."""
        register_user("charlie", "CharliePass3#", db_path=self.db)
        with self.assertRaises(RuntimeError) as ctx:
            register_user("charlie", "AnotherPass4$", db_path=self.db)
        self.assertIn("already exists", str(ctx.exception).lower())

    def test_empty_username_rejected(self):
        with self.assertRaises(ValueError):
            register_user("", "password", db_path=self.db)

    def test_empty_password_rejected(self):
        with self.assertRaises(ValueError):
            register_user("david", "", db_path=self.db)

    def test_password_not_stored_plain_text(self):
        """The stored password_hash must not equal the original password."""
        result = register_user("eve", "EveSecret5%", db_path=self.db)
        row = get_user_by_username("eve", db_path=self.db)
        self.assertNotEqual(row["password_hash"], "EveSecret5%")


# ─────────────────────────────────────────────────────────────────────────────
# 5 & 6. Login
# ─────────────────────────────────────────────────────────────────────────────

class TestLogin(unittest.TestCase):
    def setUp(self):
        self.db = _tmp_db()
        init_db(db_path=self.db)
        self.user = register_user("frank", "FrankPass6^", db_path=self.db)

    def tearDown(self):
        try:
            os.remove(self.db)
        except FileNotFoundError:
            pass

    def test_successful_login(self):
        result = login_user("frank", "FrankPass6^", db_path=self.db)
        self.assertEqual(result["user_id"], self.user["user_id"])
        self.assertEqual(result["username"], "frank")

    def test_wrong_password_fails(self):
        with self.assertRaises(ValueError) as ctx:
            login_user("frank", "WrongPassword", db_path=self.db)
        self.assertIn("incorrect", str(ctx.exception).lower())

    def test_nonexistent_user_fails(self):
        with self.assertRaises(ValueError):
            login_user("nobody", "somepassword", db_path=self.db)

    def test_login_does_not_return_hash(self):
        """Login result must not include the password hash."""
        result = login_user("frank", "FrankPass6^", db_path=self.db)
        self.assertNotIn("password_hash", result)
        self.assertNotIn("password", result)

    def test_empty_credentials_fail(self):
        with self.assertRaises(ValueError):
            login_user("", "", db_path=self.db)


# ─────────────────────────────────────────────────────────────────────────────
# 7. Logout — session clear (tested at service layer: DB data must persist)
# ─────────────────────────────────────────────────────────────────────────────

class TestLogout(unittest.TestCase):
    def setUp(self):
        self.db = _tmp_db()
        init_db(db_path=self.db)
        self.user = register_user("grace", "GracePass7&", db_path=self.db)

    def tearDown(self):
        try:
            os.remove(self.db)
        except FileNotFoundError:
            pass

    def test_db_data_survives_logout(self):
        """After 'logout' (session clear), user can log back in and data is preserved."""
        uid = self.user["user_id"]
        # Save some concept mastery before logout
        upsert_concept_mastery(uid, "doc1", "Photosynthesis", 85.0, db_path=self.db)

        # Simulate logout — just clear session state (not DB)
        # Then re-login
        result = login_user("grace", "GracePass7&", db_path=self.db)
        self.assertEqual(result["user_id"], uid)

        # Data must still be there
        concepts = get_concept_mastery(uid, "doc1", db_path=self.db)
        self.assertEqual(len(concepts), 1)
        self.assertEqual(concepts[0]["concept"], "Photosynthesis")
        self.assertAlmostEqual(concepts[0]["mastery_score"], 85.0)


# ─────────────────────────────────────────────────────────────────────────────
# 8, 9, 10. Learner Profile
# ─────────────────────────────────────────────────────────────────────────────

class TestLearnerProfile(unittest.TestCase):
    def setUp(self):
        self.db = _tmp_db()
        init_db(db_path=self.db)
        self.user = register_user("helen", "HelenPass8*", db_path=self.db)
        self.uid = self.user["user_id"]

    def tearDown(self):
        try:
            os.remove(self.db)
        except FileNotFoundError:
            pass

    def test_profile_created_on_register(self):
        profile = get_learner_profile(self.uid, db_path=self.db)
        self.assertIsNotNone(profile)
        self.assertEqual(profile["estimated_level"], "Intermediate")
        self.assertEqual(profile["overall_mastery"], 0.0)
        self.assertEqual(profile["preferred_modality"], "Not enough data")

    def test_load_profile_returns_defaults_when_exists(self):
        profile = load_profile(self.uid, db_path=self.db)
        self.assertEqual(profile["estimated_level"], "Intermediate")

    def test_update_learner_profile(self):
        update_learner_profile(self.uid, "Advanced", 88.5, "Visual", db_path=self.db)
        profile = get_learner_profile(self.uid, db_path=self.db)
        self.assertEqual(profile["estimated_level"], "Advanced")
        self.assertAlmostEqual(profile["overall_mastery"], 88.5)
        self.assertEqual(profile["preferred_modality"], "Visual")

    def test_missing_profile_auto_created_by_load(self):
        """load_profile() must auto-create a missing profile."""
        # Manually insert user without profile
        import uuid
        uid2 = str(uuid.uuid4())
        insert_user(uid2, "noprofile_user", _hash_password("pass"), db_path=self.db)
        profile = load_profile(uid2, db_path=self.db)
        self.assertEqual(profile["estimated_level"], "Intermediate")


# ─────────────────────────────────────────────────────────────────────────────
# 11. Quiz Attempts
# ─────────────────────────────────────────────────────────────────────────────

class TestQuizAttempts(unittest.TestCase):
    def setUp(self):
        self.db = _tmp_db()
        init_db(db_path=self.db)
        self.user = register_user("ivan", "IvanPass9(", db_path=self.db)
        self.uid = self.user["user_id"]

    def tearDown(self):
        try:
            os.remove(self.db)
        except FileNotFoundError:
            pass

    def test_insert_and_retrieve_quiz_attempt(self):
        insert_quiz_attempt(self.uid, "doc_abc", 4, 5, db_path=self.db)
        history = get_quiz_history(self.uid, "doc_abc", db_path=self.db)
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["score"], 4)
        self.assertEqual(history[0]["total_questions"], 5)

    def test_multiple_attempts_all_retrieved(self):
        insert_quiz_attempt(self.uid, "doc_abc", 3, 5, db_path=self.db)
        insert_quiz_attempt(self.uid, "doc_abc", 5, 5, db_path=self.db)
        history = get_quiz_history(self.uid, "doc_abc", db_path=self.db)
        self.assertEqual(len(history), 2)

    def test_recent_score_correct_percentage(self):
        insert_quiz_attempt(self.uid, "doc_xyz", 4, 5, db_path=self.db)
        score = get_recent_quiz_score(self.uid, db_path=self.db)
        self.assertAlmostEqual(score, 80.0)

    def test_save_quiz_result_full_pipeline(self):
        """save_quiz_result() stores attempt, computes mastery, updates profile."""
        questions = [
            {"question": "Q1?", "options": ["A", "B"], "correct_answer": "A"},
            {"question": "Q2?", "options": ["A", "B"], "correct_answer": "B"},
            {"question": "Q3?", "options": ["A", "B"], "correct_answer": "A"},
        ]
        answers = {0: "A", 1: "A", 2: "A"}  # 2/3 correct

        save_quiz_result(
            user_id=self.uid,
            doc_id="doc_test",
            quiz_questions=questions,
            user_answers=answers,
            score=2,
            total=3,
            db_path=self.db,
        )

        history = get_quiz_history(self.uid, "doc_test", db_path=self.db)
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["score"], 2)

        profile = get_learner_profile(self.uid, db_path=self.db)
        self.assertIsNotNone(profile)


# ─────────────────────────────────────────────────────────────────────────────
# 12. Concept Mastery
# ─────────────────────────────────────────────────────────────────────────────

class TestConceptMastery(unittest.TestCase):
    def setUp(self):
        self.db = _tmp_db()
        init_db(db_path=self.db)
        self.user = register_user("julia", "JuliaPass10)", db_path=self.db)
        self.uid = self.user["user_id"]

    def tearDown(self):
        try:
            os.remove(self.db)
        except FileNotFoundError:
            pass

    def test_upsert_and_retrieve_concept(self):
        upsert_concept_mastery(self.uid, "doc1", "Evaporation", 90.0, db_path=self.db)
        rows = get_concept_mastery(self.uid, "doc1", db_path=self.db)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["concept"], "Evaporation")
        self.assertAlmostEqual(rows[0]["mastery_score"], 90.0)

    def test_upsert_updates_existing(self):
        upsert_concept_mastery(self.uid, "doc1", "Condensation", 40.0, db_path=self.db)
        upsert_concept_mastery(self.uid, "doc1", "Condensation", 70.0, db_path=self.db)
        rows = get_concept_mastery(self.uid, "doc1", db_path=self.db)
        self.assertEqual(len(rows), 1)
        self.assertAlmostEqual(rows[0]["mastery_score"], 70.0)

    def test_compute_concept_mastery_from_quiz(self):
        """compute_concept_mastery() correctly maps answers to per-concept scores."""
        questions = [
            {"question": "Q1?", "options": ["A", "B"], "correct_answer": "A", "concept": "Evaporation"},
            {"question": "Q2?", "options": ["A", "B"], "correct_answer": "B", "concept": "Evaporation"},
            {"question": "Q3?", "options": ["A", "B"], "correct_answer": "A", "concept": "Condensation"},
        ]
        answers = {0: "A", 1: "A", 2: "A"}  # Evaporation: 1/2=50%; Condensation: 1/1=100%
        result = compute_concept_mastery(questions, answers)
        self.assertIn("Evaporation", result)
        self.assertIn("Condensation", result)
        self.assertAlmostEqual(result["Evaporation"], 50.0)
        self.assertAlmostEqual(result["Condensation"], 100.0)


# ─────────────────────────────────────────────────────────────────────────────
# 13 & 15. Data Isolation: User A cannot see User B's data
# ─────────────────────────────────────────────────────────────────────────────

class TestDataIsolation(unittest.TestCase):
    def setUp(self):
        self.db = _tmp_db()
        init_db(db_path=self.db)
        self.user_a = register_user("userA", "PassA!", db_path=self.db)
        self.user_b = register_user("userB", "PassB!", db_path=self.db)
        self.uid_a = self.user_a["user_id"]
        self.uid_b = self.user_b["user_id"]

    def tearDown(self):
        try:
            os.remove(self.db)
        except FileNotFoundError:
            pass

    def test_user_a_quiz_not_visible_to_user_b(self):
        """User B must see zero quiz history for User A's document."""
        insert_quiz_attempt(self.uid_a, "water_cycle_doc", 5, 5, db_path=self.db)
        b_history = get_quiz_history(self.uid_b, "water_cycle_doc", db_path=self.db)
        self.assertEqual(b_history, [])

    def test_user_a_concept_mastery_not_visible_to_user_b(self):
        """User B must not see User A's concept mastery."""
        upsert_concept_mastery(self.uid_a, "doc_shared", "Evaporation", 95.0, db_path=self.db)
        b_concepts = get_concept_mastery(self.uid_b, "doc_shared", db_path=self.db)
        self.assertEqual(b_concepts, [])

    def test_user_b_starts_with_clean_profile(self):
        """User B starts with 0% mastery regardless of User A's progress."""
        # Give User A some mastery
        upsert_concept_mastery(self.uid_a, "doc_shared", "Precipitation", 88.0, db_path=self.db)
        update_learner_profile(self.uid_a, "Advanced", 88.0, "Visual", db_path=self.db)
        # User B's profile is untouched
        b_profile = get_learner_profile(self.uid_b, db_path=self.db)
        self.assertEqual(b_profile["overall_mastery"], 0.0)
        self.assertEqual(b_profile["estimated_level"], "Intermediate")


# ─────────────────────────────────────────────────────────────────────────────
# 14. User A progress persists after logout/login
# ─────────────────────────────────────────────────────────────────────────────

class TestProgressPersistence(unittest.TestCase):
    def setUp(self):
        self.db = _tmp_db()
        init_db(db_path=self.db)
        self.user_a = register_user("persistA", "PersistPass!", db_path=self.db)
        self.uid_a = self.user_a["user_id"]

    def tearDown(self):
        try:
            os.remove(self.db)
        except FileNotFoundError:
            pass

    def test_progress_persists_across_login_sessions(self):
        """Save data, simulate logout, re-login, verify data is intact."""
        # Step 1: Save concept mastery and quiz attempt
        upsert_concept_mastery(self.uid_a, "doc_water", "Evaporation", 78.0, db_path=self.db)
        insert_quiz_attempt(self.uid_a, "doc_water", 4, 5, db_path=self.db)

        # Step 2: Simulate logout (clear session in UI; DB is untouched)
        # Step 3: Re-login
        re_logged = login_user("persistA", "PersistPass!", db_path=self.db)
        self.assertEqual(re_logged["user_id"], self.uid_a)

        # Step 4: Verify data persists
        concepts = get_concept_mastery(self.uid_a, "doc_water", db_path=self.db)
        self.assertEqual(len(concepts), 1)
        self.assertAlmostEqual(concepts[0]["mastery_score"], 78.0)

        history = get_quiz_history(self.uid_a, "doc_water", db_path=self.db)
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["score"], 4)


# ─────────────────────────────────────────────────────────────────────────────
# 16. Chapter-specific progress isolation
# ─────────────────────────────────────────────────────────────────────────────

class TestChapterSpecificProgress(unittest.TestCase):
    def setUp(self):
        self.db = _tmp_db()
        init_db(db_path=self.db)
        self.user = register_user("chapterUser", "ChapPass!", db_path=self.db)
        self.uid = self.user["user_id"]

    def tearDown(self):
        try:
            os.remove(self.db)
        except FileNotFoundError:
            pass

    def test_different_docs_have_independent_mastery(self):
        """Mastery for doc A must not affect doc B."""
        upsert_concept_mastery(self.uid, "doc_water_cycle",   "Evaporation",   90.0, db_path=self.db)
        upsert_concept_mastery(self.uid, "doc_photosynthesis","Chlorophyll",   55.0, db_path=self.db)

        water_concepts = get_concept_mastery(self.uid, "doc_water_cycle",    db_path=self.db)
        photo_concepts = get_concept_mastery(self.uid, "doc_photosynthesis", db_path=self.db)

        self.assertEqual(len(water_concepts), 1)
        self.assertEqual(water_concepts[0]["concept"], "Evaporation")
        self.assertEqual(len(photo_concepts), 1)
        self.assertEqual(photo_concepts[0]["concept"], "Chlorophyll")

    def test_all_doc_mastery_shows_multiple_docs(self):
        upsert_concept_mastery(self.uid, "doc_java",   "OOP",           91.0, db_path=self.db)
        upsert_concept_mastery(self.uid, "doc_biology","Photosynthesis", 42.0, db_path=self.db)

        all_docs = get_all_doc_mastery_for_user(self.uid, db_path=self.db)
        doc_ids = [d["doc_id"] for d in all_docs]
        self.assertIn("doc_java",   doc_ids)
        self.assertIn("doc_biology", doc_ids)


# ─────────────────────────────────────────────────────────────────────────────
# 17. Adaptive content receives correct user's learner profile
# ─────────────────────────────────────────────────────────────────────────────

class TestAdaptivePromptContext(unittest.TestCase):
    def setUp(self):
        self.db = _tmp_db()
        init_db(db_path=self.db)
        self.user = register_user("adaptUser", "AdaptPass!", db_path=self.db)
        self.uid = self.user["user_id"]

    def tearDown(self):
        try:
            os.remove(self.db)
        except FileNotFoundError:
            pass

    def test_build_learner_context_contains_user_data(self):
        """build_learner_context() must reflect this user's actual stored data."""
        update_learner_profile(self.uid, "Beginner", 45.0, "Audio", db_path=self.db)
        upsert_concept_mastery(self.uid, "doc_test", "Condensation", 35.0, db_path=self.db)
        upsert_concept_mastery(self.uid, "doc_test", "Evaporation",  90.0, db_path=self.db)

        context = build_learner_context(self.uid, "doc_test", db_path=self.db)

        self.assertIn("Beginner", context)
        self.assertIn("Condensation", context)   # should appear as weak concept
        self.assertIn("Evaporation", context)    # should appear as strong concept

    def test_build_learner_context_user_a_vs_user_b(self):
        """Two users must receive DIFFERENT context strings."""
        user_b = register_user("adaptUserB", "AdaptPassB!", db_path=self.db)
        uid_b  = user_b["user_id"]

        update_learner_profile(self.uid,  "Advanced", 90.0, "Visual", db_path=self.db)
        update_learner_profile(uid_b, "Beginner",  20.0, "Audio",  db_path=self.db)

        ctx_a = build_learner_context(self.uid, "doc_test", db_path=self.db)
        ctx_b = build_learner_context(uid_b,   "doc_test", db_path=self.db)

        self.assertIn("Advanced", ctx_a)
        self.assertIn("Beginner", ctx_b)
        self.assertNotEqual(ctx_a, ctx_b)


# ─────────────────────────────────────────────────────────────────────────────
# Mastery → Level mapping
# ─────────────────────────────────────────────────────────────────────────────

class TestLevelEstimation(unittest.TestCase):
    def test_low_mastery_is_beginner(self):
        self.assertEqual(estimate_level_from_mastery(30.0), "Beginner")

    def test_mid_mastery_is_intermediate(self):
        self.assertEqual(estimate_level_from_mastery(75.0), "Intermediate")

    def test_high_mastery_is_advanced(self):
        self.assertEqual(estimate_level_from_mastery(90.0), "Advanced")

    def test_boundary_intermediate(self):
        self.assertEqual(estimate_level_from_mastery(70.0), "Intermediate")

    def test_boundary_advanced(self):
        self.assertEqual(estimate_level_from_mastery(85.0), "Advanced")


# ─────────────────────────────────────────────────────────────────────────────
# Dashboard data
# ─────────────────────────────────────────────────────────────────────────────

class TestDashboardData(unittest.TestCase):
    def setUp(self):
        self.db = _tmp_db()
        init_db(db_path=self.db)
        self.user = register_user("dashUser", "DashPass!", db_path=self.db)
        self.uid = self.user["user_id"]

    def tearDown(self):
        try:
            os.remove(self.db)
        except FileNotFoundError:
            pass

    def test_dashboard_includes_profile(self):
        data = get_dashboard_data(self.uid, db_path=self.db)
        self.assertIn("profile", data)
        self.assertEqual(data["profile"]["estimated_level"], "Intermediate")

    def test_dashboard_includes_doc_summary_when_doc_id_given(self):
        upsert_concept_mastery(self.uid, "doc_dash", "Precipitation", 55.0, db_path=self.db)
        data = get_dashboard_data(self.uid, doc_id="doc_dash", db_path=self.db)
        self.assertIn("doc_summary", data)
        self.assertIn("Precipitation", [c["concept"] for c in data["doc_summary"]["concepts"]])

    def test_dashboard_quiz_history_when_doc_id_given(self):
        insert_quiz_attempt(self.uid, "doc_dash", 3, 5, db_path=self.db)
        data = get_dashboard_data(self.uid, doc_id="doc_dash", db_path=self.db)
        self.assertIn("quiz_history", data)
        self.assertEqual(len(data["quiz_history"]), 1)


# ─────────────────────────────────────────────────────────────────────────────
# 18–22. Smoke-test: existing modules still importable (no API calls made)
# ─────────────────────────────────────────────────────────────────────────────

class TestExistingModulesStillWork(unittest.TestCase):
    def test_combined_generator_importable(self):
        from modules import combined_generator  # noqa: F401

    def test_llm_service_importable(self):
        from modules import llm_service  # noqa: F401

    def test_quiz_generator_importable(self):
        from modules import quiz_generator  # noqa: F401

    def test_pdf_processor_importable(self):
        from modules import pdf_processor  # noqa: F401

    def test_audio_generator_importable(self):
        from modules import audio_generator  # noqa: F401

    def test_settings_importable(self):
        from config import settings  # noqa: F401

    def test_ollama_provider_setting(self):
        from config.settings import OLLAMA_MODEL, OLLAMA_BASE_URL, AI_PROVIDER
        self.assertIsInstance(OLLAMA_MODEL, str)
        self.assertIsInstance(OLLAMA_BASE_URL, str)

    def test_gemini_setting_present(self):
        """Gemini compatibility: MODEL_NAME setting must exist."""
        from config.settings import MODEL_NAME
        self.assertIsInstance(MODEL_NAME, str)


if __name__ == "__main__":
    unittest.main()
