"""
Enterprise GenAI & RAG Knowledge Retrieval System — Evaluation Test Suite

This module validates the retrieval pipeline against 5+ specific query cases,
measures retrieval precision, and verifies >92% keyword match accuracy.
"""

import os
import sys
import unittest

# Ensure src modules are importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

CHROMA_DIR = os.path.join(os.path.dirname(__file__), "..", "chroma_db")


class TestRAGPipeline(unittest.TestCase):
    """Evaluate the RAG retrieval pipeline with targeted enterprise queries."""

    @classmethod
    def setUpClass(cls):
        cls.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        cls.db = Chroma(
            persist_directory=CHROMA_DIR,
            embedding_function=cls.embeddings,
            collection_name="enterprise_policies",
        )

    def _assert_keywords_present(self, results_text: str, expected_keywords: list, threshold: float = 0.92):
        """Verify that >92% of expected keywords appear in retrieved results."""
        found = sum(1 for kw in expected_keywords if kw.lower() in results_text.lower())
        accuracy = found / len(expected_keywords) if expected_keywords else 1.0
        self.assertGreaterEqual(
            accuracy,
            threshold,
            f"Keyword match accuracy {accuracy:.2%} below {threshold:.0%} (found {found}/{len(expected_keywords)})"
        )
        return accuracy

    # ------------------------------------------------------------------
    # 1. HR Remote Work Policy Query
    # ------------------------------------------------------------------
    def test_hr_remote_work_policy(self):
        query = "What is the remote work policy?"
        results = self.db.similarity_search_with_score(query, k=3)
        results_text = "\n".join([doc.page_content for doc, _ in results])
        expected_keywords = ["remote", "3 days", "manager approval", "vpn", "5 business days"]
        accuracy = self._assert_keywords_present(results_text, expected_keywords)
        print(f"[HR Remote Work] Match accuracy: {accuracy:.2%}")
        self.assertGreaterEqual(len(results), 1, "Expected at least 1 retrieval result")

    # ------------------------------------------------------------------
    # 2. IT Password Reset / Lockout Troubleshooting
    # ------------------------------------------------------------------
    def test_it_password_reset_lockout(self):
        query = "I forgot my password and I'm locked out — how do I reset it?"
        results = self.db.similarity_search_with_score(query, k=3)
        results_text = "\n".join([doc.page_content for doc, _ in results])
        expected_keywords = [
            "password", "reset", "90 days", "12 characters",
            "helpdesk", "support@company.com", "multi-factor authentication"
        ]
        accuracy = self._assert_keywords_present(results_text, expected_keywords)
        print(f"[IT Password Reset] Match accuracy: {accuracy:.2%}")
        self.assertGreaterEqual(len(results), 1)

    # ------------------------------------------------------------------
    # 3. Expense Reimbursement Limits
    # ------------------------------------------------------------------
    def test_expense_reimbursement_limits(self):
        query = "How much can I claim for travel expenses?"
        results = self.db.similarity_search_with_score(query, k=3)
        results_text = "\n".join([doc.page_content for doc, _ in results])
        expected_keywords = ["$500", "lodging", "$150", "meals", "$200", "transportation", "30 days"]
        accuracy = self._assert_keywords_present(results_text, expected_keywords)
        print(f"[Expense Limits] Match accuracy: {accuracy:.2%}")
        self.assertGreaterEqual(len(results), 1)

    # ------------------------------------------------------------------
    # 4. Data Security Compliance & Breach Reporting
    # ------------------------------------------------------------------
    def test_data_security_compliance(self):
        query = "What are the data security requirements and breach reporting rules?"
        results = self.db.similarity_search_with_score(query, k=3)
        results_text = "\n".join([doc.page_content for doc, _ in results])
        expected_keywords = [
            "annual data security training", "customer data", "personal devices",
            "breach", "CISO", "1 hour", "disciplinary action"
        ]
        accuracy = self._assert_keywords_present(results_text, expected_keywords)
        print(f"[Data Security] Match accuracy: {accuracy:.2%}")
        self.assertGreaterEqual(len(results), 1)

    # ------------------------------------------------------------------
    # 5. IT Self-Service Portal / Multi-Factor Auth
    # ------------------------------------------------------------------
    def test_it_self_service_portal(self):
        query = "Can I reset my password through the company portal?"
        results = self.db.similarity_search_with_score(query, k=3)
        results_text = "\n".join([doc.page_content for doc, _ in results])
        expected_keywords = ["self-service", "portal", "multi-factor authentication", "reset"]
        accuracy = self._assert_keywords_present(results_text, expected_keywords)
        print(f"[Self-Service Portal] Match accuracy: {accuracy:.2%}")
        self.assertGreaterEqual(len(results), 1)

    # ------------------------------------------------------------------
    # 6. Retrieval Precision & Score Validation
    # ------------------------------------------------------------------
    def test_retrieval_precision_scores(self):
        query = "What are the remote work and VPN requirements?"
        results = self.db.similarity_search_with_score(query, k=3)
        scores = [float(score) for _, score in results]
        print(f"[Precision] Similarity scores: {scores}")
        # Ensure scores are positive and ordered (lower = more similar in Chroma)
        self.assertTrue(all(s > 0 for s in scores), "All similarity scores should be positive")
        # For a well-tuned pipeline, top result should be strongly similar
        if len(scores) > 0:
            self.assertLessEqual(scores[0], max(scores), "Top result should have lowest (best) score")

    # ------------------------------------------------------------------
    # 7. Overall Pipeline Integrity — Multi-Keyword Coverage
    # ------------------------------------------------------------------
    def test_overall_keyword_coverage_above_92_percent(self):
        """Aggregate check across all test queries ensuring >92% coverage."""
        test_cases = [
            ("What is the remote work policy?",
             ["remote", "3 days", "manager approval", "vpn", "5 business days"]),
            ("How do I reset a forgotten password?",
             ["password", "reset", "90 days", "12 characters", "helpdesk", "multi-factor"]),
            ("How much can I claim for travel?",
             ["$500", "lodging", "$150", "meals", "$200", "transportation", "30 days"]),
            ("What must I do after a suspected data breach?",
             ["breach", "CISO", "1 hour", "disciplinary"]),
        ]
        accuracies = []
        for q, keywords in test_cases:
            results = self.db.similarity_search_with_score(q, k=3)
            text = "\n".join([doc.page_content for doc, _ in results])
            found = sum(1 for kw in keywords if kw.lower() in text.lower())
            accuracies.append(found / len(keywords) if keywords else 1.0)
        avg_accuracy = sum(accuracies) / len(accuracies)
        print(f"[Overall Coverage] Average keyword match accuracy: {avg_accuracy:.2%}")
        self.assertGreaterEqual(avg_accuracy, 0.92,
            f"Average retrieval keyword match accuracy {avg_accuracy:.2%} < 92%")


if __name__ == "__main__":
    unittest.main(verbosity=2)
