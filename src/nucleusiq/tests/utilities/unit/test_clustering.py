"""Tests for utilities/clustering.py."""

import pytest

from nucleusiq.utilities.clustering import cluster_questions


class TestClusterQuestions:

    def test_empty_list(self):
        assert cluster_questions([]) == {}

    def test_basic_clustering(self):
        questions = [
            "What is Python?",
            "How does Python work?",
            "What is Java?",
            "How does Java work?",
            "What is Rust?",
            "How does Rust work?",
            "What is Go?",
            "How does Go work?",
            "What is C++?",
            "How does C++ work?",
        ]
        clusters = cluster_questions(questions, num_clusters=3)
        assert isinstance(clusters, dict)
        total = sum(len(v) for v in clusters.values())
        assert total == 10

    def test_single_cluster(self):
        qs = ["Q1", "Q2", "Q3"]
        clusters = cluster_questions(qs, num_clusters=1)
        assert len(clusters) == 1
        assert len(list(clusters.values())[0]) == 3
