"""Unit tests for predictions module

This module contains unit tests for the predictions functionality,
ensuring that ROI calculations and cluster assignments work correctly.
"""
import unittest
from src.predictions import get_cluster_name, calculate_roi


class TestPredictions(unittest.TestCase):
    def test_cluster_name(self):
        self.assertIn("Tech-Savvy", get_cluster_name(0))

    def test_roi_calculation(self):
        roi = calculate_roi(0.8, 2000)
        self.assertGreater(roi, 0)


if __name__ == '__main__':
    unittest.main()
