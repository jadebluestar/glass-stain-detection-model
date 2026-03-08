"""
Test Configuration and Fixtures

Provides pytest configuration and shared fixtures for all tests.
"""

import os
import sys
from pathlib import Path

import pytest

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def test_images_dir():
    """Fixture providing path to test images directory."""
    return PROJECT_ROOT / "data" / "test_images"


@pytest.fixture
def mock_detection():
    """Fixture providing a mock detection object."""
    return {
        "class_id": 0,
        "class_name": "rain_streaks",
        "confidence": 0.85,
        "bbox": [100, 100, 200, 200],
        "area": 10000
    }


@pytest.fixture
def mock_detections():
    """Fixture providing multiple mock detections."""
    return [
        {
            "class_id": 0,
            "class_name": "rain_streaks",
            "confidence": 0.92,
            "bbox": [320, 220, 380, 280],
            "area": 3600
        },
        {
            "class_id": 1,
            "class_name": "bird_droppings",
            "confidence": 0.77,
            "bbox": [700, 140, 760, 200],
            "area": 3600
        },
        {
            "class_id": 2,
            "class_name": "dust_spots",
            "confidence": 0.58,
            "bbox": [450, 430, 520, 500],
            "area": 4900
        }
    ]


@pytest.fixture
def image_dimensions():
    """Fixture providing standard image dimensions."""
    return {"width": 1024, "height": 768}


# Pytest configuration
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow"
    )
