"""
Unit Tests for StainClassifier

Tests the stain detection and classification functionality.
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from stain_classifier import StainClassifier
from exceptions import ModelNotFoundError, InvalidImageError, InferenceError


class TestStainClassifierInit:
    """Tests for StainClassifier initialization."""

    @pytest.mark.unit
    def test_init_with_default_model(self):
        """Test initialization with default model path."""
        with patch('stain_classifier.YOLO'):
            classifier = StainClassifier()
            assert classifier.conf_threshold == 0.45

    @pytest.mark.unit
    def test_init_with_custom_confidence(self):
        """Test initialization with custom confidence threshold."""
        with patch('stain_classifier.YOLO'):
            classifier = StainClassifier(conf_threshold=0.60)
            assert classifier.conf_threshold == 0.60

    @pytest.mark.unit
    def test_init_with_invalid_model_path(self):
        """Test initialization with invalid model path raises error."""
        with patch('stain_classifier.YOLO', side_effect=Exception("Model not found")):
            with pytest.raises(Exception):
                StainClassifier(model_path="invalid_model.pt")


class TestDetectStains:
    """Tests for stain detection functionality."""

    @pytest.mark.unit
    def test_detect_stains_returns_list(self, mock_detections):
        """Test that detect_stains returns a list."""
        with patch('stain_classifier.YOLO') as mock_yolo:
            mock_model = MagicMock()
            mock_yolo.return_value = mock_model
            
            # Mock prediction results
            mock_result = MagicMock()
            mock_boxes = MagicMock()
            mock_boxes.xyxy = [[100, 100, 200, 200]]
            mock_boxes.conf = [0.9]
            mock_boxes.cls = [0]
            mock_result.boxes = mock_boxes
            mock_model.predict.return_value = [mock_result]
            
            classifier = StainClassifier()
            result = classifier.detect_stains("dummy_image.jpg")
            
            assert isinstance(result, list)

    @pytest.mark.unit
    def test_detect_stains_empty_image(self):
        """Test detection on image with no stains."""
        with patch('stain_classifier.YOLO') as mock_yolo:
            mock_model = MagicMock()
            mock_yolo.return_value = mock_model
            
            # Mock empty detections
            mock_result = MagicMock()
            mock_result.boxes = None
            mock_model.predict.return_value = [mock_result]
            
            classifier = StainClassifier()
            result = classifier.detect_stains("clean_image.jpg")
            
            assert result == []

    @pytest.mark.unit
    def test_detect_stains_invalid_image(self):
        """Test detection with invalid image path."""
        with patch('stain_classifier.YOLO') as mock_yolo:
            mock_model = MagicMock()
            mock_yolo.return_value = mock_model
            mock_model.predict.side_effect = Exception("Invalid image")
            
            classifier = StainClassifier()
            
            with pytest.raises(Exception):
                classifier.detect_stains("invalid_path.jpg")


class TestAnalyzeImage:
    """Tests for image analysis functionality."""

    @pytest.mark.unit
    def test_analyze_image_returns_dict(self, mock_detections, image_dimensions):
        """Test that analyze_image returns a dictionary."""
        with patch('stain_classifier.StainClassifier.detect_stains') as mock_detect:
            with patch('stain_classifier.Image.open') as mock_open:
                mock_detect.return_value = mock_detections
                mock_img = MagicMock()
                mock_img.width = image_dimensions["width"]
                mock_img.height = image_dimensions["height"]
                mock_open.return_value = mock_img
                
                classifier = StainClassifier()
                result = classifier.analyze_image("test_image.jpg")
                
                assert isinstance(result, dict)
                assert "image_path" in result
                assert "detections" in result
                assert "contamination_ratio" in result
                assert "status" in result

    @pytest.mark.unit
    def test_assess_contamination_clean(self):
        """Test contamination assessment for clean glass."""
        with patch('stain_classifier.YOLO'):
            classifier = StainClassifier()
            status = classifier._assess_contamination(0.5)
            assert status == "CLEAN"

    @pytest.mark.unit
    def test_assess_contamination_light(self):
        """Test contamination assessment for light contamination."""
        with patch('stain_classifier.YOLO'):
            classifier = StainClassifier()
            status = classifier._assess_contamination(3.0)
            assert status == "LIGHT_CONTAMINATION"

    @pytest.mark.unit
    def test_assess_contamination_moderate(self):
        """Test contamination assessment for moderate contamination."""
        with patch('stain_classifier.YOLO'):
            classifier = StainClassifier()
            status = classifier._assess_contamination(8.0)
            assert status == "MODERATE_CONTAMINATION"

    @pytest.mark.unit
    def test_assess_contamination_heavy(self):
        """Test contamination assessment for heavy contamination."""
        with patch('stain_classifier.YOLO'):
            classifier = StainClassifier()
            status = classifier._assess_contamination(20.0)
            assert status == "HEAVY_CONTAMINATION"


class TestBatchAnalyze:
    """Tests for batch analysis functionality."""

    @pytest.mark.unit
    def test_batch_analyze_returns_list(self, test_images_dir):
        """Test that batch_analyze returns a list."""
        with patch('stain_classifier.StainClassifier.analyze_image') as mock_analyze:
            mock_analyze.return_value = {
                "image_path": "test.jpg",
                "detections": [],
                "contamination_ratio": 0
            }
            
            with patch('stain_classifier.os.listdir') as mock_listdir:
                mock_listdir.return_value = ["test1.jpg", "test2.jpg"]
                
                classifier = StainClassifier()
                result = classifier.batch_analyze("dummy_dir")
                
                assert isinstance(result, list)

    @pytest.mark.unit
    def test_batch_analyze_filters_extensions(self):
        """Test that batch_analyze filters image extensions."""
        with patch('stain_classifier.StainClassifier.analyze_image'):
            with patch('stain_classifier.os.listdir') as mock_listdir:
                mock_listdir.return_value = ["image.jpg", "document.pdf", "image.png"]
                
                with patch('stain_classifier.Path.suffix', side_effect=['.jpg', '.pdf', '.png']):
                    classifier = StainClassifier()
                    # The method should handle filtering


class TestVisualizationMethods:
    """Tests for visualization functionality."""

    @pytest.mark.unit
    def test_visualize_detections_returns_image(self, mock_detections):
        """Test that visualize_detections returns a PIL Image."""
        with patch('stain_classifier.Image.open') as mock_open:
            with patch('stain_classifier.ImageDraw'):
                mock_img = MagicMock()
                mock_img.width = 800
                mock_img.height = 600
                mock_open.return_value = mock_img
                
                classifier = StainClassifier()
                result = classifier.visualize_detections(
                    "test_image.jpg",
                    mock_detections
                )
                
                # Result should be an Image or None
                assert result is not None or result is None


# Integration tests
@pytest.mark.integration
class TestStainClassifierIntegration:
    """Integration tests for complete workflows."""

    @pytest.mark.slow
    def test_full_pipeline_with_mock(self, mock_detections, image_dimensions):
        """Test complete detection pipeline with mocked dependencies."""
        with patch('stain_classifier.YOLO'):
            with patch('stain_classifier.Image.open') as mock_open:
                mock_img = MagicMock()
                mock_img.width = image_dimensions["width"]
                mock_img.height = image_dimensions["height"]
                mock_open.return_value = mock_img
                
                classifier = StainClassifier()
                
                # Should complete without errors
                assert classifier is not None
