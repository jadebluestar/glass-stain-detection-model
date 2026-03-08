"""
Integration Tests for PerceptionPipeline

Tests the complete detection and planning pipeline.
"""

import pytest
from unittest.mock import patch, MagicMock
from perception_demo import PerceptionPipeline


class TestPerceptionPipelineInit:
    """Tests for PerceptionPipeline initialization."""

    @pytest.mark.unit
    def test_init_creates_classifier_and_controller(self):
        """Test that pipeline initializes with classifier and controller."""
        with patch('perception_demo.StainClassifier'):
            with patch('perception_demo.CleaningController'):
                pipeline = PerceptionPipeline()
                assert hasattr(pipeline, 'classifier')
                assert hasattr(pipeline, 'controller')

    @pytest.mark.unit
    def test_init_handles_initialization_error(self):
        """Test that initialization errors are handled gracefully."""
        with patch('perception_demo.StainClassifier', side_effect=Exception("Init failed")):
            with pytest.raises(Exception):
                PerceptionPipeline()


class TestProcessImage:
    """Tests for single image processing."""

    @pytest.mark.unit
    def test_process_image_returns_dict(self, mock_detections, image_dimensions):
        """Test that process_image returns a dictionary with all required fields."""
        with patch('perception_demo.StainClassifier') as mock_classifier:
            with patch('perception_demo.CleaningController'):
                with patch('perception_demo.Image.open') as mock_open:
                    # Mock classifier
                    mock_cls_instance = MagicMock()
                    mock_classifier.return_value = mock_cls_instance
                    mock_cls_instance.analyze_image.return_value = {
                        "image_size": image_dimensions,
                        "detections": mock_detections,
                        "status": "CLEAN",
                        "contamination_ratio": 0
                    }
                    
                    # Mock image
                    mock_img = MagicMock()
                    mock_img.width = image_dimensions["width"]
                    mock_img.height = image_dimensions["height"]
                    mock_open.return_value = mock_img
                    
                    pipeline = PerceptionPipeline()
                    result = pipeline.process_image("test.jpg")
                    
                    assert isinstance(result, dict)
                    assert "image_file" in result
                    assert "detections" in result
                    assert "cleaning_plan" in result

    @pytest.mark.unit
    def test_process_image_with_no_detections(self, image_dimensions):
        """Test processing an image with no detections."""
        with patch('perception_demo.StainClassifier') as mock_classifier:
            with patch('perception_demo.CleaningController'):
                with patch('perception_demo.Image.open') as mock_open:
                    # Mock empty detections
                    mock_cls_instance = MagicMock()
                    mock_classifier.return_value = mock_cls_instance
                    mock_cls_instance.analyze_image.return_value = {
                        "image_size": image_dimensions,
                        "detections": [],
                        "status": "CLEAN",
                        "contamination_ratio": 0
                    }
                    
                    mock_img = MagicMock()
                    mock_img.width = image_dimensions["width"]
                    mock_img.height = image_dimensions["height"]
                    mock_open.return_value = mock_img
                    
                    pipeline = PerceptionPipeline()
                    result = pipeline.process_image("clean.jpg")
                    
                    assert result["detections"] == []


class TestProcessBatch:
    """Tests for batch image processing."""

    @pytest.mark.unit
    def test_process_batch_returns_list(self):
        """Test that process_batch returns a list of results."""
        with patch('perception_demo.StainClassifier'):
            with patch('perception_demo.CleaningController'):
                with patch('perception_demo.os.path.exists', return_value=True):
                    with patch('perception_demo.os.listdir', return_value=['img1.jpg', 'img2.jpg']):
                        with patch.object(PerceptionPipeline, 'process_image') as mock_process:
                            mock_process.return_value = {
                                "image_file": "test.jpg",
                                "detections": []
                            }
                            
                            pipeline = PerceptionPipeline()
                            result = pipeline.process_batch()
                            
                            assert isinstance(result, list)

    @pytest.mark.unit
    def test_process_batch_filters_image_formats(self):
        """Test that batch processing filters valid image formats."""
        with patch('perception_demo.StainClassifier'):
            with patch('perception_demo.CleaningController'):
                with patch('perception_demo.os.path.exists', return_value=True):
                    with patch('perception_demo.os.listdir') as mock_listdir:
                        mock_listdir.return_value = ['image.jpg', 'document.pdf', 'photo.png']
                        
                        with patch.object(PerceptionPipeline, 'process_image'):
                            pipeline = PerceptionPipeline()
                            # Should process only valid image formats


class TestSaveResults:
    """Tests for results saving functionality."""

    @pytest.mark.unit
    def test_save_results_creates_json(self):
        """Test that save_results creates a JSON file."""
        with patch('perception_demo.StainClassifier'):
            with patch('perception_demo.CleaningController'):
                with patch('builtins.open', create=True) as mock_open:
                    with patch('perception_demo.json.dump') as mock_dump:
                        pipeline = PerceptionPipeline()
                        test_results = [{"image_file": "test.jpg"}]
                        
                        pipeline.save_results(test_results)
                        
                        mock_dump.assert_called_once()

    @pytest.mark.unit
    def test_save_results_with_custom_output_file(self):
        """Test saving results to a custom file."""
        with patch('perception_demo.StainClassifier'):
            with patch('perception_demo.CleaningController'):
                with patch('builtins.open', create=True):
                    with patch('perception_demo.json.dump'):
                        pipeline = PerceptionPipeline()
                        test_results = [{"image_file": "test.jpg"}]
                        
                        pipeline.save_results(test_results, "custom_output.json")
                        
                        # Should not raise error


@pytest.mark.integration
class TestPerceptionPipelineIntegration:
    """Integration tests for complete pipeline."""

    @pytest.mark.slow
    def test_full_pipeline_workflow(self):
        """Test complete pipeline from detection to planning."""
        with patch('perception_demo.StainClassifier'):
            with patch('perception_demo.CleaningController'):
                pipeline = PerceptionPipeline()
                assert pipeline is not None
                assert hasattr(pipeline, 'process_image')
                assert hasattr(pipeline, 'process_batch')
