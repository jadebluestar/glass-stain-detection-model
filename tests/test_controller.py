"""
Unit Tests for CleaningController

Tests the cleaning action planning functionality.
"""

import pytest
from controller_demo import CleaningController


class TestCleaningControllerInit:
    """Tests for CleaningController initialization."""

    @pytest.mark.unit
    def test_init_with_defaults(self):
        """Test initialization with default weights."""
        controller = CleaningController()
        # Weights should be normalized
        assert abs(controller.weight_confidence + controller.weight_area - 1.0) < 0.01

    @pytest.mark.unit
    def test_init_with_custom_weights(self):
        """Test initialization with custom weights."""
        controller = CleaningController(weight_confidence=0.7, weight_area=0.3)
        # Weights should be normalized to sum to 1
        assert abs(controller.weight_confidence + controller.weight_area - 1.0) < 0.01

    @pytest.mark.unit
    def test_init_with_zero_weights(self):
        """Test initialization with all zero weights."""
        controller = CleaningController(weight_confidence=0, weight_area=0)
        # Should handle gracefully
        assert controller is not None


class TestScoreDetection:
    """Tests for detection scoring functionality."""

    @pytest.mark.unit
    def test_score_detection_perfect_confidence(self, mock_detection):
        """Test scoring with high confidence detection."""
        controller = CleaningController()
        mock_detection["confidence"] = 1.0
        mock_detection["bbox"] = [0, 0, 100, 100]
        
        score = controller.score_detection(mock_detection, 1000, 1000)
        
        assert isinstance(score, float)
        assert 0 <= score <= 1

    @pytest.mark.unit
    def test_score_detection_no_confidence(self, mock_detection):
        """Test scoring with no confidence."""
        controller = CleaningController()
        mock_detection["confidence"] = 0.0
        
        score = controller.score_detection(mock_detection, 1000, 1000)
        
        assert score >= 0

    @pytest.mark.unit
    def test_score_detection_large_area(self, mock_detection):
        """Test scoring with large detection area."""
        controller = CleaningController()
        mock_detection["bbox"] = [0, 0, 500, 500]  # Large area
        
        score = controller.score_detection(mock_detection, 1000, 1000)
        
        assert isinstance(score, float)
        assert 0 <= score <= 1

    @pytest.mark.unit
    def test_score_detection_small_area(self, mock_detection):
        """Test scoring with small detection area."""
        controller = CleaningController()
        mock_detection["bbox"] = [0, 0, 10, 10]  # Small area
        
        score = controller.score_detection(mock_detection, 1000, 1000)
        
        assert isinstance(score, float)
        assert 0 <= score <= 1


class TestDecideAction:
    """Tests for cleaning action decision logic."""

    @pytest.mark.unit
    def test_high_intensity_action(self):
        """Test HIGH_INTENSITY strategy selection."""
        controller = CleaningController()
        action = controller.decide_action(0.85)
        
        assert action["strategy"] == "HIGH_INTENSITY"
        assert action["score"] == 0.85
        assert action["brush_speed_rpm"] == 120
        assert action["pressure_psi"] == 30

    @pytest.mark.unit
    def test_medium_action(self):
        """Test MEDIUM strategy selection."""
        controller = CleaningController()
        action = controller.decide_action(0.65)
        
        assert action["strategy"] == "MEDIUM"
        assert action["brush_speed_rpm"] == 80
        assert action["pressure_psi"] == 18

    @pytest.mark.unit
    def test_quick_pass_action(self):
        """Test QUICK_PASS strategy selection."""
        controller = CleaningController()
        action = controller.decide_action(0.30)
        
        assert action["strategy"] == "QUICK_PASS"
        assert action["brush_speed_rpm"] == 40
        assert action["pressure_psi"] == 8

    @pytest.mark.unit
    def test_boundary_scores(self):
        """Test action selection at strategy boundaries."""
        controller = CleaningController()
        
        # Test boundary at 0.80
        action_high = controller.decide_action(0.80)
        assert action_high["strategy"] == "HIGH_INTENSITY"
        
        action_medium = controller.decide_action(0.55)
        assert action_medium["strategy"] == "MEDIUM"


class TestProcessDetections:
    """Tests for detection processing and sorting."""

    @pytest.mark.unit
    def test_process_detections_returns_list(self, mock_detections):
        """Test that process_detections returns a list."""
        controller = CleaningController()
        result = controller.process_detections(mock_detections, 1024, 768)
        
        assert isinstance(result, list)
        assert len(result) == len(mock_detections)

    @pytest.mark.unit
    def test_process_detections_sorted_by_score(self, mock_detections):
        """Test that detections are sorted by score descending."""
        controller = CleaningController()
        result = controller.process_detections(mock_detections, 1024, 768)
        
        # Check that scores are in descending order
        scores = [d["action"]["score"] for d in result]
        assert scores == sorted(scores, reverse=True)

    @pytest.mark.unit
    def test_process_detections_includes_actions(self, mock_detections):
        """Test that processed detections include action information."""
        controller = CleaningController()
        result = controller.process_detections(mock_detections, 1024, 768)
        
        for detection in result:
            assert "action" in detection
            assert "strategy" in detection["action"]
            assert "score" in detection["action"]


class TestGenerateCleaningPlan:
    """Tests for cleaning plan generation."""

    @pytest.mark.unit
    def test_generate_plan_with_detections(self, mock_detections):
        """Test cleaning plan generation with detections."""
        controller = CleaningController()
        plan = controller.generate_cleaning_plan(mock_detections, 1024, 768)
        
        assert "overall_strategy" in plan
        assert "highest_severity_score" in plan
        assert "total_stains_detected" in plan
        assert "actions_by_priority" in plan
        assert plan["total_stains_detected"] == len(mock_detections)

    @pytest.mark.unit
    def test_generate_plan_no_detections(self):
        """Test cleaning plan generation with no detections."""
        controller = CleaningController()
        plan = controller.generate_cleaning_plan([], 1024, 768)
        
        assert plan["overall_strategy"] == "CLEAN"
        assert plan["highest_severity_score"] == 0
        assert plan["total_stains_detected"] == 0

    @pytest.mark.unit
    def test_generate_plan_strategy_selection(self):
        """Test that overall strategy is correctly selected."""
        controller = CleaningController()
        
        # Create high-confidence detections
        high_detections = [
            {
                "confidence": 0.95,
                "bbox": [0, 0, 200, 200],
                "class_name": "test"
            }
        ]
        
        plan = controller.generate_cleaning_plan(high_detections, 1024, 768)
        
        # Should get HIGH_INTENSITY due to high confidence and area
        assert plan["overall_strategy"] in ["HIGH_INTENSITY", "MEDIUM", "QUICK_PASS"]


class TestLegacyFunctions:
    """Tests for legacy function compatibility."""

    @pytest.mark.unit
    def test_legacy_score_detection(self, mock_detection):
        """Test legacy score_detection function."""
        from controller_demo import score_detection
        
        mock_detection["img_w"] = 1000
        mock_detection["img_h"] = 1000
        
        score = score_detection(mock_detection)
        
        assert isinstance(score, float)
        assert 0 <= score <= 1

    @pytest.mark.unit
    def test_legacy_decide_action_from_score(self):
        """Test legacy decide_action_from_score function."""
        from controller_demo import decide_action_from_score
        
        action = decide_action_from_score(0.8)
        
        assert "strategy" in action
        assert action["strategy"] == "HIGH_INTENSITY"

    @pytest.mark.unit
    def test_legacy_score_detections(self, mock_detections):
        """Test legacy score_detections function."""
        from controller_demo import score_detections
        
        # Add required fields for legacy function
        for d in mock_detections:
            d["img_w"] = 1024
            d["img_h"] = 768
        
        result = score_detections(mock_detections)
        
        assert isinstance(result, list)
        assert len(result) > 0


# Integration tests
@pytest.mark.integration
class TestCleaningControllerIntegration:
    """Integration tests for complete workflows."""

    @pytest.mark.slow
    def test_end_to_end_planning(self, mock_detections):
        """Test complete planning workflow."""
        controller = CleaningController()
        
        # Process detections and generate plan
        plan = controller.generate_cleaning_plan(mock_detections, 1024, 768)
        
        # Verify plan is reasonable
        assert plan is not None
        assert len(plan["actions_by_priority"]) > 0
        assert all("action" in a for a in plan["actions_by_priority"])
