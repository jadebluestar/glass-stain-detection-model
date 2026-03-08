"""
Tests for Custom Exception Classes

Verifies exception handling and hierarchy.
"""

import pytest
from exceptions import (
    GlassStainError,
    ModelNotFoundError,
    InvalidImageError,
    InferenceError,
    ConfigurationError,
    DatasetError,
    ValidationError
)


class TestExceptionHierarchy:
    """Tests for exception class hierarchy."""

    @pytest.mark.unit
    def test_all_exceptions_inherit_from_base(self):
        """Test that all custom exceptions inherit from GlassStainError."""
        assert issubclass(ModelNotFoundError, GlassStainError)
        assert issubclass(InvalidImageError, GlassStainError)
        assert issubclass(InferenceError, GlassStainError)
        assert issubclass(ConfigurationError, GlassStainError)
        assert issubclass(DatasetError, GlassStainError)
        assert issubclass(ValidationError, GlassStainError)

    @pytest.mark.unit
    def test_base_exception_inherits_from_exception(self):
        """Test that base exception inherits from Exception."""
        assert issubclass(GlassStainError, Exception)


class TestExceptionRaising:
    """Tests for raising and catching exceptions."""

    @pytest.mark.unit
    def test_raise_model_not_found(self):
        """Test raising ModelNotFoundError."""
        with pytest.raises(ModelNotFoundError):
            raise ModelNotFoundError("Model file not found")

    @pytest.mark.unit
    def test_raise_invalid_image(self):
        """Test raising InvalidImageError."""
        with pytest.raises(InvalidImageError):
            raise InvalidImageError("Image format not supported")

    @pytest.mark.unit
    def test_raise_inference_error(self):
        """Test raising InferenceError."""
        with pytest.raises(InferenceError):
            raise InferenceError("Model inference failed")

    @pytest.mark.unit
    def test_raise_configuration_error(self):
        """Test raising ConfigurationError."""
        with pytest.raises(ConfigurationError):
            raise ConfigurationError("Invalid configuration")

    @pytest.mark.unit
    def test_raise_dataset_error(self):
        """Test raising DatasetError."""
        with pytest.raises(DatasetError):
            raise DatasetError("Dataset validation failed")

    @pytest.mark.unit
    def test_raise_validation_error(self):
        """Test raising ValidationError."""
        with pytest.raises(ValidationError):
            raise ValidationError("Validation failed")


class TestExceptionCatching:
    """Tests for catching exceptions at different hierarchy levels."""

    @pytest.mark.unit
    def test_catch_specific_exception(self):
        """Test catching specific exception type."""
        try:
            raise ModelNotFoundError("Test")
        except ModelNotFoundError:
            pass  # Should catch successfully

    @pytest.mark.unit
    def test_catch_base_exception(self):
        """Test catching base GlassStainError catches all subclasses."""
        exceptions = [
            ModelNotFoundError("Test"),
            InvalidImageError("Test"),
            InferenceError("Test"),
            ConfigurationError("Test"),
            DatasetError("Test"),
            ValidationError("Test")
        ]
        
        for exc in exceptions:
            try:
                raise exc
            except GlassStainError:
                pass  # Should catch all subclasses


class TestExceptionMessages:
    """Tests for exception messages."""

    @pytest.mark.unit
    def test_exception_with_message(self):
        """Test that exceptions can include messages."""
        message = "Test error message"
        with pytest.raises(GlassStainError) as exc_info:
            raise GlassStainError(message)
        assert str(exc_info.value) == message

    @pytest.mark.unit
    def test_exception_without_message(self):
        """Test that exceptions can be raised without messages."""
        with pytest.raises(GlassStainError):
            raise GlassStainError()
