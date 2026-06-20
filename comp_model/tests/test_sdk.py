"""Tests for SDK components."""

import pytest


class TestProjectModel:
    """Tests for ProjectModel."""

    def test_model_import(self):
        """Test model can be imported."""
        from comp_model.sdk.model import CompModelModel
        
        model = CompModelModel()
        assert model.name == "comp_model"


class TestProjectFeatures:
    """Tests for ProjectFeatures."""

    def test_features_import(self):
        """Test features can be imported."""
        from comp_model.sdk.features import CompModelFeatures
        
        features = CompModelFeatures()
        assert features is not None
