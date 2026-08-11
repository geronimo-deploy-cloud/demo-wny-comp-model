"""Tests for SDK components."""

import pytest


class TestProjectModel:
    """Tests for ProjectModel."""

    def test_model_import(self):
        """Test model can be imported."""
        from comp_finder.sdk.model import CompFinderModel
        
        model = CompFinderModel()
        assert model.name == "comp-finder"


class TestProjectFeatures:
    """Tests for ProjectFeatures."""

    def test_features_import(self):
        """Test features can be imported."""
        from comp_finder.sdk.features import CompFinderFeatures
        
        features = CompFinderFeatures()
        assert features is not None
