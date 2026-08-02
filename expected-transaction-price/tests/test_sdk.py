"""Tests for SDK components."""

import pytest


class TestProjectModel:
    """Tests for ProjectModel."""

    def test_model_import(self):
        """Test model can be imported."""
        from expected_transaction_price.sdk.model import ExpectedTransactionPriceModel
        
        model = ExpectedTransactionPriceModel()
        assert model.name == "expected-transaction-price"


class TestProjectFeatures:
    """Tests for ProjectFeatures."""

    def test_features_import(self):
        """Test features can be imported."""
        from expected_transaction_price.sdk.features import ExpectedTransactionPriceFeatures
        
        features = ExpectedTransactionPriceFeatures()
        assert features is not None
