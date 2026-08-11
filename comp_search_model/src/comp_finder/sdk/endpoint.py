"""Endpoint definition - handle incoming prediction requests."""

from geronimo.serving import Endpoint
from .model import CompFinderModel


class CompFinderEndpoint(Endpoint):
    """Prediction endpoint for real-time serving.
    
    This is a working demo endpoint. Replace the preprocess/postprocess
    methods with your actual implementation once you have a trained model.
    
    To train a model:
        uv run python -m comp_finder.train
    """

    model_class = CompFinderModel

    def preprocess(self, request: dict):
        """Transform incoming request to model input.
        
        Args:
            request: JSON request body with "features" key
            
        Returns:
            Feature matrix ready for model.predict()
        """
        # Demo mode: just return the features dict
        # TODO: Replace with actual preprocessing once model is trained
        # import pandas as pd
        # df = pd.DataFrame([request["features"]])
        # return self.model.features.transform(df)
        return request.get("features", request)

    def postprocess(self, prediction):
        """Format model output for response.
        
        Args:
            prediction: Raw model output
            
        Returns:
            JSON-serializable response
        """
        # Demo mode: echo the input back
        # TODO: Replace with actual postprocessing once model is trained
        # return {"prediction": int(prediction[0]), "confidence": 0.95}
        return {"result": prediction, "status": "demo_mode"}
    
    def initialize(self, project=None, version=None):
        """Initialize endpoint.
        
        Demo mode: Skip model loading if artifacts don't exist.
        """
        try:
            super().initialize(project=project, version=version)
        except Exception:
            # Demo mode: continue without trained model
            self.model = None
            self._is_initialized = True
    
    def handle(self, request: dict) -> dict:
        """Handle prediction request.
        
        Demo mode: If no model loaded, echo the request back.
        """
        if self.model is None:
            # Demo mode
            features = self.preprocess(request)
            return self.postprocess(features)
        
        # Normal mode with trained model
        return super().handle(request)
