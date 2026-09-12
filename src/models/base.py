from abc import ABC, abstractmethod
import pandas as pd

class BaseModel(ABC):
    """
    Abstract Base Class for all machine learning models.
    
    Any model that wants to be used in the ModelTrainer MUST inherit from this class
    and implement both 'fit' and 'predict' methods.
    """
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Train the model on the provided features and target.

        Args:
            X: Feature matrix (DataFrame)
            y: Target vector (Series)
        """
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame):
        """
        Generate predictions on new data.
        
        Args:
            X: Feature matrix (DataFrame).
        
        Returns:
            Predictions (array or Series).
        """
        pass
