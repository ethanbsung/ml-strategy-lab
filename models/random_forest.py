"""
Random Forest Model for Bitcoin Walk-Forward Backtesting
Baseline model for comparison with CNN-LSTM
"""
import numpy as np
import pandas as pd
from typing import Tuple, Optional
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import warnings
warnings.filterwarnings('ignore')

import config


class RandomForestModel:
    """
    Random Forest model for time series classification
    Serves as baseline model for comparison
    """
    
    def __init__(self, n_estimators: int = None, max_depth: int = None,
                 min_samples_split: int = None, min_samples_leaf: int = None,
                 random_state: int = None):
        """
        Initialize Random Forest model
        
        Parameters:
        -----------
        n_estimators : int
            Number of trees in the forest
        max_depth : int
            Maximum depth of trees
        min_samples_split : int
            Minimum samples required to split a node
        min_samples_leaf : int
            Minimum samples required at a leaf node
        random_state : int
            Random state for reproducibility
        """
        # Use config defaults if not specified
        if n_estimators is None:
            n_estimators = config.RF_N_ESTIMATORS
        if max_depth is None:
            max_depth = config.RF_MAX_DEPTH
        if min_samples_split is None:
            min_samples_split = config.RF_MIN_SAMPLES_SPLIT
        if min_samples_leaf is None:
            min_samples_leaf = config.RF_MIN_SAMPLES_LEAF
        if random_state is None:
            random_state = config.RANDOM_SEED
        
        # Initialize models with ANTI-OVERFITTING parameters
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=getattr(config, 'RF_MAX_FEATURES', 0.6),  # Feature subsampling
            class_weight='balanced',
            bootstrap=True,          # Ensure bootstrap sampling
            oob_score=True,         # Out-of-bag scoring for validation
            random_state=random_state,
            n_jobs=-1
        )
        
        self.scaler = StandardScaler()
        self.is_fitted = False
    
    def flatten_sequences(self, X: np.ndarray) -> np.ndarray:
        """
        Flatten 3D sequences to 2D for Random Forest
        
        Parameters:
        -----------
        X : np.ndarray
            Input sequences of shape (n_samples, lookback, n_features)
        
        Returns:
        --------
        np.ndarray of shape (n_samples, lookback * n_features)
        """
        if len(X.shape) == 3:
            n_samples, lookback, n_features = X.shape
            return X.reshape(n_samples, lookback * n_features)
        else:
            return X
    
    def preprocess_data(self, X: np.ndarray, is_training: bool = True) -> np.ndarray:
        """
        Preprocess data for Random Forest
        
        Parameters:
        -----------
        X : np.ndarray
            Input data
        is_training : bool
            Whether this is training data (affects scaler fitting)
        
        Returns:
        --------
        np.ndarray of scaled features
        """
        # Flatten sequences if necessary
        X_flat = self.flatten_sequences(X)
        
        if is_training:
            # Fit scaler on training data
            X_scaled = self.scaler.fit_transform(X_flat)
        else:
            # Use fitted scaler for validation/test data
            X_scaled = self.scaler.transform(X_flat)
        
        return X_scaled
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: np.ndarray = None, y_val: np.ndarray = None,
            verbose: int = 1) -> dict:
        """
        Train the Random Forest model
        
        Parameters:
        -----------
        X_train : np.ndarray
            Training data
        y_train : np.ndarray
            Training labels
        X_val : np.ndarray, optional
            Validation data (for evaluation only)
        y_val : np.ndarray, optional
            Validation labels (for evaluation only)
        verbose : int
            Verbosity level
        
        Returns:
        --------
        Training metrics
        """
        if verbose > 0:
            print("Training Random Forest model...")
        
        # Preprocess training data
        X_train_scaled = self.preprocess_data(X_train, is_training=True)
        
        # Train model
        self.model.fit(X_train_scaled, y_train)
        self.is_fitted = True

        # Calculate training metrics
        train_accuracy = self.model.score(X_train_scaled, y_train)
        oob_score = getattr(self.model, "oob_score_", None)
        
        if verbose > 0:
            print(f"Training accuracy: {train_accuracy:.4f}")
            if oob_score is not None:
                print(f"OOB score: {oob_score:.4f}")
        
        # Evaluate on validation set if provided
        val_accuracy = None
        if X_val is not None and y_val is not None:
            val_accuracy = self.evaluate(X_val, y_val, verbose=verbose)['accuracy']
        
        return {
            'train_accuracy': train_accuracy,
            'val_accuracy': val_accuracy,
            'oob_score': oob_score,
        }
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities
        
        Parameters:
        -----------
        X : np.ndarray
            Input data
        
        Returns:
        --------
        np.ndarray of shape (n_samples, n_classes)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X_scaled = self.preprocess_data(X, is_training=False)
        probabilities = self.model.predict_proba(X_scaled)
        
        return probabilities
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels
        
        Parameters:
        -----------
        X : np.ndarray
            Input data
        
        Returns:
        --------
        np.ndarray of predicted labels
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X_scaled = self.preprocess_data(X, is_training=False)
        predictions = self.model.predict(X_scaled)
        
        return predictions
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray, 
                verbose: int = 1) -> dict:
        """
        Evaluate model performance
        
        Parameters:
        -----------
        X_test : np.ndarray
            Test data
        y_test : np.ndarray
            Test labels
        verbose : int
            Verbosity level
        
        Returns:
        --------
        Dictionary with evaluation metrics
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before evaluation")
        
        # Get predictions
        y_pred = self.predict(X_test)
        y_pred_proba = self.predict_proba(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        
        if verbose > 0:
            print(f"\nRandom Forest Model Evaluation:")
            print(f"Accuracy: {accuracy:.4f}")
            print("\nClassification Report:")
            print(classification_report(y_test, y_pred, target_names=['Stop', 'Timeout', 'Profit']))
            print("\nConfusion Matrix:")
            print(confusion_matrix(y_test, y_pred))
        
        return {
            'accuracy': accuracy,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
    
    def get_feature_importance(self, feature_names: list = None) -> pd.DataFrame:
        """
        Get feature importance scores
        
        Parameters:
        -----------
        feature_names : list, optional
            Names of features
        
        Returns:
        --------
        pd.DataFrame with feature importance scores
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting feature importance")
        
        importance = self.model.feature_importances_
        
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(importance))]
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def save_model(self, filepath: str):
        """Save the trained model"""
        if self.is_fitted:
            joblib.dump({
                'model': self.model,
                'scaler': self.scaler,
                'is_fitted': self.is_fitted
            }, filepath)
            print(f"Model saved to {filepath}")
        else:
            print("No fitted model to save")
    
    def load_model(self, filepath: str):
        """Load a trained model"""
        try:
            model_data = joblib.load(filepath)
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.is_fitted = model_data['is_fitted']
            print(f"Model loaded from {filepath}")
        except Exception as e:
            print(f"Error loading model: {e}")


if __name__ == "__main__":
    # Test Random Forest model
    print("Testing Random Forest model...")
    
    # Create dummy data for testing
    lookback, n_features = config.LOOKBACK, 18  # Assuming 18 features
    n_samples = 1000
    
    # Test with both 2D and 3D input
    print("\nTesting with 3D input (sequences):")
    X_test_3d = np.random.randn(n_samples, lookback, n_features)
    y_test = np.random.choice([-1, 0, 1], size=n_samples)
    
    # Initialize and test model
    model = RandomForestModel()
    
    # Split data
    split_idx = int(0.8 * n_samples)
    X_train_3d, X_val_3d = X_test_3d[:split_idx], X_test_3d[split_idx:]
    y_train, y_val = y_test[:split_idx], y_test[split_idx:]
    
    print(f"Training data shape: {X_train_3d.shape}")
    print(f"Training labels shape: {y_train.shape}")
    
    # Train model
    train_results = model.fit(X_train_3d, y_train, X_val_3d, y_val, verbose=1)
    
    # Evaluate
    eval_results = model.evaluate(X_val_3d, y_val)
    
    # Feature importance
    feature_importance = model.get_feature_importance()
    print(f"\nTop 10 most important features:")
    print(feature_importance.head(10))
    
    print("\nTesting with 2D input (flattened features):")
    X_test_2d = np.random.randn(n_samples, n_features)
    
    # Test with 2D input
    model_2d = RandomForestModel()
    X_train_2d, X_val_2d = X_test_2d[:split_idx], X_test_2d[split_idx:]
    
    train_results_2d = model_2d.fit(X_train_2d, y_train, X_val_2d, y_val, verbose=1)
    eval_results_2d = model_2d.evaluate(X_val_2d, y_val)
    
    print("Random Forest model test completed successfully!") 