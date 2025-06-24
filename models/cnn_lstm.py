"""
CNN-LSTM Model for Bitcoin Walk-Forward Backtesting
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd
from typing import Tuple, Optional
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

import config


class CNNLSTMModel:
    """
    CNN-LSTM model for time series classification
    Architecture: Conv1D -> Conv1D -> LSTM -> Dense
    """
    
    def __init__(self, input_shape: Tuple[int, int], num_classes: int = 3):
        """
        Initialize CNN-LSTM model
        
        Parameters:
        -----------
        input_shape : tuple
            Shape of input sequences (lookback, n_features)
        num_classes : int
            Number of output classes (3 for triple barrier: -1, 0, 1)
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
        self.scaler = StandardScaler()
        self.is_fitted = False
        
        # Set random seeds for reproducibility
        tf.random.set_seed(config.RANDOM_SEED)
        np.random.seed(config.RANDOM_SEED)
    
    def build_model(self) -> keras.Model:
        """Build MUCH MORE COMPLEX CNN-LSTM architecture"""
        model = keras.Sequential([
            # Input layer
            layers.Input(shape=self.input_shape),
            
            # First Conv1D layer - much larger
            layers.Conv1D(
                filters=config.CNN_FILTERS_1,
                kernel_size=config.CNN_KERNEL_SIZE,
                activation='relu',
                padding='same',
                kernel_regularizer=keras.regularizers.l2(config.L2_REG)
            ),
            layers.BatchNormalization(),
            layers.Dropout(config.DROPOUT_RATE),
            
            # Second Conv1D layer with different kernel size
            layers.Conv1D(
                filters=config.CNN_FILTERS_2,
                kernel_size=config.CNN_KERNEL_SIZE_2,
                activation='relu',
                padding='same',
                kernel_regularizer=keras.regularizers.l2(config.L2_REG)
            ),
            layers.BatchNormalization(),
            layers.Dropout(config.DROPOUT_RATE),
            
            # Third Conv1D layer
            layers.Conv1D(
                filters=config.CNN_FILTERS_3,
                kernel_size=config.CNN_KERNEL_SIZE,
                activation='relu',
                padding='same',
                kernel_regularizer=keras.regularizers.l2(config.L2_REG)
            ),
            layers.BatchNormalization(),
            layers.Dropout(config.DROPOUT_RATE),
            
            # First LSTM layer - much larger with return sequences
            layers.LSTM(
                units=config.LSTM_UNITS_1,
                return_sequences=True,
                dropout=config.DROPOUT_RATE_LSTM,
                recurrent_dropout=config.DROPOUT_RATE_LSTM,
                kernel_regularizer=keras.regularizers.l2(config.L2_REG)
            ),
            layers.BatchNormalization(),
            
            # Second LSTM layer
            layers.LSTM(
                units=config.LSTM_UNITS_2,
                return_sequences=False,
                dropout=config.DROPOUT_RATE_LSTM,
                recurrent_dropout=config.DROPOUT_RATE_LSTM,
                kernel_regularizer=keras.regularizers.l2(config.L2_REG)
            ),
            layers.BatchNormalization(),
            layers.Dropout(config.DROPOUT_RATE),
            
            # First Dense layer - large
            layers.Dense(
                units=config.DENSE_UNITS_1,
                activation='relu',
                kernel_regularizer=keras.regularizers.l2(config.L2_REG)
            ),
            layers.BatchNormalization(),
            layers.Dropout(config.DROPOUT_RATE),
            
            # Second Dense layer
            layers.Dense(
                units=config.DENSE_UNITS_2,
                activation='relu',
                kernel_regularizer=keras.regularizers.l2(config.L2_REG)
            ),
            layers.Dropout(config.DROPOUT_RATE),
            
            # Third Dense layer
            layers.Dense(
                units=32,
                activation='relu',
                kernel_regularizer=keras.regularizers.l2(config.L2_REG)
            ),
            layers.Dropout(config.DROPOUT_RATE),
            
            # Dense output layer
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        # Compile with learning rate scheduling
        initial_learning_rate = config.LEARNING_RATE
        lr_schedule = keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate,
            decay_steps=100,
            decay_rate=config.LEARNING_RATE_DECAY,
            staircase=True
        )
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def preprocess_data(self, X: np.ndarray, y: np.ndarray = None, 
                       is_training: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Preprocess data for training/prediction
        
        Parameters:
        -----------
        X : np.ndarray
            Input sequences of shape (n_samples, lookback, n_features)
        y : np.ndarray, optional
            Labels of shape (n_samples,)
        is_training : bool
            Whether this is training data (affects scaler fitting)
        
        Returns:
        --------
        Tuple of (X_scaled, y_encoded)
        """
        # Reshape X for scaling: (n_samples * lookback, n_features)
        n_samples, lookback, n_features = X.shape
        X_reshaped = X.reshape(-1, n_features)
        
        if is_training:
            # Fit scaler on training data
            X_scaled_reshaped = self.scaler.fit_transform(X_reshaped)
        else:
            # Use fitted scaler for validation/test data
            X_scaled_reshaped = self.scaler.transform(X_reshaped)
        
        # Reshape back to original shape
        X_scaled = X_scaled_reshaped.reshape(n_samples, lookback, n_features)
        
        # Encode labels if provided
        y_encoded = None
        if y is not None:
            # Convert labels from {-1, 0, 1} to {0, 1, 2}
            y_shifted = y + 1
            y_encoded = keras.utils.to_categorical(y_shifted, num_classes=self.num_classes)
        
        return X_scaled, y_encoded
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: np.ndarray = None, y_val: np.ndarray = None,
            epochs: int = None, batch_size: int = None,
            verbose: int = 1) -> keras.callbacks.History:
        """
        Train the CNN-LSTM model
        
        Parameters:
        -----------
        X_train : np.ndarray
            Training sequences
        y_train : np.ndarray
            Training labels
        X_val : np.ndarray, optional
            Validation sequences
        y_val : np.ndarray, optional
            Validation labels
        epochs : int, optional
            Number of training epochs
        batch_size : int, optional
            Batch size for training
        verbose : int
            Verbosity level
        
        Returns:
        --------
        Training history
        """
        if epochs is None:
            epochs = config.EPOCHS
        if batch_size is None:
            batch_size = config.BATCH_SIZE
        
        # Build model if not already built
        if self.model is None:
            self.model = self.build_model()
        
        # Preprocess training data
        X_train_scaled, y_train_encoded = self.preprocess_data(
            X_train, y_train, is_training=True)

        # Compute class weights to handle any label imbalance
        y_train_shifted = y_train + 1
        class_weights = keras.utils.class_weight.compute_class_weight(
            "balanced", classes=np.arange(self.num_classes), y=y_train_shifted)
        class_weight_dict = {i: w for i, w in enumerate(class_weights)}
        
        # Preprocess validation data if provided
        validation_data = None
        if X_val is not None and y_val is not None:
            X_val_scaled, y_val_encoded = self.preprocess_data(X_val, y_val, is_training=False)
            validation_data = (X_val_scaled, y_val_encoded)
        
        # Setup callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss' if validation_data else 'loss',
                patience=10,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss' if validation_data else 'loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7
            )
        ]
        
        # Train model
        history = self.model.fit(
            X_train_scaled,
            y_train_encoded,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=verbose,
            class_weight=class_weight_dict,
        )
        
        self.is_fitted = True
        return history
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities
        
        Parameters:
        -----------
        X : np.ndarray
            Input sequences
        
        Returns:
        --------
        np.ndarray of shape (n_samples, n_classes)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X_scaled, _ = self.preprocess_data(X, is_training=False)
        probabilities = self.model.predict(X_scaled, verbose=0)
        
        return probabilities
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels
        
        Parameters:
        -----------
        X : np.ndarray
            Input sequences
        
        Returns:
        --------
        np.ndarray of predicted labels in {-1, 0, 1}
        """
        probabilities = self.predict_proba(X)
        predictions = np.argmax(probabilities, axis=1)
        
        # Convert back from {0, 1, 2} to {-1, 0, 1}
        predictions = predictions - 1
        
        return predictions
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray, 
                verbose: int = 1) -> dict:
        """
        Evaluate model performance
        
        Parameters:
        -----------
        X_test : np.ndarray
            Test sequences
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
        X_test_scaled, y_test_encoded = self.preprocess_data(X_test, y_test, is_training=False)
        loss, accuracy = self.model.evaluate(X_test_scaled, y_test_encoded, verbose=0)
        
        if verbose > 0:
            print(f"\nCNN-LSTM Model Evaluation:")
            print(f"Loss: {loss:.4f}")
            print(f"Accuracy: {accuracy:.4f}")
            print("\nClassification Report:")
            print(classification_report(y_test, y_pred, target_names=['Stop', 'Timeout', 'Profit']))
            print("\nConfusion Matrix:")
            print(confusion_matrix(y_test, y_pred))
        
        return {
            'loss': loss,
            'accuracy': accuracy,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
    
    def save_model(self, filepath: str):
        """Save the trained model"""
        if self.model is not None:
            self.model.save(filepath)
            print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained model"""
        self.model = keras.models.load_model(filepath)
        self.is_fitted = True
        print(f"Model loaded from {filepath}")


if __name__ == "__main__":
    # Test CNN-LSTM model
    print("Testing CNN-LSTM model...")
    
    # Create dummy data for testing
    lookback, n_features = config.LOOKBACK, 18  # Assuming 18 features
    n_samples = 1000
    
    X_test = np.random.randn(n_samples, lookback, n_features)
    y_test = np.random.choice([-1, 0, 1], size=n_samples)
    
    # Initialize and test model
    model = CNNLSTMModel(input_shape=(lookback, n_features))
    
    # Split data
    split_idx = int(0.8 * n_samples)
    X_train, X_val = X_test[:split_idx], X_test[split_idx:]
    y_train, y_val = y_test[:split_idx], y_test[split_idx:]
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Training labels shape: {y_train.shape}")
    
    # Train model
    history = model.fit(X_train, y_train, X_val, y_val, epochs=5, verbose=1)
    
    # Evaluate
    results = model.evaluate(X_val, y_val)
    
    print("CNN-LSTM model test completed successfully!") 