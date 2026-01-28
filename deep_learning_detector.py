"""
Deep Learning NIDS - LSTM-based sequence detection
Detects multi-step and sophisticated attacks using temporal patterns
"""

import numpy as np
import logging
from typing import Tuple, Optional
import pickle

logger = logging.getLogger(__name__)


class LSTMDetector:
    """Deep learning detector for sequence-based attacks"""

    def __init__(self, sequence_length: int = 10, features: int = 11):
        """
        Initialize LSTM detector

        Args:
            sequence_length: Number of packets in sequence
            features: Number of features per packet
        """
        self.sequence_length = sequence_length
        self.features = features
        self.model = None
        self.packet_buffer = []
        self._build_model()

    def _build_model(self):
        """Build LSTM model"""
        try:
            import tensorflow as tf
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization

            logger.info("Building LSTM model...")

            self.model = Sequential([
                LSTM(128, input_shape=(self.sequence_length, self.features),
                     return_sequences=True, name='lstm_1'),
                Dropout(0.2, name='dropout_1'),
                BatchNormalization(name='batch_norm_1'),

                LSTM(64, return_sequences=False, name='lstm_2'),
                Dropout(0.2, name='dropout_2'),
                BatchNormalization(name='batch_norm_2'),

                Dense(32, activation='relu', name='dense_1'),
                Dropout(0.1, name='dropout_3'),

                Dense(1, activation='sigmoid', name='output')
            ])

            self.model.compile(
                optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy', 'precision', 'recall']
            )

            logger.info("✅ LSTM model built successfully")
            logger.info(f"Model parameters: {self.model.count_params():,}")

        except ImportError:
            logger.error("TensorFlow not installed. Install with: pip install tensorflow")
            raise
        except Exception as e:
            logger.error(f"Error building LSTM model: {e}")
            raise

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              epochs: int = 20, batch_size: int = 32, validation_split: float = 0.2):
        """
        Train LSTM model

        Args:
            X_train: Training sequences (samples, sequence_length, features)
            y_train: Training labels (samples,)
            epochs: Number of training epochs
            batch_size: Batch size for training
            validation_split: Fraction of data for validation
        """
        if self.model is None:
            raise ValueError("Model not built")

        logger.info(f"Training LSTM model on {len(X_train)} sequences...")
        logger.info(f"Sequence shape: {X_train.shape}")

        try:
            import tensorflow as tf

            # Callbacks
            callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=5,
                    restore_best_weights=True
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=3,
                    min_lr=0.00001
                )
            ]

            # Train
            history = self.model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=validation_split,
                callbacks=callbacks,
                verbose=1
            )

            logger.info("✅ Training complete")

            # Print final metrics
            final_acc = history.history['accuracy'][-1]
            final_val_acc = history.history['val_accuracy'][-1]
            logger.info(f"Final accuracy: {final_acc:.4f}")
            logger.info(f"Final validation accuracy: {final_val_acc:.4f}")

            return history

        except Exception as e:
            logger.error(f"Training error: {e}")
            raise

    def predict_sequence(self, packet_sequence: np.ndarray) -> float:
        """
        Predict if sequence is attack

        Args:
            packet_sequence: Sequence array (sequence_length, features)

        Returns:
            Attack probability (0-1)
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded")

        try:
            # Reshape for prediction
            X = packet_sequence.reshape(1, self.sequence_length, self.features)

            # Predict
            prediction = self.model.predict(X, verbose=0)[0][0]

            return float(prediction)

        except Exception as e:
            logger.debug(f"Prediction error: {e}")
            return 0.0

    def add_packet_and_predict(self, packet_features: np.ndarray,
                               threshold: float = 0.5) -> Tuple[bool, float]:
        """
        Add packet to buffer and check sequence

        Args:
            packet_features: Feature array for single packet
            threshold: Detection threshold (0-1)

        Returns:
            Tuple of (is_attack, probability)
        """
        # Add to buffer
        self.packet_buffer.append(packet_features)

        # Keep only last N packets
        if len(self.packet_buffer) > self.sequence_length:
            self.packet_buffer.pop(0)

        # Need full sequence
        if len(self.packet_buffer) < self.sequence_length:
            return False, 0.0

        # Predict
        sequence = np.array(self.packet_buffer)
        probability = self.predict_sequence(sequence)

        return probability > threshold, probability

    def reset_buffer(self):
        """Reset packet buffer"""
        self.packet_buffer = []

    def save_model(self, filepath: str):
        """
        Save trained model

        Args:
            filepath: Path to save model
        """
        if self.model is None:
            raise ValueError("No model to save")

        try:
            self.model.save(filepath)
            logger.info(f"✅ Model saved to {filepath}")

        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise

    def load_model(self, filepath: str):
        """
        Load trained model

        Args:
            filepath: Path to model file
        """
        try:
            import tensorflow as tf

            self.model = tf.keras.models.load_model(filepath)
            logger.info(f"✅ Model loaded from {filepath}")

        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise


def create_sequences_from_data(data: np.ndarray, labels: np.ndarray,
                               sequence_length: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sequences from flat data

    Args:
        data: Feature array (samples, features)
        labels: Label array (samples,)
        sequence_length: Length of sequences

    Returns:
        Tuple of (sequence_data, sequence_labels)
    """
    X_sequences = []
    y_sequences = []

    for i in range(len(data) - sequence_length):
        X_sequences.append(data[i:i+sequence_length])
        y_sequences.append(labels[i+sequence_length])

    return np.array(X_sequences), np.array(y_sequences)


def train_lstm_model(data_path: Optional[str] = None):
    """
    Train LSTM on NSL-KDD sequences

    Args:
        data_path: Path to dataset (optional)
    """
    logger.info("=== LSTM Model Training ===")

    # Load data
    from main import load_nsl_kdd_dataset

    logger.info("Loading NSL-KDD dataset...")
    df = load_nsl_kdd_dataset()

    # Prepare features
    feature_columns = [
        'duration', 'src_bytes', 'dst_bytes', 'wrong_fragment',
        'urgent', 'hot', 'num_failed_logins', 'num_compromised',
        'num_root', 'num_file_creations', 'count'
    ]

    X = df[feature_columns].values
    y = df['label'].values

    # Normalize
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Create sequences
    logger.info("Creating sequences...")
    sequence_length = 10
    X_sequences, y_sequences = create_sequences_from_data(
        X_scaled, y, sequence_length
    )

    logger.info(f"Created {len(X_sequences)} sequences")

    # Split train/test
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X_sequences, y_sequences, test_size=0.2, random_state=42
    )

    # Train
    detector = LSTMDetector(sequence_length=sequence_length, features=len(feature_columns))
    history = detector.train(X_train, y_train, epochs=20, batch_size=32)

    # Evaluate
    logger.info("Evaluating model...")
    loss, accuracy, precision, recall = detector.model.evaluate(X_test, y_test, verbose=0)

    logger.info(f"\n{'='*60}")
    logger.info(f"Final Test Metrics:")
    logger.info(f"  Accuracy:  {accuracy:.4f}")
    logger.info(f"  Precision: {precision:.4f}")
    logger.info(f"  Recall:    {recall:.4f}")
    logger.info(f"  F1-Score:  {2 * (precision * recall) / (precision + recall):.4f}")
    logger.info(f"{'='*60}\n")

    # Save
    detector.save_model('lstm_nids_model.h5')

    # Save scaler
    with open('lstm_scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    logger.info("✅ Scaler saved to lstm_scaler.pkl")

    return detector


# Example usage
def example_usage():
    """Example of using LSTM detector"""

    # Create detector
    detector = LSTMDetector(sequence_length=10, features=11)

    # Simulate packet stream
    print("\nProcessing packet stream...")
    for i in range(15):
        # Random packet features
        packet_features = np.random.randn(11)

        # Check sequence
        is_attack, probability = detector.add_packet_and_predict(packet_features)

        if is_attack:
            print(f"🚨 Packet {i}: ATTACK DETECTED (prob: {probability:.4f})")
        else:
            print(f"✓ Packet {i}: Normal (prob: {probability:.4f})")


if __name__ == "__main__":
    import sys

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    if len(sys.argv) > 1 and sys.argv[1] == 'train':
        print("Training LSTM model...")
        train_lstm_model()
    else:
        print("LSTM Detector Example\n")
        example_usage()
