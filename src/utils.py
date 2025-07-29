"""
This module provides utility functions for the federated learning pipeline.
"""
import os
import pickle
import logging
import tensorflow as tf
import numpy as np
import pandas as pd

def save_model_robust(server_state, model_save_path, artifacts_path, create_keras_model_fn):
    """
    Saves the model and training state robustly.
    """
    try:
        logging.info("Saving model...")
        save_dir = os.path.dirname(model_save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        model_weights = None
        if hasattr(server_state, 'global_model_weights'):
            model_weights = server_state.global_model_weights.trainable
        elif hasattr(server_state, 'model'):
            model_weights = server_state.model.trainable

        if model_weights is not None:
            final_model = create_keras_model_fn()
            final_model.set_weights(model_weights)
            final_model.compile(optimizer='adam', loss='mse', metrics=['mae'])
            final_model.save(model_save_path)
            logging.info("Model saved to: %s", model_save_path)

            if artifacts_path:
                state_path = artifacts_path.replace('.pkl', '_state.pkl')
                with open(state_path, 'wb') as f:
                    pickle.dump({'server_state': server_state, 'model_weights': model_weights}, f)
                logging.info("Training state saved to: %s", state_path)
            return True

        logging.error("Could not extract model weights.")
        return False
    except Exception as e:
        logging.error("Failed to save model: %s", e)
        return False

def load_artifacts(artifacts_path):
    """
    Loads artifacts from a pickle file.
    """
    try:
        with open(artifacts_path, 'rb') as f:
            artifacts = pickle.load(f)
        logging.info("Artifacts loaded from %s", artifacts_path)
        return artifacts
    except Exception as e:
        logging.error("Failed to load artifacts: %s", e)
        return None

def predict_efficiency(input_data, model, feature_scaler, target_scaler, feature_columns):
    """
    Predicts resource allocation efficiency for a given input.
    """
    input_df = pd.DataFrame([input_data], columns=feature_columns)
    scaled_input = feature_scaler.transform(input_df)
    scaled_prediction = model.predict(scaled_input, verbose=0)
    unscaled_prediction = target_scaler.inverse_transform(scaled_prediction)
    return unscaled_prediction[0][0]

def verify_model_loading(model_save_path, global_feature_scaler, default_values):
    """
    Verifies that a saved model can be loaded and used for prediction.
    """
    if os.path.exists(model_save_path):
        logging.info("Model exists at '%s'", model_save_path)
        loaded_model = tf.keras.models.load_model(model_save_path)
        logging.info("Model loaded successfully for verification.")

        test_input = np.array([list(default_values.values())])
        test_input_scaled = global_feature_scaler.transform(test_input)

        try:
            loaded_model.predict(test_input_scaled, verbose=0)
            logging.info("Prediction with loaded model successful.")
        except Exception as e:
            logging.error("Prediction with loaded model failed: %s", e)
    else:
        logging.warning("Model file does not exist at '%s'", model_save_path)
