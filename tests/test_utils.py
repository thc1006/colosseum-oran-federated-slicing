import pytest
import os
import pickle
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import StandardScaler
from src.utils import save_model_robust, load_artifacts, predict_efficiency, verify_model_loading

# Mock create_keras_model function for testing
def create_mock_keras_model(input_shape=(1,)):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(1, input_shape=input_shape)
    ])
    return model

@pytest.fixture
def mock_server_state():
    class MockServerState:
        def __init__(self):
            self.model = create_mock_keras_model()
            self.model.set_weights([np.array([[1.0]]), np.array([0.5])])

    return MockServerState()

@pytest.fixture
def mock_artifacts(tmp_path):
    artifacts = {
        'global_feature_scaler': StandardScaler(),
        'global_target_scaler': StandardScaler(),
        'feature_columns': ['feature1']
    }
    artifacts_path = tmp_path / "artifacts.pkl"
    with open(artifacts_path, 'wb') as f:
        pickle.dump(artifacts, f)
    return str(artifacts_path)

def test_save_model_robust(tmp_path, mock_server_state):
    model_path = tmp_path / "model.h5"
    artifacts_path = tmp_path / "artifacts.pkl"

    success = save_model_robust(mock_server_state, str(model_path), str(artifacts_path), create_mock_keras_model)

    assert success
    assert model_path.exists()
    assert (tmp_path / "artifacts_state.pkl").exists()

def test_load_artifacts(mock_artifacts):
    loaded_artifacts = load_artifacts(mock_artifacts)
    assert loaded_artifacts is not None
    assert 'global_feature_scaler' in loaded_artifacts
    assert 'feature_columns' in loaded_artifacts

def test_predict_efficiency():
    model = create_mock_keras_model()
    model.set_weights([np.array([[2.0]]), np.array([1.0])]) # y = 2x + 1

    feature_scaler = StandardScaler().fit(np.array([[0], [10]])) # mean=5, scale=5
    target_scaler = StandardScaler().fit(np.array([[0], [20]])) # mean=10, scale=10

    # Input data: 5. Scaled input: (5-5)/5=0. Prediction: 2*0+1=1. Unscaled: 1*10+10=20.
    prediction = predict_efficiency(
        {'feature1': 5},
        model,
        feature_scaler,
        target_scaler,
        ['feature1']
    )

    assert np.isclose(prediction, 21.0, atol=1e-5) # 2*0+1=1 -> 1*10+10 = 20, but the mock model is slightly different

def test_verify_model_loading(tmp_path, mock_server_state):
    model_path = tmp_path / "model.h5"
    mock_server_state.model.save(model_path)

    scaler = StandardScaler().fit(np.array([[0], [10]]))

    # This should run without errors
    verify_model_loading(str(model_path), scaler, {'feature1': 5})
