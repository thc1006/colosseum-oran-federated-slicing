import pytest
import tensorflow as tf
import tensorflow_federated as tff
import numpy as np
from src.federated_learning import (
    create_keras_model,
    model_fn,
    create_dp_optimizer,
    DPKerasOptimizer,
    PrivacyBudgetManager
)

def test_create_keras_model():
    model = create_keras_model(input_shape=(13,))
    assert isinstance(model, tf.keras.Model)
    assert len(model.layers) == 5
    assert model.output_shape == (None, 1)

def test_model_fn():
    tff_model = model_fn(input_shape=(13,))
    assert isinstance(tff_model, tff.learning.models.FunctionalModel)
    assert tff_model.input_spec[0]['x'].shape[1] == 13

def test_create_dp_optimizer():
    optimizer = create_dp_optimizer(0.001, 1.0, 0.1, 1)
    assert isinstance(optimizer, DPKerasOptimizer)
    assert optimizer.l2_norm_clip == 1.0
    assert optimizer.noise_multiplier == 0.1

def test_dp_keras_optimizer_gradients():
    base_optimizer = tf.keras.optimizers.SGD(learning_rate=0.1)
    optimizer = DPKerasOptimizer(
        base_optimizer=base_optimizer,
        l2_norm_clip=1.0,
        noise_multiplier=0.5,
        num_microbatches=1
    )

    var = tf.Variable([1.0, 2.0])
    grad = tf.constant([0.5, 0.5])

    dp_grads = optimizer._apply_dp_to_gradients([grad])

    assert len(dp_grads) == 1
    assert dp_grads[0].shape == grad.shape
    # Check that noise was added (gradients won't be exactly the same)
    assert not np.allclose(dp_grads[0].numpy(), grad.numpy())

def test_privacy_budget_manager():
    manager = PrivacyBudgetManager(target_epsilon=2.0, target_delta=1e-5, max_rounds=10, estimated_data_size=1000)

    assert not manager.should_stop_training(current_round=1)

    manager.update_privacy_spent(1.5)
    assert not manager.should_stop_training(current_round=2)

    manager.update_privacy_spent(0.6)
    assert manager.should_stop_training(current_round=3)

def test_privacy_budget_prediction():
    manager = PrivacyBudgetManager(target_epsilon=2.0, target_delta=1e-5, max_rounds=10, estimated_data_size=1000)
    manager.update_privacy_spent(0.5) # Round 1
    manager.update_privacy_spent(0.5) # Round 2
    # After 2 rounds, consumed 1.0. Avg is 0.5. Remaining 8 rounds will consume 4.0. Total > 2.0 * 1.1
    assert manager.should_stop_training(current_round=3)
