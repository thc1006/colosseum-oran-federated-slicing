"""
This module provides classes and functions for federated learning with differential privacy.
"""
from collections import OrderedDict
from typing import Optional, Tuple, List
import warnings
import numpy as np
import tensorflow as tf
import tensorflow_federated as tff

warnings.filterwarnings('ignore')

class DPKerasOptimizer(tf.keras.optimizers.Optimizer):
    """
    A Keras optimizer that implements differential privacy.
    """
    def __init__(self,
                 base_optimizer: tf.keras.optimizers.Optimizer,
                 l2_norm_clip: float,
                 noise_multiplier: float,
                 num_microbatches: Optional[int] = None,
                 gradient_accumulation_steps: int = 1,
                 name="DPKerasOptimizer",
                 **kwargs):
        super().__init__(name, **kwargs)
        self.base_optimizer = base_optimizer
        self.l2_norm_clip = l2_norm_clip
        self.noise_multiplier = noise_multiplier
        self.num_microbatches = num_microbatches
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self._accumulated_gradients = None
        self._accumulation_counter = tf.Variable(0, dtype=tf.int64, trainable=False)

    def _create_slots(self, var_list):
        self.base_optimizer._create_slots(var_list)
        if self.gradient_accumulation_steps > 1:
            for var in var_list:
                self.add_slot(var, 'accum_grad')

    def _resource_apply_dense(self, grad, var, apply_state=None):
        if self.gradient_accumulation_steps > 1:
            accum_grad = self.get_slot(var, 'accum_grad')

            def update_step():
                avg_grad = accum_grad / tf.cast(self.gradient_accumulation_steps, grad.dtype)
                self.base_optimizer._resource_apply_dense(avg_grad, var, apply_state)
                return accum_grad.assign(tf.zeros_like(accum_grad), use_locking=self._use_locking)

            def accumulation_step():
                return accum_grad.assign_add(grad, use_locking=self._use_locking)

            return tf.cond(
                tf.equal(self._accumulation_counter, self.gradient_accumulation_steps - 1),
                update_step,
                accumulation_step
            )
        return self.base_optimizer._resource_apply_dense(grad, var, apply_state)

    def _resource_apply_sparse(self, grad, var, apply_state=None):
        raise NotImplementedError("Sparse gradients are not supported.")

    def get_gradients(self, loss, params):
        grads = super().get_gradients(loss, params)
        return self._apply_dp_to_gradients(grads)

    def _apply_dp_to_gradients(self, gradients: List[tf.Tensor]) -> List[tf.Tensor]:
        dp_gradients = []
        for grad in gradients:
            if grad is None:
                dp_gradients.append(None)
                continue

            grad_norm = tf.norm(tf.reshape(grad, [-1]), ord=2)
            divisor = tf.maximum(grad_norm / self.l2_norm_clip, 1.)
            clipped_grad = grad / divisor

            noise_stddev = self.l2_norm_clip * self.noise_multiplier
            noise = tf.random.normal(
                shape=tf.shape(clipped_grad),
                mean=0.0,
                stddev=noise_stddev,
                dtype=clipped_grad.dtype
            )

            if self.num_microbatches is not None and self.num_microbatches > 1:
                noise *= tf.sqrt(tf.cast(self.num_microbatches, clipped_grad.dtype))

            dp_grad = clipped_grad + noise
            dp_gradients.append(dp_grad)

        return dp_gradients

    def get_config(self):
        config = self.base_optimizer.get_config()
        config.update({
            'l2_norm_clip': self.l2_norm_clip,
            'noise_multiplier': self.noise_multiplier,
            'num_microbatches': self.num_microbatches,
            'gradient_accumulation_steps': self.gradient_accumulation_steps,
        })
        return config

class PrivacyBudgetManager:
    """
    Manages the privacy budget for differentially private federated learning.
    """
    def __init__(self, target_epsilon: float, target_delta: float,
                 max_rounds: int, estimated_data_size: int):
        self.target_epsilon = target_epsilon
        self.target_delta = target_delta
        self.max_rounds = max_rounds
        self.estimated_data_size = estimated_data_size
        self.consumed_epsilon = 0.0
        self.round_epsilons = []

    def should_stop_training(self, current_round: int) -> bool:
        """
        Determines if the training should stop based on the privacy budget.
        """
        if self.consumed_epsilon >= self.target_epsilon:
            return True
        if self.round_epsilons:
            avg_epsilon_per_round = np.mean(self.round_epsilons)
            remaining_rounds = self.max_rounds - current_round
            predicted_total = self.consumed_epsilon + avg_epsilon_per_round * remaining_rounds
            if predicted_total > self.target_epsilon * 1.1:
                return True
        return False

    def update_privacy_spent(self, round_epsilon: float):
        """
        Updates the consumed privacy budget.
        """
        self.consumed_epsilon += round_epsilon
        self.round_epsilons.append(round_epsilon)

def create_keras_model(input_shape: Tuple[int, ...]) -> tf.keras.Model:
    """
    Creates a Keras model for federated learning.
    """
    return tf.keras.Sequential([
        tf.keras.layers.Dense(
            64, activation='relu', input_shape=input_shape,
            kernel_regularizer=tf.keras.regularizers.l2(1e-4)
        ),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1, activation='linear')
    ])

def model_fn(input_shape: Tuple[int, ...]) -> tff.learning.models.FunctionalModel:
    """
    Returns a TFF functional model.
    """
    keras_model = create_keras_model(input_shape)
    return tff.learning.models.from_keras_model(
        keras_model,
        input_spec=OrderedDict([
            ('x', tf.TensorSpec(shape=[None, input_shape[0]], dtype=tf.float32)),
            ('y', tf.TensorSpec(shape=[None, 1], dtype=tf.float32))
        ]),
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=[tf.keras.metrics.MeanAbsoluteError()]
    )

def create_dp_optimizer(client_lr: float, dp_l2_norm_clip: float,
                        dp_noise_multiplier: float, dp_microbatch_size: int) -> DPKerasOptimizer:
    """
    Creates a differentially private Keras optimizer.
    """
    base_optimizer = tf.keras.optimizers.Adam(learning_rate=client_lr)
    return DPKerasOptimizer(
        base_optimizer=base_optimizer,
        l2_norm_clip=dp_l2_norm_clip,
        noise_multiplier=dp_noise_multiplier,
        num_microbatches=dp_microbatch_size
    )

def client_optimizer_fn(client_lr: float, dp_l2_norm_clip: float,
                          dp_noise_multiplier: float, dp_microbatch_size: int) -> DPKerasOptimizer:
    """
    Returns a client optimizer function for TFF.
    """
    return create_dp_optimizer(client_lr, dp_l2_norm_clip, dp_noise_multiplier, dp_microbatch_size)

def server_optimizer_fn(server_lr: float) -> tf.keras.optimizers.Optimizer:
    """
    Returns a server optimizer function for TFF.
    """
    return tf.keras.optimizers.SGD(learning_rate=server_lr)
