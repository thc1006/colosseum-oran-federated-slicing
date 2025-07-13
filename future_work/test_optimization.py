# tests/test_optimization.py
import pytest
import numpy as np
from unittest.mock import MagicMock
from src.optimization.allocator import SliceResourceAllocator

@pytest.fixture
def mock_predictor():
    predictor = MagicMock()
    # Mock to return efficiency equal to the sum of allocated RBGs
    # divided by a constant, to have a predictable outcome.
    def mock_predict(feature_matrix):
        return np.clip(feature_matrix[:, 3] / 20.0, 0, 1) # allocated_rbgs is 4th feature
    predictor.predict.side_effect = mock_predict
    return predictor

@pytest.fixture
def mock_allocator(mock_predictor):
    # The allocator requires the predictor object, not the class
    allocator_obj = SliceResourceAllocator.__new__(SliceResourceAllocator)
    allocator_obj.model = mock_predictor
    allocator_obj.scaler = MagicMock()
    allocator_obj.features = [f'f{i}' for i in range(15)]
    allocator_obj.total_rbgs = 17
    return allocator_obj

def test_exhaustive_optimizer(mock_allocator):
    from src.optimization.exhaustive_search import ExhaustiveOptimizer
    
    def evaluate(state, allocations):
        matrix = mock_allocator._get_feature_matrix(state, allocations)
        preds = mock_allocator._predict_efficiency(matrix)
        return preds.reshape(len(allocations), 3).mean(axis=1)

    optimizer = ExhaustiveOptimizer(evaluate, total_rbgs=17)
    state = {'sum_requested_prbs': 10, 'num_ues': 10} # dummy state
    
    best_alloc, best_efficiency = optimizer.run(state)
    
    assert sum(best_alloc) == 17
    # With our mock, higher RBG sum should be better, but the mean balances it.
    # The key is that it returns a valid allocation.
    assert all(x >= 1 for x in best_alloc)

