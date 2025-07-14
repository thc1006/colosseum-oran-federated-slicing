import pytest
import pandas as pd
import numpy as np
from src.data_preprocessing import ColoRANDataProcessor, MemoryOptimizedFeatureProcessor, optimize_datatypes, save_processed_data

@pytest.fixture
def mock_dataset(tmp_path):
    dataset_path = tmp_path / "colosseum-oran-coloran-dataset"
    dataset_path.mkdir()
    rome_path = dataset_path / "rome_static_medium"
    rome_path.mkdir()

    for sched in ['sched0', 'sched1']:
        sched_path = rome_path / sched
        sched_path.mkdir()
        tr_path = sched_path / "tr0"
        tr_path.mkdir()
        exp_path = tr_path / "exp1"
        exp_path.mkdir()
        bs_path = exp_path / "bs1"
        bs_path.mkdir()

        # Create dummy csv files
        pd.DataFrame({'col1': [1], 'col2': [2]}).to_csv(bs_path / "bs1.csv", index=False)
        pd.DataFrame({'col1': [1], 'col2': [2]}).to_csv(bs_path / "ue1.csv", index=False)

        slices_path = bs_path / "slices_bs1"
        slices_path.mkdir()
        pd.DataFrame({
            'Timestamp': [pd.Timestamp.now().timestamp() * 1000],
            'slice_id': [0],
            'sum_requested_prbs': [100],
            'sum_granted_prbs': [90],
            'tx_brate downlink [Mbps]': [50],
            'tx_errors downlink (%)': [1],
            'rx_errors uplink (%)': [2],
            'dl_cqi': [10],
            'num_ues': [5]
        }).to_csv(slices_path / "imsi1_metrics.csv", index=False)

    return str(dataset_path)

def test_colorn_data_processor_init(mock_dataset):
    processor = ColoRANDataProcessor(mock_dataset)
    assert processor.dataset_path == mock_dataset

def test_load_all_data(mock_dataset):
    processor = ColoRANDataProcessor(mock_dataset)
    bs_df, ue_df, slice_df = processor.load_all_data()
    assert not bs_df.empty
    assert not ue_df.empty
    assert not slice_df.empty
    assert 'bs_id' in bs_df.columns
    assert 'ue_id' in ue_df.columns
    assert 'imsi' in slice_df.columns

def test_memory_optimized_feature_processor_init():
    processor = MemoryOptimizedFeatureProcessor({}, batch_size=100)
    assert processor.batch_size == 100

def test_process_in_batches(mock_dataset):
    data_processor = ColoRANDataProcessor(mock_dataset)
    _, _, slice_df = data_processor.load_all_data()

    feature_processor = MemoryOptimizedFeatureProcessor(data_processor.slice_configs)
    processed_data = feature_processor.process_in_batches(slice_df)

    assert not processed_data.empty
    assert 'allocation_efficiency' in processed_data.columns
    assert processed_data['allocation_efficiency'].isnull().sum() == 0

def test_optimize_datatypes():
    df = pd.DataFrame({
        'int_col': np.random.randint(0, 200, size=100),
        'float_col': np.random.rand(100),
        'cat_col': ['A', 'B'] * 50
    })
    optimized_df = optimize_datatypes(df.copy())
    assert optimized_df['int_col'].dtype == 'uint8'
    assert optimized_df['float_col'].dtype == 'float32'
    assert optimized_df['cat_col'].dtype == 'category'

def test_save_processed_data(tmp_path):
    df = pd.DataFrame({'a': [1], 'b': [2]})
    output_path = tmp_path / "processed.parquet"
    metadata_path = tmp_path / "metadata.json"

    save_processed_data(df, ['a', 'b'], str(output_path), str(metadata_path))

    assert output_path.exists()
    assert metadata_path.exists()

    loaded_df = pd.read_parquet(output_path)
    assert df.equals(loaded_df)
