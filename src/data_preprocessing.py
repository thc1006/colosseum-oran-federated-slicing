"""
This module provides classes and functions for preprocessing the ColoRAN dataset.
"""

import os
import subprocess
import logging
import json
import gc
import warnings
from datetime import datetime
import glob
import pandas as pd
import numpy as np

warnings.filterwarnings('ignore')

class ColoRANDataProcessor:
    """
    A class to process the ColoRAN dataset, including downloading and loading the data.
    """
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.base_stations = [1, 8, 15, 22, 29, 36, 43]
        self.scheduling_policies = ['sched0', 'sched1', 'sched2']
        self.training_configs = [f'tr{i}' for i in range(28)]
        self.slice_configs = {
            'tr0': [2, 13, 2], 'tr1': [4, 11, 2], 'tr2': [6, 9, 2], 'tr3': [8, 7, 2],
            'tr4': [10, 5, 2], 'tr5': [12, 3, 2], 'tr6': [14, 1, 2], 'tr7': [2, 11, 4],
            'tr8': [4, 9, 4], 'tr9': [6, 7, 4], 'tr10': [8, 5, 4], 'tr11': [10, 3, 4],
            'tr12': [12, 1, 4], 'tr13': [2, 9, 6], 'tr14': [4, 7, 6], 'tr15': [6, 5, 6],
            'tr16': [8, 3, 6], 'tr17': [10, 1, 6], 'tr18': [2, 7, 8], 'tr19': [4, 5, 8],
            'tr20': [6, 3, 8], 'tr21': [8, 1, 8], 'tr22': [2, 5, 10], 'tr23': [4, 3, 10],
            'tr24': [6, 1, 10], 'tr25': [2, 3, 12], 'tr26': [4, 1, 12], 'tr27': [2, 1, 14]
        }
        self.logger = logging.getLogger('ColoRANDataProcessor')
        self.logger.info("Initializing ColoRANDataProcessor with dataset_path: %s", self.dataset_path)

    def download_dataset(self):
        """
        Downloads the dataset from the specified git repository.
        """
        dataset_repo_url = "https://github.com/wineslab/colosseum-oran-coloran-dataset.git"
        if os.path.exists(self.dataset_path):
            self.logger.info("Dataset already exists. Skipping download.")
            return

        self.logger.info("Downloading dataset from %s to %s", dataset_repo_url, self.dataset_path)
        try:
            subprocess.run(
                ["git", "clone", dataset_repo_url, self.dataset_path],
                capture_output=True, text=True, timeout=600, check=True
            )
            self.logger.info("Dataset downloaded successfully.")
        except subprocess.TimeoutExpired:
            self.logger.error("Dataset download timed out.")
        except subprocess.CalledProcessError as e:
            self.logger.error("Failed to download dataset: %s", e.stderr)
        except Exception as e:
            self.logger.error("An error occurred during dataset download: %s", e)

    def load_all_data(self):
        """
        Loads all data from the dataset into pandas DataFrames.
        """
        base_data_path = self._auto_detect_structure()
        if not base_data_path:
            return None, None, None

        data_list = {'bs': [], 'ue': [], 'slice': []}
        for sched_policy in self.scheduling_policies:
            for training_config in self.training_configs:
                for data_type in ['bs', 'ue', 'slice']:
                    files = self._get_files(base_data_path, sched_policy, training_config, data_type)
                    for file in files:
                        try:
                            df = pd.read_csv(file)
                            metadata = self._extract_metadata(file, sched_policy, training_config, data_type)
                            df = df.assign(**metadata)
                            data_list[data_type].append(df)
                        except Exception as e:
                            self.logger.warning("Failed to load %s file %s: %s", data_type, file, e)

        bs_df = pd.concat(data_list['bs'], ignore_index=True) if data_list['bs'] else pd.DataFrame()
        ue_df = pd.concat(data_list['ue'], ignore_index=True) if data_list['ue'] else pd.DataFrame()
        slice_df = pd.concat(data_list['slice'], ignore_index=True) if data_list['slice'] else pd.DataFrame()

        return bs_df, ue_df, slice_df

    def _get_files(self, base_path, sched, config, data_type):
        patterns = {
            'bs': f"{base_path}/{sched}/{config}/exp*/bs*/bs*.csv",
            'ue': f"{base_path}/{sched}/{config}/exp*/bs*/ue*.csv",
            'slice': f"{base_path}/{sched}/{config}/exp*/bs*/slices_bs*/*_metrics.csv"
        }
        return glob.glob(patterns[data_type])

    def _extract_metadata(self, file_path, sched, config, data_type):
        parts = file_path.split('/')
        exp_folder = next((p for p in parts if p.startswith('exp')), 'exp1')
        bs_folder = next((p for p in parts if p.startswith('bs') and not p.endswith('.csv')), 'bs1')

        metadata = {
            'bs_id': int(bs_folder.replace('bs', '')),
            'exp_id': int(exp_folder.replace('exp', '')),
            'sched_policy': sched,
            'training_config': config,
            'file_path': file_path
        }

        if data_type == 'ue':
            ue_file_name = os.path.basename(file_path)
            metadata['ue_id'] = int(ue_file_name.replace('ue', '').replace('.csv', ''))
        elif data_type == 'slice':
            slice_file_name = os.path.basename(file_path)
            metadata['imsi'] = slice_file_name.replace('_metrics.csv', '')

        return metadata

    def _auto_detect_structure(self):
        possible_paths = [
            os.path.join(self.dataset_path, "rome_static_medium"),
            os.path.join(self.dataset_path, "colosseum-oran-coloran-dataset", "rome_static_medium"),
            self.dataset_path
        ]
        for path in possible_paths:
            if os.path.exists(path) and any(d.startswith('sched') for d in os.listdir(path)):
                self.logger.info("Found valid data structure at: %s", path)
                return path
        self.logger.error("Could not find standard data structure.")
        return None

class MemoryOptimizedFeatureProcessor:
    """
    Processes features in a memory-optimized way using batching.
    """
    def __init__(self, slice_configs, batch_size=100000):
        self.slice_configs = slice_configs
        self.batch_size = batch_size
        self.logger = logging.getLogger('FeatureProcessor')
        self.logger.info("Initializing MemoryOptimizedFeatureProcessor with batch size: %d", self.batch_size)

    def process_in_batches(self, slice_data):
        """
        Process the slice data in batches to optimize memory usage.
        """
        if slice_data is None or slice_data.empty:
            self.logger.error("No slice data to process.")
            return None

        total_rows = len(slice_data)
        num_batches = (total_rows + self.batch_size - 1) // self.batch_size
        processed_batches = []

        for i in range(num_batches):
            start_idx = i * self.batch_size
            end_idx = min((i + 1) * self.batch_size, total_rows)
            batch_data = slice_data.iloc[start_idx:end_idx].copy()

            processed_batch = self._process_batch(batch_data)
            if processed_batch is not None:
                processed_batches.append(processed_batch)

            gc.collect()

        return pd.concat(processed_batches, ignore_index=True) if processed_batches else None

    def _process_batch(self, df):
        try:
            if 'Timestamp' in df.columns:
                timestamps = pd.to_datetime(df['Timestamp'], unit='ms', errors='coerce')
                df['hour'] = timestamps.dt.hour
                df['minute'] = timestamps.dt.minute
                df['day_of_week'] = timestamps.dt.dayofweek

            sched_mapping = {'sched0': 0, 'sched1': 1, 'sched2': 2}
            df['sched_policy_num'] = df['sched_policy'].map(sched_mapping)

            df['allocated_rbgs'] = self._vectorized_rbg_allocation(df)

            df['sum_requested_prbs'] = df.get('sum_requested_prbs', 0).fillna(0)
            df['sum_granted_prbs'] = df.get('sum_granted_prbs', 0).fillna(0)
            df['prb_utilization'] = np.where(
                df['sum_requested_prbs'] > 0, df['sum_granted_prbs'] / df['sum_requested_prbs'], 0
            ).clip(0, 1)

            throughput_col = 'tx_brate downlink [Mbps]'
            df['throughput_efficiency'] = np.where(
                df['sum_granted_prbs'] > 0, df.get(throughput_col, 0).fillna(0) / df['sum_granted_prbs'], 0
            )

            df['qos_score'] = self._calculate_qos_score_vectorized(df)

            df['num_ues'] = df.get('num_ues', 1).fillna(1)
            df['network_load'] = df['num_ues'] / 42.0

            df['allocation_efficiency'] = (
                0.5 * df['throughput_efficiency'] + 0.3 * df['qos_score'] + 0.2 * df['prb_utilization']
            ).clip(0, 1)

            required_columns = [
                'num_ues', 'slice_id', 'sched_policy_num', 'allocated_rbgs', 'bs_id', 'exp_id',
                'sum_requested_prbs', 'sum_granted_prbs', 'prb_utilization',
                'throughput_efficiency', 'qos_score', 'network_load', 'hour', 'minute',
                'day_of_week', 'allocation_efficiency', 'sched_policy', 'training_config'
            ]

            return df[[col for col in required_columns if col in df.columns]].dropna(
                subset=['allocation_efficiency']
            )
        except Exception as e:
            self.logger.error("Error processing batch: %s", e)
            return None

    def _vectorized_rbg_allocation(self, df):
        config_map = {
            (config, slice_id): rbg_list[slice_id]
            for config, rbg_list in self.slice_configs.items()
            for slice_id in range(len(rbg_list))
        }
        keys = list(zip(df['training_config'], df.get('slice_id', 0).fillna(0)))
        return pd.Series(keys).map(config_map).fillna(0).values

    def _calculate_qos_score_vectorized(self, df):
        dl_error_col = 'tx_errors downlink (%)'
        ul_error_col = 'rx_errors uplink (%)'
        cqi_col = 'dl_cqi'

        dl_score = (100 - df.get(dl_error_col, 50).fillna(50)) / 100
        ul_score = (100 - df.get(ul_error_col, 50).fillna(50)) / 100
        cqi_score = df.get(cqi_col, 7.5).fillna(7.5) / 15

        return (0.4 * dl_score + 0.3 * ul_score + 0.3 * cqi_score).clip(0, 1)

def optimize_datatypes(df):
    """
    Optimizes the data types of a pandas DataFrame to reduce memory usage.
    """
    initial_mem = df.memory_usage(deep=True).sum() / 1024**2

    for col in df.select_dtypes(include=['int64']).columns:
        if df[col].min() >= 0:
            if df[col].max() < 255:
                df[col] = df[col].astype('uint8')
            elif df[col].max() < 65535:
                df[col] = df[col].astype('uint16')
        else:
            if df[col].min() >= -128 and df[col].max() < 127:
                df[col] = df[col].astype('int8')
            elif df[col].min() >= -32768 and df[col].max() < 32767:
                df[col] = df[col].astype('int16')

    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='float')

    for col in df.select_dtypes(include=['object']).columns:
        if df[col].nunique() / len(df) < 0.5:
            df[col] = df[col].astype('category')

    final_mem = df.memory_usage(deep=True).sum() / 1024**2
    logging.info("Memory optimized from %.2f MB to %.2f MB", initial_mem, final_mem)
    return df

def save_processed_data(processed_data, feature_names, output_filename, metadata_filename):
    """
    Saves the processed data to a parquet file and metadata to a JSON file.
    """
    if processed_data is None or processed_data.empty:
        logging.error("No processed data to save.")
        return

    optimized_data = optimize_datatypes(processed_data.copy())
    optimized_data.to_parquet(output_filename, compression='snappy', index=False, engine='pyarrow')

    metadata = {
        'feature_names': feature_names,
        'total_records': len(optimized_data),
        'processing_date': datetime.now().isoformat(),
        'file_size_mb': os.path.getsize(output_filename) / 1024**2
    }
    with open(metadata_filename, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)

    logging.info("Processed data saved to %s", output_filename)
    logging.info("Metadata saved to %s", metadata_filename)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    DATASET_PATH = "/content/colosseum-oran-coloran-dataset"

    # Step 1: Download and load raw data
    data_processor = ColoRANDataProcessor(DATASET_PATH)
    data_processor.download_dataset()
    bs_df, ue_df, slice_df = data_processor.load_all_data()

    # Step 2: Feature engineering
    if slice_df is not None:
        feature_processor = MemoryOptimizedFeatureProcessor(data_processor.slice_configs)
        processed_slice_data = feature_processor.process_in_batches(slice_df)

        # Step 3: Save processed data
        if processed_slice_data is not None:
            feature_columns = [
                'num_ues', 'slice_id', 'sched_policy_num', 'allocated_rbgs', 'bs_id', 'exp_id',
                'sum_requested_prbs', 'sum_granted_prbs', 'prb_utilization',
                'throughput_efficiency', 'qos_score', 'network_load', 'hour', 'minute',
                'day_of_week'
            ]
            available_features = [f for f in feature_columns if f in processed_slice_data.columns]

            save_processed_data(
                processed_slice_data,
                available_features,
                'coloran_processed_features.parquet',
                'feature_metadata.json'
            )
