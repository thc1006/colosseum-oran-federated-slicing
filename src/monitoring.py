"""
This module provides functions for plotting training and performance metrics.
"""
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

warnings.filterwarnings('ignore')

def plot_training_history(history_df):
    """
    Plots the training history of a model, including loss, MAE, and RMSE.
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    sns.set_style("whitegrid")
    fig.suptitle('Differential Privacy Federated Learning Training History', fontsize=16)

    if 'train_rmse' not in history_df.columns:
        history_df['train_rmse'] = np.sqrt(history_df['train_loss'])
        history_df['test_rmse'] = np.sqrt(history_df['test_loss'])

    axes[0, 0].plot(history_df['round'], history_df['train_loss'], label='Training Loss', marker='o')
    axes[0, 0].plot(history_df['round'], history_df['test_loss'], label='Test Loss', marker='s')
    axes[0, 0].set_title('Loss (MSE) Curves')
    axes[0, 0].set_xlabel('Training Round')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(history_df['round'], history_df['train_mae'], label='Training MAE', marker='o')
    axes[0, 1].plot(history_df['round'], history_df['test_mae'], label='Test MAE', marker='s')
    axes[0, 1].set_title('Mean Absolute Error (MAE) Curves')
    axes[0, 1].set_xlabel('Training Round')
    axes[0, 1].set_ylabel('MAE (Scaled)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(history_df['round'], history_df['train_rmse'], label='Training RMSE', marker='o', color='blue')
    axes[1, 0].plot(history_df['round'], history_df['test_rmse'], label='Test RMSE', marker='s', color='red')
    axes[1, 0].set_title('Root Mean Square Error (RMSE) Curves')
    axes[1, 0].set_xlabel('Training Round')
    axes[1, 0].set_ylabel('RMSE (Scaled)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    if 'epsilon' in history_df.columns:
        axes[1, 1].plot(history_df['round'], history_df['epsilon'], marker='o', color='purple')
        axes[1, 1].set_title('Privacy Budget (ε) Consumption')
        axes[1, 1].set_xlabel('Training Round')
        axes[1, 1].set_ylabel('Cumulative ε')
        axes[1, 1].grid(True, alpha=0.3)
    else:
        rmse_diff = history_df['test_rmse'] - history_df['train_rmse']
        axes[1, 1].plot(history_df['round'], rmse_diff, marker='o', color='purple')
        axes[1, 1].set_title('Test-Train RMSE Gap')
        axes[1, 1].set_xlabel('Training Round')
        axes[1, 1].set_ylabel('RMSE Gap')
        axes[1, 1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def plot_client_performance(client_results_df):
    """
    Plots the final model performance comparison by client.
    """
    fig, ax = plt.subplots(figsize=(12, 7))
    x = np.arange(len(client_results_df))
    width = 0.35

    bars1 = ax.bar(x - width/2, client_results_df['mae'], width, label='MAE')
    bars2 = ax.bar(x + width/2, client_results_df['rmse'], width, label='RMSE')

    ax.set_title('Final Model Performance Comparison by Client')
    ax.set_xlabel('Client (Base Station) ID')
    ax.set_ylabel('Error (Unscaled)')
    ax.set_xticks(x)
    ax.set_xticklabels(client_results_df['client_id'])
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    ax.bar_label(bars1, fmt='%.3f')
    ax.bar_label(bars2, fmt='%.3f')

    plt.tight_layout()
    plt.show()

def plot_scenario_predictions(scenarios, predictions):
    """
    Plots resource allocation efficiency predictions under different scenarios.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(scenarios.keys(), predictions, color=['green', 'orange', 'red'], alpha=0.7)
    ax.set_title('Resource Allocation Efficiency Prediction under Different Network Load Scenarios')
    ax.set_ylabel('Predicted Resource Allocation Efficiency')
    ax.set_ylim(0, max(predictions) * 1.2 if predictions else 1.0)
    ax.bar_label(bars, fmt='%.3f')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

def plot_training_comparison(history_cen, history_df):
    """
    Compares the training progress of centralized and federated models.
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.plot(history_cen.history['val_root_mean_squared_error'], label='Centralized Validation RMSE', marker='o')
    ax.plot(history_df['test_rmse'], label='Federated Test RMSE', marker='s')
    ax.set_title('Training Progress Comparison: RMSE')
    ax.set_xlabel('Epoch/Round')
    ax.set_ylabel('RMSE (Scaled)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.show()
