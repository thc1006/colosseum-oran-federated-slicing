# 差分隱私聯邦學習生產環境部署與監控系統

import tensorflow as tf
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json
import os
import logging
from typing import Dict, List, Optional, Tuple
import asyncio
from concurrent.futures import ThreadPoolExecutor
import hashlib
import pickle

# ======================== 生產環境部署管理器 ========================

class DPFLProductionDeployment:
    """
    差分隱私聯邦學習生產環境部署管理器
    """
    
    def __init__(self, model_path: str, config_path: str, 
                 log_dir: str = "./production_logs"):
        self.model_path = model_path
        self.config_path = config_path
        self.log_dir = log_dir
        
        # 初始化日誌
        self._setup_logging()
        
        # 載入模型和配置
        self.model = self._load_model()
        self.config = self._load_config()
        self.scalers = self._load_scalers()
        
        # 性能監控
        self.performance_monitor = PerformanceMonitor()
        
        # 隱私預算追蹤器
        self.privacy_tracker = ProductionPrivacyTracker(
            initial_budget=self.config.get('remaining_privacy_budget', 10.0)
        )
        
    def _setup_logging(self):
        """設置生產環境日誌"""
        os.makedirs(self.log_dir, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f"{self.log_dir}/dpfl_production.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('DPFLProduction')
        
    def _load_model(self) -> tf.keras.Model:
        """載入訓練好的模型"""
        try:
            model = tf.keras.models.load_model(self.model_path)
            self.logger.info(f"模型載入成功: {self.model_path}")
            return model
        except Exception as e:
            self.logger.error(f"模型載入失敗: {e}")
            raise
            
    def _load_config(self) -> Dict:
        """載入配置"""
        try:
            # 嘗試載入元數據
            metadata_path = self.model_path.replace('.keras', '_metadata.json')
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    config = json.load(f)
            else:
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
            return config
        except Exception as e:
            self.logger.error(f"配置載入失敗: {e}")
            return {}
            
    def _load_scalers(self) -> Dict:
        """載入縮放器"""
        try:
            artifacts_path = self.config_path.replace('.json', '_artifacts.pkl')
            with open(artifacts_path, 'rb') as f:
                artifacts = pickle.load(f)
            return {
                'feature_scaler': artifacts['feature_scaler'],
                'target_scaler': artifacts['target_scaler']
            }
        except Exception as e:
            self.logger.error(f"縮放器載入失敗: {e}")
            return {}
            
    async def predict_async(self, features: np.ndarray) -> np.ndarray:
        """異步預測"""
        loop = asyncio.get_event_loop()
        
        # 在線程池中執行預測
        with ThreadPoolExecutor() as executor:
            # 縮放輸入
            scaled_features = await loop.run_in_executor(
                executor,
                self.scalers['feature_scaler'].transform,
                features
            )
            
            # 預測
            scaled_predictions = await loop.run_in_executor(
                executor,
                self.model.predict,
                scaled_features,
                {'verbose': 0}
            )
            
            # 反縮放
            predictions = await loop.run_in_executor(
                executor,
                self.scalers['target_scaler'].inverse_transform,
                scaled_predictions
            )
            
        return predictions
        
    def predict_batch(self, features_list: List[np.ndarray],
                     add_noise: bool = True) -> List[float]:
        """
        批量預測with隱私保護
        """
        predictions = []
        
        for features in features_list:
            try:
                # 性能監控
                start_time = datetime.now()
                
                # 縮放
                scaled_features = self.scalers['feature_scaler'].transform(
                    features.reshape(1, -1)
                )
                
                # 預測
                scaled_pred = self.model.predict(scaled_features, verbose=0)
                
                # 反縮放
                pred = self.scalers['target_scaler'].inverse_transform(scaled_pred)
                
                # 添加隱私噪音（生產環境）
                if add_noise:
                    noise_scale = 0.01  # 小量噪音保護輸出隱私
                    noise = np.random.laplace(0, noise_scale)
                    pred = pred + noise
                    
                predictions.append(float(pred[0, 0]))
                
                # 記錄性能
                inference_time = (datetime.now() - start_time).total_seconds()
                self.performance_monitor.record_inference(inference_time)
                
            except Exception as e:
                self.logger.error(f"預測錯誤: {e}")
                predictions.append(None)
                
        return predictions
        
    def update_model_incremental(self, new_model_path: str) -> bool:
        """
        增量更新模型（保持隱私預算）
        """
        try:
            # 驗證新模型
            new_model = tf.keras.models.load_model(new_model_path)
            
            # 檢查模型架構一致性
            if not self._validate_model_architecture(new_model):
                raise ValueError("模型架構不一致")
                
            # 備份當前模型
            backup_path = f"{self.model_path}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self.model.save(backup_path)
            
            # 更新模型
            self.model = new_model
            self.logger.info(f"模型更新成功，備份保存至: {backup_path}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"模型更新失敗: {e}")
            return False
            
    def _validate_model_architecture(self, new_model: tf.keras.Model) -> bool:
        """驗證模型架構一致性"""
        return (new_model.input_shape == self.model.input_shape and
                new_model.output_shape == self.model.output_shape)

# ======================== 性能監控器 ========================

class PerformanceMonitor:
    """
    生產環境性能監控
    """
    
    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self.inference_times = []
        self.prediction_values = []
        self.timestamps = []
        
    def record_inference(self, inference_time: float, 
                        prediction: Optional[float] = None):
        """記錄推論性能"""
        self.inference_times.append(inference_time)
        self.timestamps.append(datetime.now())
        
        if prediction is not None:
            self.prediction_values.append(prediction)
            
        # 保持窗口大小
        if len(self.inference_times) > self.window_size:
            self.inference_times.pop(0)
            self.timestamps.pop(0)
            if self.prediction_values:
                self.prediction_values.pop(0)
                
    def get_performance_stats(self) -> Dict[str, float]:
        """獲取性能統計"""
        if not self.inference_times:
            return {}
            
        return {
            'avg_inference_time': np.mean(self.inference_times),
            'p50_inference_time': np.percentile(self.inference_times, 50),
            'p95_inference_time': np.percentile(self.inference_times, 95),
            'p99_inference_time': np.percentile(self.inference_times, 99),
            'throughput': len(self.inference_times) / sum(self.inference_times),
            'total_predictions': len(self.inference_times)
        }
        
    def check_performance_degradation(self, threshold: float = 2.0) -> bool:
        """檢查性能是否下降"""
        if len(self.inference_times) < 100:
            return False
            
        recent_avg = np.mean(self.inference_times[-100:])
        overall_avg = np.mean(self.inference_times)
        
        return recent_avg > overall_avg * threshold

# ======================== 生產環境隱私追蹤器 ========================

class ProductionPrivacyTracker:
    """
    生產環境隱私預算追蹤
    """
    
    def __init__(self, initial_budget: float):
        self.total_budget = initial_budget
        self.consumed_budget = 0.0
        self.query_history = []
        
    def can_process_query(self, estimated_cost: float) -> bool:
        """檢查是否可以處理查詢"""
        return self.consumed_budget + estimated_cost <= self.total_budget
        
    def record_query(self, query_type: str, privacy_cost: float):
        """記錄查詢和隱私消耗"""
        self.consumed_budget += privacy_cost
        self.query_history.append({
            'timestamp': datetime.now(),
            'query_type': query_type,
            'privacy_cost': privacy_cost,
            'cumulative_cost': self.consumed_budget
        })
        
    def get_remaining_budget(self) -> float:
        """獲取剩餘隱私預算"""
        return self.total_budget - self.consumed_budget
        
    def get_budget_report(self) -> Dict:
        """生成隱私預算報告"""
        return {
            'total_budget': self.total_budget,
            'consumed_budget': self.consumed_budget,
            'remaining_budget': self.get_remaining_budget(),
            'utilization_percent': (self.consumed_budget / self.total_budget) * 100,
            'total_queries': len(self.query_history),
            'avg_cost_per_query': self.consumed_budget / len(self.query_history) if self.query_history else 0
        }

# ======================== 聯邦學習協調器 ========================

class FederatedCoordinator:
    """
    生產環境聯邦學習協調器
    """
    
    def __init__(self, num_clients: int, min_clients: int = 3):
        self.num_clients = num_clients
        self.min_clients = min_clients
        self.client_registry = {}
        self.training_rounds = []
        
    def register_client(self, client_id: str, client_info: Dict) -> bool:
        """註冊客戶端"""
        try:
            self.client_registry[client_id] = {
                'info': client_info,
                'last_seen': datetime.now(),
                'participation_count': 0,
                'average_data_size': client_info.get('data_size', 0)
            }
            return True
        except Exception as e:
            logging.error(f"客戶端註冊失敗: {e}")
            return False
            
    def select_clients(self, num_to_select: int) -> List[str]:
        """選擇參與訓練的客戶端"""
        # 過濾活躍客戶端
        active_clients = [
            client_id for client_id, info in self.client_registry.items()
            if (datetime.now() - info['last_seen']).seconds < 300  # 5分鐘內活躍
        ]
        
        if len(active_clients) < self.min_clients:
            logging.warning(f"活躍客戶端不足: {len(active_clients)} < {self.min_clients}")
            return []
            
        # 基於數據量的加權採樣
        weights = [
            self.client_registry[cid]['average_data_size'] 
            for cid in active_clients
        ]
        
        if sum(weights) > 0:
            probs = np.array(weights) / sum(weights)
        else:
            probs = None
            
        selected = np.random.choice(
            active_clients,
            size=min(num_to_select, len(active_clients)),
            replace=False,
            p=probs
        )
        
        return selected.tolist()
        
    def aggregate_updates(self, client_updates: Dict[str, np.ndarray]) -> np.ndarray:
        """聚合客戶端更新"""
        # 獲取客戶端權重
        weights = []
        updates = []
        
        for client_id, update in client_updates.items():
            client_data_size = self.client_registry[client_id]['average_data_size']
            weights.append(client_data_size)
            updates.append(update)
            
        # 加權平均
        weights = np.array(weights) / sum(weights)
        aggregated = sum(w * u for w, u in zip(weights, updates))
        
        return aggregated

# ======================== 安全通信管理器 ========================

class SecureCommunicationManager:
    """
    安全通信管理器
    """
    
    def __init__(self, encryption_key: Optional[bytes] = None):
        self.encryption_key = encryption_key or os.urandom(32)
        
    def encrypt_model_update(self, update: np.ndarray) -> bytes:
        """加密模型更新"""
        # 簡化示例：實際應使用適當的加密庫
        update_bytes = pickle.dumps(update)
        
        # 添加雜湊驗證
        hash_value = hashlib.sha256(update_bytes).digest()
        
        # 組合數據
        encrypted_data = hash_value + update_bytes
        
        return encrypted_data
        
    def decrypt_model_update(self, encrypted_data: bytes) -> np.ndarray:
        """解密模型更新"""
        # 提取雜湊和數據
        hash_value = encrypted_data[:32]
        update_bytes = encrypted_data[32:]
        
        # 驗證完整性
        if hashlib.sha256(update_bytes).digest() != hash_value:
            raise ValueError("數據完整性驗證失敗")
            
        # 解密
        update = pickle.loads(update_bytes)
        
        return update

# ======================== 自動化運維系統 ========================

class AutomatedOpsManager:
    """
    自動化運維管理器
    """
    
    def __init__(self, deployment: DPFLProductionDeployment):
        self.deployment = deployment
        self.health_checks = []
        self.alert_thresholds = {
            'inference_time_p95': 0.1,  # 100ms
            'privacy_budget_remaining': 0.1,  # 10%
            'error_rate': 0.05  # 5%
        }
        
    async def health_check(self) -> Dict[str, any]:
        """執行健康檢查"""
        health_status = {
            'timestamp': datetime.now().isoformat(),
            'model_status': 'healthy',
            'performance_status': 'healthy',
            'privacy_status': 'healthy',
            'issues': []
        }
        
        # 檢查模型
        try:
            test_input = np.random.randn(1, 13)
            _ = await self.deployment.predict_async(test_input)
        except Exception as e:
            health_status['model_status'] = 'unhealthy'
            health_status['issues'].append(f"模型預測失敗: {e}")
            
        # 檢查性能
        perf_stats = self.deployment.performance_monitor.get_performance_stats()
        if perf_stats.get('p95_inference_time', 0) > self.alert_thresholds['inference_time_p95']:
            health_status['performance_status'] = 'degraded'
            health_status['issues'].append("推論延遲過高")
            
        # 檢查隱私預算
        privacy_report = self.deployment.privacy_tracker.get_budget_report()
        remaining_percent = privacy_report['remaining_budget'] / privacy_report['total_budget']
        if remaining_percent < self.alert_thresholds['privacy_budget_remaining']:
            health_status['privacy_status'] = 'critical'
            health_status['issues'].append("隱私預算即將耗盡")
            
        self.health_checks.append(health_status)
        
        return health_status
        
    def auto_scale_decision(self) -> Dict[str, any]:
        """自動擴縮容決策"""
        perf_stats = self.deployment.performance_monitor.get_performance_stats()
        
        # 基於性能指標的擴縮容建議
        if perf_stats.get('p95_inference_time', 0) > 0.2:  # 200ms
            return {
                'action': 'scale_up',
                'reason': '推論延遲過高',
                'recommended_instances': 2
            }
        elif perf_stats.get('throughput', float('inf')) < 10:  # 10 req/s
            return {
                'action': 'scale_up',
                'reason': '吞吐量不足',
                'recommended_instances': 1
            }
        elif perf_stats.get('avg_inference_time', 0) < 0.01:  # 10ms
            return {
                'action': 'scale_down',
                'reason': '資源利用率低',
                'recommended_instances': -1
            }
        else:
            return {
                'action': 'maintain',
                'reason': '性能正常'
            }

# ======================== 部署腳本範例 ========================

async def deploy_production_system():
    """
    部署生產系統範例
    """
    # 初始化部署
    deployment = DPFLProductionDeployment(
        model_path='federated_coloran_model_dp.keras',
        config_path='dpfl_config.json'
    )
    
    # 初始化協調器
    coordinator = FederatedCoordinator(num_clients=7)
    
    # 初始化運維管理器
    ops_manager = AutomatedOpsManager(deployment)
    
    # 啟動健康檢查循環
    async def health_check_loop():
        while True:
            health_status = await ops_manager.health_check()
            if health_status['issues']:
                logging.warning(f"健康檢查發現問題: {health_status['issues']}")
            await asyncio.sleep(60)  # 每分鐘檢查
            
    # 啟動自動擴縮容循環
    async def auto_scale_loop():
        while True:
            scale_decision = ops_manager.auto_scale_decision()
            if scale_decision['action'] != 'maintain':
                logging.info(f"擴縮容建議: {scale_decision}")
            await asyncio.sleep(300)  # 每5分鐘評估
            
    # 並行執行
    await asyncio.gather(
        health_check_loop(),
        auto_scale_loop()
    )

# ======================== 監控儀表板 ========================

def create_monitoring_dashboard(deployment: DPFLProductionDeployment):
    """
    創建監控儀表板
    """
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('DP-FL Production Monitoring Dashboard', fontsize=16)
    
    def update_dashboard(frame):
        # 清空圖表
        for ax in axes.flat:
            ax.clear()
            
        # 1. 推論延遲
        perf_stats = deployment.performance_monitor.get_performance_stats()
        if deployment.performance_monitor.inference_times:
            axes[0, 0].hist(deployment.performance_monitor.inference_times[-100:], 
                          bins=20, alpha=0.7, color='blue')
            axes[0, 0].axvline(perf_stats.get('p95_inference_time', 0), 
                              color='red', linestyle='--', label='P95')
            axes[0, 0].set_xlabel('Inference Time (s)')
            axes[0, 0].set_ylabel('Frequency')
            axes[0, 0].set_title('Inference Latency Distribution')
            axes[0, 0].legend()
            
        # 2. 吞吐量趨勢
        if deployment.performance_monitor.timestamps:
            # 計算每分鐘吞吐量
            current_time = datetime.now()
            one_minute_ago = current_time - timedelta(minutes=1)
            recent_timestamps = [
                ts for ts in deployment.performance_monitor.timestamps 
                if ts > one_minute_ago
            ]
            throughput = len(recent_timestamps) / 60.0
            
            axes[0, 1].text(0.5, 0.5, f'{throughput:.1f}\nrequests/sec',
                          ha='center', va='center', fontsize=24)
            axes[0, 1].set_title('Current Throughput')
            axes[0, 1].axis('off')
            
        # 3. 隱私預算使用
        privacy_report = deployment.privacy_tracker.get_budget_report()
        consumed = privacy_report['consumed_budget']
        remaining = privacy_report['remaining_budget']
        
        axes[1, 0].pie([consumed, remaining], 
                      labels=['Consumed', 'Remaining'],
                      colors=['red', 'green'],
                      autopct='%1.1f%%',
                      startangle=90)
        axes[1, 0].set_title('Privacy Budget Status')
        
        # 4. 系統健康狀態
        ax = axes[1, 1]
        health_indicators = {
            'Model': 'green',
            'Performance': 'green' if perf_stats.get('p95_inference_time', 0) < 0.1 else 'yellow',
            'Privacy': 'green' if remaining > 0.2 * privacy_report['total_budget'] else 'red',
            'System': 'green'
        }
        
        y_pos = np.arange(len(health_indicators))
        colors = list(health_indicators.values())
        
        ax.barh(y_pos, [1]*len(health_indicators), color=colors, alpha=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(health_indicators.keys())
        ax.set_xlim(0, 1)
        ax.set_title('System Health Status')
        ax.set_xticks([])
        
        plt.tight_layout()
        
    # 動畫更新
    ani = FuncAnimation(fig, update_dashboard, interval=5000)  # 每5秒更新
    plt.show()

# ======================== 主函數 ========================

if __name__ == "__main__":
    # 生產環境部署示例
    print("🚀 啟動差分隱私聯邦學習生產系統...")
    
    # 運行異步部署
    # asyncio.run(deploy_production_system())
    
    # 或者創建監控儀表板
    # deployment = DPFLProductionDeployment(...)
    # create_monitoring_dashboard(deployment)