from collections import defaultdict, deque
import os
from flask import Flask, request, jsonify
import threading
import time
from typing import Dict, List, Tuple, Any
import hydra
from omegaconf import DictConfig
import ray
from .env import VLNCEEnv, VLNDataset
from .env_config import VLNCEEnvConfig
from .service import ActorPool, VLNCEService
from .service_config import VLNCEServiceConfig

REGISTERED_ENV = {
    "vlnce": {
        "env_cls": VLNCEEnv,
        "config_cls": VLNCEEnvConfig,
        "service_cls": VLNCEService,
        "service_config_cls": VLNCEServiceConfig
    }
}

class TimeMetrics:
    def __init__(self, max_history: int = 1000, report_interval: int = 100):
        self.metrics = defaultdict(lambda: deque(maxlen=max_history))
        self.counters = defaultdict(int)
        self.report_interval = report_interval
        self.last_report_time = time.time()
    
    def record(self, operation: str, duration: float):
        self.metrics[operation].append(duration)
        self.counters[operation] += 1
        
        # Report metrics periodically
        if self.counters[operation] % self.report_interval == 0:
            self._report_metrics()
    
    def _report_metrics(self):
        current_time = time.time()
        time_since_last = current_time - self.last_report_time
        
        print(f"\n[METRICS] Time metrics report (last {time_since_last:.1f}s):")
        for operation, times in self.metrics.items():
            if times:
                avg_time = sum(times) / len(times)
                min_time = min(times)
                max_time = max(times)
                count = self.counters[operation]
                print(f"  {operation}: count={count}, avg={avg_time:.3f}s, min={min_time:.3f}s, max={max_time:.3f}s")
        
        self.last_report_time = current_time

class BatchEnvServer:
    """
    A unified server for handling batch environment operations through HTTP requests.
    Uses environment services to handle operations and properly handle serialization.
    Exposes only the standard BaseService interface.
    """
    
    def __init__(self, config, r2r_actor_pool, rxr_actor_pool):
        """
        Initialize the BatchEnvServer.
        
        Args:
            host: Host address for the server
            port: Port to listen on
            debug: Whether to run Flask in debug mode
        """
        self.host = config.server.host
        self.port = config.server.port
        self.debug = config.server.debug
        self.config = config
        
        # Create Flask app
        self.app = Flask(__name__)
        self._setup_routes()
        
        # Server state
        self.is_running = False
        self.server_thread = None
        
        self.r2r_actor_pool = r2r_actor_pool
        self.rxr_actor_pool = rxr_actor_pool
        self.service = VLNCEService.remote(
            VLNCEServiceConfig(
                save_video_dir=self.config.get("vlnce").get('save_video_dir')
            ), 
            self.r2r_actor_pool, 
            self.rxr_actor_pool
        )
        self.metrics = TimeMetrics(max_history=1000, report_interval=2)

        
    
    def _setup_routes(self):
        """Set up HTTP routes for the Flask app"""
        
        # @self.app.route('/health', methods=['GET'])
        # def health_check():
        #     """Health check endpoint"""
        #     return jsonify({
        #         "status": "ok",
        #         "message": "Environment server is running",
        #         "registered_envs": list(REGISTERED_ENV.keys()),
        #         "active_services": list(self.services.keys()),
        #         "active_environments": len(self.env_to_service)
        #     }), 200
            
        @self.app.route('/environments', methods=['POST'])
        def create_environments_batch():
            """Create environments endpoint - implements BaseService interface"""
            data = request.json
            if not data or 'ids2configs' not in data:
                return jsonify({"error": "Missing required parameter: ids2configs"}), 400
                    
            ids2configs = data['ids2configs']
            self._create_environments_batch(ids2configs)
            return jsonify({"success": True}), 200
        
        @self.app.route('/batch/reset', methods=['POST'])
        def reset_batch():
            """Reset multiple environments endpoint"""
            data = request.json
            if not data or 'ids2seeds' not in data:
                return jsonify({"error": "Missing required parameter: ids2seeds"}), 400
                
            ids2seeds = data['ids2seeds']
            results = self._reset_batch(ids2seeds)
            return jsonify({"results": results}), 200
                
        @self.app.route('/batch/step', methods=['POST'])
        def step_batch():
            """Step multiple environments endpoint"""
            data = request.json
            if not data or 'ids2actions' not in data:
                return jsonify({"error": "Missing required parameter: ids2actions"}), 400
                
            ids2actions = data['ids2actions']
            results = self._step_batch(ids2actions)
            return jsonify({"results": results}), 200
                
                
        @self.app.route('/batch/close', methods=['POST'])
        def close_batch():
            """Close multiple environments endpoint"""
            data = request.json
            if not data or 'env_ids' not in data:
                return jsonify({"error": "Missing required parameter: env_ids"}), 400
                
            env_ids = data['env_ids']
            self._close_batch(env_ids)
            return jsonify({"status": "success"}), 200
        
        @self.app.route('/batch/warmup', methods=['POST'])
        def warmup_batch():
            data = request.json
            if not data or 'cfgs' not in data:
                return jsonify({"error": "Missing required parameter: cfgs"}), 400
                    
            cfgs = data['cfgs']
            self._warmup_batch(cfgs)
            return jsonify({"success": True}), 200
        
    
    def _create_environments_batch(self, ids2configs: Dict[Any, Any]) -> None:
        """
        Create multiple environments in batch.
        Implements BaseService.create_environments_batch.
        
        Args:
            ids2configs: Dictionary mapping environment IDs to their configurations
        """
        start_time = time.time()
        ray.get(self.service.create_environments_batch.remote(ids2configs))
        self.metrics.record('create_batch', time.time() - start_time)
    
    
    def _reset_batch(self, ids2seeds: Dict[str, Any]) -> Dict[str, Tuple[Any, Any]]:
        """
        Reset multiple environments.
        
        Args:
            ids2seeds: Dictionary mapping environment IDs to seeds
            
        Returns:
            Dictionary mapping environment IDs to (observation, info) tuples
        """
        start_time = time.time()
        results = ray.get(self.service.reset_batch.remote(ids2seeds))
        results = {
            env_id: ray.get(future) for env_id, future in results.items()
        }
        self.metrics.record('reset_batch', time.time() - start_time)
        
        return results
    
    def _step_batch(self, ids2actions: Dict[str, Any]) -> Dict[str, Tuple[Dict, float, bool, Dict]]:
        """
        Step multiple environments.
        
        Args:
            ids2actions: Dictionary mapping environment IDs to actions
            
        Returns:
            Dictionary mapping environment IDs to (observation, reward, done, info) tuples
        """
        start_time = time.time()
        results = ray.get(self.service.step_batch.remote(ids2actions))
        results = {
            env_id: ray.get(future) for env_id, future in results.items()
        }
        self.metrics.record('step_batch', time.time() - start_time)
        return results
    
    def _close_batch(self, env_ids: List[str]) -> None:
        """
        Close multiple environments.
        
        Args:
            env_ids: List of environment IDs
        """
        time_start = time.time()
        ray.get(self.service.close_batch.remote(env_ids))
        self.metrics.record('close_batch', time.time() - time_start)
            
    def _warmup_batch(self, cfgs) -> None:
        time_start = time.time()
        ray.get(self.service.warmup_batch.remote(cfgs))
        self.metrics.record('warmup_batch', time.time() - time_start)
    
    def start(self, background: bool = True) -> None:
        """
        Start the server.
        
        Args:
            background: Whether to run the server in a background thread
        """
        if self.is_running:
            print("Server is already running")
            return
            
        if background:
            self.server_thread = threading.Thread(target=self._run_server)
            self.server_thread.daemon = True
            self.server_thread.start()
            self.is_running = True
            
            # Wait for server to start
            max_retries = 5
            retry_delay = 0.5
            for _ in range(max_retries):
                time.sleep(retry_delay)
                import requests
                response = requests.get(f"http://{self.host}:{self.port}/health", timeout=1)
                if response.status_code == 200:
                    print(f"Server started on http://{self.host}:{self.port}")
                    break
            else:
                print("Server may not have started properly")
        else:
            self.is_running = True
            self._run_server()
    
    def _run_server(self) -> None:
        """Run the Flask server"""
        self.app.run(host=self.host, port=self.port, debug=self.debug, use_reloader=False)

@hydra.main(version_base=None, config_path="config", config_name="server")
def main(cfg: DictConfig):
    
    print(cfg)
    r2r_gpu_plan = cfg.vlnce.r2r_gpu_plan
    rxr_gpu_plan = cfg.vlnce.rxr_gpu_plan
    r2r_config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'VLN_CE', 'vlnce_baselines', 'config', 'r2r_baselines', 'activevln_r2r.yaml')
    rxr_config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'VLN_CE', 'vlnce_baselines', 'config', 'rxr_baselines', 'activevln_rxr.yaml')
    
    avail_gpus = cfg.vlnce.gpus
    assert len(avail_gpus) == len(r2r_gpu_plan) == len(rxr_gpu_plan)
    logical_to_physical = {i: avail_gpus[i] for i in range(len(avail_gpus))}

    resources = {
        **{f"r2r_gpu{i}": count for i, count in enumerate(r2r_gpu_plan)},
        **{f"rxr_gpu{i}": count for i, count in enumerate(rxr_gpu_plan)},
    }
    ray.init(resources=resources)

    r2r_dataset = VLNDataset.remote(r2r_config_path)
    rxr_dataset = VLNDataset.remote(rxr_config_path)
    
    r2r_actor_pool = ActorPool.remote(logical_to_physical, r2r_gpu_plan, resource_prefix="r2r")
    rxr_actor_pool = ActorPool.remote(logical_to_physical, rxr_gpu_plan, resource_prefix="rxr")
    ray.get([
        r2r_actor_pool.init_pool.remote(r2r_dataset),
        rxr_actor_pool.init_pool.remote(rxr_dataset),
    ])

    server = BatchEnvServer(cfg, r2r_actor_pool, rxr_actor_pool)
    print(f"Starting Batch Environment Server on http://{cfg.server.host}:{cfg.server.port}")
    server.start(background=False)


if __name__ == "__main__":
    main()
