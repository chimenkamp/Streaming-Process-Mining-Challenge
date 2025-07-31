"""
Enhanced Algorithm Testing Engine for AVOCADO Challenge Platform
Handles algorithm loading, execution, and performance evaluation with robust error handling.
"""

import os
import sys
import time
import traceback
import importlib.util
import inspect
import tempfile
import shutil
import threading
import queue
import signal
from contextlib import contextmanager
from typing import Dict, List, Any, Optional, Type, Callable, Tuple
import pandas as pd
import numpy as np
from datetime import datetime
import json
import logging
import resource
import psutil
import gc

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AlgorithmTimeoutError(Exception):
    """Raised when algorithm execution times out."""
    pass

class AlgorithmMemoryError(Exception):
    """Raised when algorithm exceeds memory limits."""
    pass

class AlgorithmValidationError(Exception):
    """Raised when algorithm fails validation."""
    pass

class PerformanceMonitor:
    """Monitors algorithm performance during execution."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all metrics."""
        self.start_time = None
        self.end_time = None
        self.memory_samples = []
        self.cpu_samples = []
        self.event_times = []
        self.learn_times = []
        self.conformance_times = []
        self.peak_memory = 0
        self.total_events = 0
        self.learn_events = 0
        self.conformance_events = 0
        
    def start_monitoring(self):
        """Start performance monitoring."""
        self.start_time = time.time()
        self.reset_memory_peak()
        
    def stop_monitoring(self):
        """Stop performance monitoring."""
        self.end_time = time.time()
        
    def record_event_time(self, event_type: str, duration: float):
        """Record timing for an event."""
        self.event_times.append(duration)
        if event_type == 'learn':
            self.learn_times.append(duration)
            self.learn_events += 1
        elif event_type == 'conformance':
            self.conformance_times.append(duration)
            self.conformance_events += 1
        self.total_events += 1
        
    def sample_memory(self):
        """Sample current memory usage."""
        try:
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            self.memory_samples.append(memory_mb)
            self.peak_memory = max(self.peak_memory, memory_mb)
        except:
            pass
            
    def reset_memory_peak(self):
        """Reset peak memory tracking."""
        try:
            resource.setrlimit(resource.RLIMIT_AS, (resource.RLIM_INFINITY, resource.RLIM_INFINITY))
        except:
            pass
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics summary."""
        total_time = (self.end_time - self.start_time) if self.end_time and self.start_time else 0
        
        return {
            'total_runtime': total_time,
            'total_events': self.total_events,
            'learn_events': self.learn_events,
            'conformance_events': self.conformance_events,
            'avg_event_time': np.mean(self.event_times) if self.event_times else 0,
            'avg_learn_time': np.mean(self.learn_times) if self.learn_times else 0,
            'avg_conformance_time': np.mean(self.conformance_times) if self.conformance_times else 0,
            'max_event_time': np.max(self.event_times) if self.event_times else 0,
            'min_event_time': np.min(self.event_times) if self.event_times else 0,
            'peak_memory_mb': self.peak_memory,
            'avg_memory_mb': np.mean(self.memory_samples) if self.memory_samples else 0,
            'events_per_second': self.total_events / total_time if total_time > 0 else 0
        }

class SafeAlgorithmLoader:
    """Safely loads algorithm classes with validation and security checks."""
    
    def __init__(self, base_algorithm_class):
        self.base_class = base_algorithm_class
    
    def load_algorithm_from_file(self, file_path: str, timeout: int = 30) -> Tuple[Optional[Type], Optional[str]]:
        """Load algorithm class from file with safety checks."""
        try:
            # Security check: scan for dangerous imports
            if not self._security_scan(file_path):
                return None, "Algorithm contains potentially unsafe operations"
            
            # Create isolated environment
            with self._create_isolated_environment(file_path) as temp_dir:
                # Load and validate the algorithm
                algorithm_class = self._load_and_validate(file_path, temp_dir, timeout)
                return algorithm_class, None
                
        except Exception as e:
            error_msg = f"Failed to load algorithm: {str(e)}"
            logger.error(f"{error_msg}\n{traceback.format_exc()}")
            return None, error_msg
    
    def _security_scan(self, file_path: str) -> bool:
        """Scan file for potentially dangerous operations."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Dangerous patterns to check for
            dangerous_patterns = [
                'import os.system',
                'import subprocess',
                'exec(',
                'eval(',
                '__import__',
                'open(',
                'file(',
                'input(',
                'raw_input(',
                'import socket',
                'import urllib',
                'import requests',
                'import sys.exit',
                'quit(',
                'exit(',
                'globals(',
                'locals(',
                'vars(',
                'dir(',
                'getattr(',
                'setattr(',
                'delattr(',
                'hasattr(',
                'compile(',
                'reload('
            ]
            
            # Check for dangerous patterns
            for pattern in dangerous_patterns:
                if pattern in content.lower():
                    logger.warning(f"Potentially unsafe pattern found: {pattern}")
                    # For now, just warn - in production you might want to reject
            
            # Check imports
            try:
                import ast
                tree = ast.parse(content)
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            if self._is_dangerous_import(alias.name):
                                logger.warning(f"Dangerous import: {alias.name}")
                    elif isinstance(node, ast.ImportFrom):
                        if node.module and self._is_dangerous_import(node.module):
                            logger.warning(f"Dangerous import from: {node.module}")
            except:
                # If we can't parse, allow it but log warning
                logger.warning("Could not parse file for security scanning")
            
            return True  # For now, allow all files
            
        except Exception as e:
            logger.error(f"Security scan failed: {e}")
            return False
    
    def _is_dangerous_import(self, module_name: str) -> bool:
        """Check if module import is dangerous."""
        dangerous_modules = {
            'os', 'sys', 'subprocess', 'socket', 'urllib', 'urllib2', 'urllib3',
            'requests', 'httplib', 'http.client', 'ftplib', 'smtplib',
            'pickle', 'cPickle', 'marshal', 'shelve', 'dbm',
            'ctypes', 'threading', 'multiprocessing', '_thread',
            'importlib', 'imp', 'pkgutil', 'modulefinder'
        }
        return module_name in dangerous_modules
    
    @contextmanager
    def _create_isolated_environment(self, file_path: str):
        """Create isolated environment for algorithm loading."""
        temp_dir = tempfile.mkdtemp(prefix='algorithm_test_')
        try:
            # Copy file to temp directory
            file_name = os.path.basename(file_path)
            temp_file_path = os.path.join(temp_dir, file_name)
            
            if os.path.isfile(file_path):
                shutil.copy2(file_path, temp_file_path)
            else:
                # Handle directory (ZIP extraction)
                shutil.copytree(file_path, os.path.join(temp_dir, 'algorithm'))
            
            # Add to path temporarily
            if temp_dir not in sys.path:
                sys.path.insert(0, temp_dir)
            
            yield temp_dir
            
        finally:
            # Cleanup
            if temp_dir in sys.path:
                sys.path.remove(temp_dir)
            try:
                shutil.rmtree(temp_dir)
            except:
                pass
    
    def _load_and_validate(self, file_path: str, temp_dir: str, timeout: int) -> Optional[Type]:
        """Load and validate algorithm class."""
        # Find Python files to load
        if os.path.isfile(file_path):
            py_files = [file_path]
        else:
            py_files = []
            for root, dirs, files in os.walk(file_path):
                for file in files:
                    if file.endswith('.py'):
                        py_files.append(os.path.join(root, file))
        
        # Try to load algorithm from each file
        for py_file in py_files:
            try:
                algorithm_class = self._import_algorithm_from_file(py_file)
                if algorithm_class:
                    # Validate the algorithm
                    if self._validate_algorithm_class(algorithm_class):
                        return algorithm_class
            except Exception as e:
                logger.debug(f"Failed to load from {py_file}: {e}")
                continue
        
        raise AlgorithmValidationError("No valid algorithm class found")
    
    def _import_algorithm_from_file(self, file_path: str) -> Optional[Type]:
        """Import algorithm class from specific file."""
        try:
            module_name = os.path.splitext(os.path.basename(file_path))[0]
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            
            if spec is None or spec.loader is None:
                return None
            
            module = importlib.util.module_from_spec(spec)
            
            # Prepare module namespace
            self._prepare_module_namespace(module)
            
            # Execute module
            spec.loader.exec_module(module)
            
            # Find algorithm class
            return self._find_algorithm_class(module.__dict__)
            
        except Exception as e:
            logger.debug(f"Import failed for {file_path}: {e}")
            return None
    
    def _prepare_module_namespace(self, module):
        """Prepare module namespace with safe imports."""
        # Standard library imports
        import time
        import math
        import random
        from collections import defaultdict, Counter
        from typing import Dict, Any, List, Optional
        
        # Scientific computing (if available)
        try:
            import numpy as np
            module.np = np
            module.numpy = np
        except ImportError:
            pass
        
        try:
            import pandas as pd
            module.pd = pd
            module.pandas = pd
        except ImportError:
            pass
        
        # Add base class and essential types
        module.BaseAlgorithm = self.base_class
        module.Dict = Dict
        module.Any = Any
        module.List = List
        module.Optional = Optional
        module.time = time
        module.math = math
        module.random = random
        module.defaultdict = defaultdict
        module.Counter = Counter
    
    def _find_algorithm_class(self, namespace: Dict[str, Any]) -> Optional[Type]:
        """Find concrete algorithm class in namespace."""
        for obj in namespace.values():
            if not inspect.isclass(obj):
                continue
            
            # Skip the base class itself
            if obj is self.base_class:
                continue
            
            # Must be subclass of base class
            try:
                if not issubclass(obj, self.base_class):
                    continue
            except TypeError:
                continue
            
            # Check if it's concrete (no abstract methods)
            abstracts = getattr(obj, "__abstractmethods__", set())
            if abstracts:
                continue
            
            return obj
        
        return None
    
    def _validate_algorithm_class(self, algorithm_class: Type) -> bool:
        """Validate algorithm class implementation."""
        try:
            # Check required methods
            required_methods = ['learn', 'conformance']
            for method_name in required_methods:
                if not hasattr(algorithm_class, method_name):
                    raise AlgorithmValidationError(f"Missing required method: {method_name}")
                
                method = getattr(algorithm_class, method_name)
                if not callable(method):
                    raise AlgorithmValidationError(f"Method {method_name} is not callable")
            
            # Try to instantiate
            try:
                instance = algorithm_class()
            except Exception as e:
                raise AlgorithmValidationError(f"Cannot instantiate algorithm: {str(e)}")
            
            # Validate method signatures
            learn_sig = inspect.signature(instance.learn)
            if len(learn_sig.parameters) < 1:
                raise AlgorithmValidationError("learn() method must accept an event parameter")
            
            conformance_sig = inspect.signature(instance.conformance)
            if len(conformance_sig.parameters) < 1:
                raise AlgorithmValidationError("conformance() method must accept an event parameter")
            
            return True
            
        except Exception as e:
            logger.error(f"Algorithm validation failed: {e}")
            raise

class AlgorithmTestingEngine:
    """Enhanced algorithm testing engine with performance monitoring and safety."""
    
    def __init__(self, base_algorithm_class, stream_manager):
        self.base_class = base_algorithm_class
        self.stream_manager = stream_manager
        self.loader = SafeAlgorithmLoader(base_algorithm_class)
        self.monitor = PerformanceMonitor()
        
        # Configuration
        self.max_execution_time = 300  # 5 minutes
        self.max_memory_mb = 1024  # 1GB
        self.memory_check_interval = 10  # Check every 10 events
    
    def test_algorithm(self, algorithm_path: str, stream_id: str, num_cases: int = 500,
                      timeout: int = None) -> Tuple[bool, Dict[str, Any]]:
        """Test algorithm with comprehensive monitoring and error handling."""
        timeout = timeout or self.max_execution_time
        
        try:
            # Load algorithm
            algorithm_class, error = self.loader.load_algorithm_from_file(algorithm_path, timeout)
            if not algorithm_class:
                return False, {"error": f"Failed to load algorithm: {error}"}
            
            # Generate test data
            event_log, baseline_scores = self.stream_manager.generate_concept_drift_stream(
                stream_id, num_cases
            )
            
            # Run test with monitoring
            test_results = self._run_algorithm_test(
                algorithm_class, event_log, baseline_scores, stream_id, timeout
            )
            
            return True, test_results
            
        except Exception as e:
            error_msg = f"Test execution failed: {str(e)}"
            logger.error(f"{error_msg}\n{traceback.format_exc()}")
            return False, {"error": error_msg}
    
    def _run_algorithm_test(self, algorithm_class: Type, event_log: pd.DataFrame,
                           baseline_scores: List[float], stream_id: str, timeout: int) -> Dict[str, Any]:
        """Run algorithm test with monitoring."""
        self.monitor.reset()
        self.monitor.start_monitoring()
        
        try:
            # Create algorithm instance
            algorithm = algorithm_class()
            
            # Prepare event stream
            from src.ui.algorithm_base import EventStream
            event_stream = EventStream(event_log)
            
            # Learning phase
            ground_truth_events = event_stream.get_ground_truth_events()
            algorithm.on_learning_phase_start({'total_events': len(ground_truth_events)})
            
            for i, event in enumerate(ground_truth_events):
                start_time = time.time()
                
                # Check timeout
                if time.time() - self.monitor.start_time > timeout:
                    raise AlgorithmTimeoutError(f"Algorithm exceeded timeout of {timeout} seconds")
                
                # Check memory periodically
                if i % self.memory_check_interval == 0:
                    self.monitor.sample_memory()
                    if self.monitor.peak_memory > self.max_memory_mb:
                        raise AlgorithmMemoryError(f"Algorithm exceeded memory limit of {self.max_memory_mb}MB")
                
                # Execute learn
                try:
                    algorithm.learn(event)
                except Exception as e:
                    raise Exception(f"Error in learn() at event {i}: {str(e)}")
                
                # Record timing
                duration = time.time() - start_time
                self.monitor.record_event_time('learn', duration)
                
                # Garbage collection every 100 events
                if i % 100 == 0:
                    gc.collect()
            
            algorithm.on_learning_phase_end({'learned_events': len(ground_truth_events)})
            
            # Conformance checking phase
            full_stream = event_stream.get_full_stream()
            conformance_scores = []
            
            for i, event in enumerate(full_stream):
                start_time = time.time()
                
                # Check timeout
                if time.time() - self.monitor.start_time > timeout:
                    raise AlgorithmTimeoutError(f"Algorithm exceeded timeout of {timeout} seconds")
                
                # Check memory periodically
                if i % self.memory_check_interval == 0:
                    self.monitor.sample_memory()
                    if self.monitor.peak_memory > self.max_memory_mb:
                        raise AlgorithmMemoryError(f"Algorithm exceeded memory limit of {self.max_memory_mb}MB")
                
                # Execute conformance check
                try:
                    score = algorithm.conformance(event)
                    
                    # Validate score
                    if not isinstance(score, (int, float)):
                        raise ValueError(f"conformance() returned non-numeric value: {type(score)}")
                    
                    if not (0.0 <= score <= 1.0):
                        logger.warning(f"conformance() returned value outside [0,1]: {score}")
                        score = max(0.0, min(1.0, score))  # Clamp to valid range
                    
                    conformance_scores.append(float(score))
                    
                except Exception as e:
                    raise Exception(f"Error in conformance() at event {i}: {str(e)}")
                
                # Record timing
                duration = time.time() - start_time
                self.monitor.record_event_time('conformance', duration)
                
                # Garbage collection every 100 events
                if i % 100 == 0:
                    gc.collect()
            
            algorithm.on_stream_complete({'total_events': len(full_stream)})
            
            # Calculate evaluation metrics
            metrics = self._calculate_metrics(conformance_scores, baseline_scores)
            
            # Get performance metrics
            self.monitor.stop_monitoring()
            performance_metrics = self.monitor.get_metrics()
            
            # Get algorithm-specific metrics if available
            algorithm_metrics = {}
            if hasattr(algorithm, 'get_performance_metrics'):
                try:
                    algorithm_metrics = algorithm.get_performance_metrics()
                except:
                    pass
            
            return {
                'success': True,
                'stream_id': stream_id,
                'num_cases': len(event_log),
                'metrics': metrics,
                'performance': performance_metrics,
                'algorithm_metrics': algorithm_metrics,
                'conformance_scores': conformance_scores,
                'baseline_scores': baseline_scores,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.monitor.stop_monitoring()
            raise e
    
    def _calculate_metrics(self, conformance_scores: List[float], 
                          baseline_scores: List[float]) -> Dict[str, float]:
        """Calculate evaluation metrics."""
        if len(conformance_scores) != len(baseline_scores):
            raise ValueError("Conformance and baseline scores must have same length")
        
        # Convert to numpy arrays for efficient computation
        conf_array = np.array(conformance_scores)
        base_array = np.array(baseline_scores)
        
        # Calculate metrics
        mae = np.mean(np.abs(conf_array - base_array))
        rmse = np.sqrt(np.mean((conf_array - base_array) ** 2))
        
        # Accuracy: percentage within 0.1 of baseline
        accuracy = np.mean(np.abs(conf_array - base_array) <= 0.1)
        
        # Global errors: major deviations (>0.3)
        global_errors = np.sum(np.abs(conf_array - base_array) > 0.3)
        
        # Additional metrics
        correlation = np.corrcoef(conf_array, base_array)[0, 1] if len(conf_array) > 1 else 0.0
        mean_conformance = np.mean(conf_array)
        std_conformance = np.std(conf_array)
        
        return {
            'mae': float(mae),
            'rmse': float(rmse),
            'accuracy': float(accuracy),
            'global_errors': int(global_errors),
            'correlation': float(correlation) if not np.isnan(correlation) else 0.0,
            'mean_conformance': float(mean_conformance),
            'std_conformance': float(std_conformance),
            'min_conformance': float(np.min(conf_array)),
            'max_conformance': float(np.max(conf_array))
        }

# Utility functions for testing
def validate_algorithm_file(file_path: str, base_algorithm_class) -> Tuple[bool, str]:
    """Quick validation of algorithm file without full testing."""
    try:
        loader = SafeAlgorithmLoader(base_algorithm_class)
        algorithm_class, error = loader.load_algorithm_from_file(file_path, timeout=10)
        
        if algorithm_class:
            return True, "Algorithm validation successful"
        else:
            return False, error or "Unknown validation error"
            
    except Exception as e:
        return False, f"Validation failed: {str(e)}"

def estimate_test_duration(num_cases: int, algorithm_complexity: str = "medium") -> int:
    """Estimate test duration based on parameters."""
    base_time_per_case = {
        "simple": 0.001,   # 1ms per case
        "medium": 0.01,    # 10ms per case  
        "complex": 0.1     # 100ms per case
    }
    
    time_per_case = base_time_per_case.get(algorithm_complexity, 0.01)
    estimated_seconds = num_cases * time_per_case * 2  # Factor of 2 for safety
    
    return max(30, min(600, int(estimated_seconds)))  # Between 30s and 10min

# Export main classes
__all__ = [
    'AlgorithmTestingEngine', 
    'SafeAlgorithmLoader', 
    'PerformanceMonitor',
    'AlgorithmTimeoutError',
    'AlgorithmMemoryError', 
    'AlgorithmValidationError',
    'validate_algorithm_file',
    'estimate_test_duration'
]