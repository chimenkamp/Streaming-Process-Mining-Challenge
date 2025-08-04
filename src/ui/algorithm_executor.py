"""
Advanced Algorithm Executor and Evaluator for AVOCADO Challenge Platform

This module provides sophisticated algorithm execution, monitoring, and evaluation
with proper error handling, timeout management, and performance metrics.
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
import gc
import psutil
import resource
from contextlib import contextmanager
from typing import Dict, List, Any, Optional, Type, Callable, Tuple, Union
import pandas as pd
import numpy as np
from datetime import datetime
import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ExecutionResult:
    """Result of algorithm execution."""
    success: bool
    algorithm_id: int
    stream_id: str
    num_cases: int
    execution_time: float
    memory_usage_mb: float
    conformance_scores: List[float]
    baseline_scores: List[float]
    metrics: Dict[str, float]
    error_message: Optional[str] = None
    performance_data: Optional[Dict[str, Any]] = None


@dataclass
class AlgorithmStats:
    """Statistics for algorithm performance."""
    total_events: int = 0
    learn_events: int = 0
    conformance_events: int = 0
    avg_event_time: float = 0.0
    max_event_time: float = 0.0
    total_execution_time: float = 0.0
    peak_memory_mb: float = 0.0
    errors_count: int = 0


class AlgorithmTimeoutError(Exception):
    """Raised when algorithm execution times out."""
    pass


class AlgorithmMemoryError(Exception):
    """Raised when algorithm exceeds memory limits."""
    pass


class AlgorithmExecutionError(Exception):
    """Raised when algorithm execution fails."""
    pass


class PerformanceMonitor:
    """Monitors algorithm performance during execution."""

    def __init__(self, max_memory_mb: int = 1024, memory_check_interval: int = 10):
        self.max_memory_mb = max_memory_mb
        self.memory_check_interval = memory_check_interval
        self.reset()

    def reset(self):
        """Reset all monitoring data."""
        self.start_time = None
        self.end_time = None
        self.event_times = []
        self.memory_samples = []
        self.peak_memory = 0.0
        self.event_count = 0
        self.error_count = 0

    def start_monitoring(self):
        """Start performance monitoring."""
        self.start_time = time.time()
        self.sample_memory()

    def stop_monitoring(self):
        """Stop performance monitoring."""
        self.end_time = time.time()
        self.sample_memory()

    def record_event(self, duration: float, had_error: bool = False):
        """Record event execution time."""
        self.event_times.append(duration)
        self.event_count += 1

        if had_error:
            self.error_count += 1

        # Sample memory periodically
        if self.event_count % self.memory_check_interval == 0:
            self.sample_memory()

    def sample_memory(self):
        """Sample current memory usage."""
        try:
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            self.memory_samples.append(memory_mb)
            self.peak_memory = max(self.peak_memory, memory_mb)

            # Check memory limit
            if memory_mb > self.max_memory_mb:
                raise AlgorithmMemoryError(f"Memory usage {memory_mb:.1f}MB exceeds limit {self.max_memory_mb}MB")

        except psutil.NoSuchProcess:
            pass

    def get_stats(self) -> AlgorithmStats:
        """Get performance statistics."""
        total_time = (self.end_time - self.start_time) if self.end_time and self.start_time else 0.0

        return AlgorithmStats(
            total_events=self.event_count,
            avg_event_time=np.mean(self.event_times) if self.event_times else 0.0,
            max_event_time=np.max(self.event_times) if self.event_times else 0.0,
            total_execution_time=total_time,
            peak_memory_mb=self.peak_memory,
            errors_count=self.error_count
        )


class SafeAlgorithmLoader:
    """Safely loads and validates algorithm classes."""

    def __init__(self, base_algorithm_class):
        self.base_class = base_algorithm_class

    def load_algorithm_from_path(self, algorithm_path: str) -> Tuple[Optional[Type], Optional[str]]:
        """Load algorithm class from file or directory path."""
        try:
            path = Path(algorithm_path)

            if path.is_file() and path.suffix == '.py':
                return self._load_from_file(path)
            elif path.is_dir():
                return self._load_from_directory(path)
            else:
                return None, f"Invalid path: {algorithm_path}"

        except Exception as e:
            error_msg = f"Failed to load algorithm: {str(e)}"
            logger.error(f"{error_msg}\n{traceback.format_exc()}")
            return None, error_msg

    def _load_from_file(self, file_path: Path) -> Tuple[Optional[Type], Optional[str]]:
        """Load algorithm from a single Python file."""
        try:
            # Security check
            if not self._security_check(file_path):
                return None, "Algorithm contains potentially unsafe operations"

            # Create isolated environment
            with self._isolated_import_context(file_path.parent):
                algorithm_class = self._import_algorithm_class(file_path)

                if algorithm_class:
                    validation_result = self._validate_algorithm_class(algorithm_class)
                    if validation_result[0]:
                        return algorithm_class, None
                    else:
                        return None, f"Validation failed: {', '.join(validation_result[1])}"
                else:
                    return None, "No valid algorithm class found"

        except Exception as e:
            return None, f"Import error: {str(e)}"

    def _load_from_directory(self, dir_path: Path) -> Tuple[Optional[Type], Optional[str]]:
        """Load algorithm from directory containing Python files."""
        python_files = list(dir_path.rglob('*.py'))

        if not python_files:
            return None, "No Python files found in directory"

        # Try each Python file
        for py_file in python_files:
            try:
                algorithm_class, error = self._load_from_file(py_file)
                if algorithm_class:
                    return algorithm_class, None
            except Exception as e:
                logger.debug(f"Failed to load from {py_file}: {e}")
                continue

        return None, "No valid algorithm class found in any Python file"

    def _security_check(self, file_path: Path) -> bool:
        """Basic security check for dangerous operations."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Check for obviously dangerous patterns
            dangerous_patterns = [
                'import os.system', 'import subprocess', 'exec(', 'eval(',
                '__import__', 'open(', 'file(', 'input(', 'raw_input(',
                'import socket', 'import urllib', 'import requests',
                'sys.exit', 'quit(', 'exit(', 'globals(', 'locals(',
                'vars(', 'dir(', 'getattr(', 'setattr(', 'delattr(',
                'hasattr(', 'compile(', 'reload('
            ]

            content_lower = content.lower()
            for pattern in dangerous_patterns:
                if pattern in content_lower:
                    logger.warning(f"Potentially unsafe pattern found: {pattern}")
                    # For now just warn, in production might want to reject

            return True  # Allow for now

        except Exception as e:
            logger.error(f"Security check failed: {e}")
            return False

    @contextmanager
    def _isolated_import_context(self, directory: Path):
        """Create isolated import context."""
        old_sys_path = sys.path.copy()
        try:
            if str(directory) not in sys.path:
                sys.path.insert(0, str(directory))
            yield
        finally:
            sys.path = old_sys_path

    def _import_algorithm_class(self, file_path: Path) -> Optional[Type]:
        """Import and find algorithm class from file."""
        try:
            module_name = file_path.stem
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
        """Prepare module with safe imports."""
        # Essential imports
        import time
        import math
        import random
        from collections import defaultdict, Counter
        from typing import Dict, Any, List, Optional

        # Add to module namespace
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

        # Scientific libraries (if available)
        try:
            import numpy as np
            import pandas as pd
            module.np = np
            module.numpy = np
            module.pd = pd
            module.pandas = pd
        except ImportError:
            pass

    def _find_algorithm_class(self, namespace: Dict[str, Any]) -> Optional[Type]:
        """Find concrete algorithm class in namespace."""
        for obj in namespace.values():
            if not inspect.isclass(obj):
                continue

            # Skip base class
            if obj is self.base_class:
                continue

            # Must be subclass
            try:
                if not issubclass(obj, self.base_class):
                    continue
            except TypeError:
                continue

            # Check if concrete (no abstract methods)
            abstracts = getattr(obj, "__abstractmethods__", set())
            if abstracts:
                continue

            return obj

        return None

    def _validate_algorithm_class(self, algorithm_class: Type) -> Tuple[bool, List[str]]:
        """Validate algorithm class meets requirements."""
        issues = []

        try:
            # Check required methods
            required_methods = ['learn', 'conformance']
            for method_name in required_methods:
                if not hasattr(algorithm_class, method_name):
                    issues.append(f"Missing required method: {method_name}")
                elif not callable(getattr(algorithm_class, method_name)):
                    issues.append(f"Method {method_name} is not callable")

            # Try to instantiate
            try:
                instance = algorithm_class()

                # Check method signatures
                if hasattr(instance, 'learn'):
                    sig = inspect.signature(instance.learn)
                    if len(sig.parameters) < 1:
                        issues.append("learn() method must accept an event parameter")

                if hasattr(instance, 'conformance'):
                    sig = inspect.signature(instance.conformance)
                    if len(sig.parameters) < 1:
                        issues.append("conformance() method must accept an event parameter")

            except Exception as e:
                issues.append(f"Cannot instantiate class: {str(e)}")

        except Exception as e:
            issues.append(f"Validation error: {str(e)}")

        return len(issues) == 0, issues


class AlgorithmExecutor:
    """Executes algorithms with comprehensive monitoring and safety."""

    def __init__(self, base_algorithm_class, stream_manager, database_manager):
        self.base_class = base_algorithm_class
        self.stream_manager = stream_manager
        self.database_manager = database_manager
        self.loader = SafeAlgorithmLoader(base_algorithm_class)

        # Configuration
        self.max_execution_time = 300  # 5 minutes
        self.max_memory_mb = 1024  # 1GB
        self.warmup_ratio = 0.1  # 10% for learning

    def execute_algorithm_test(self, algorithm_id: int, token: str, stream_id: str,
                               num_cases: int = 500) -> ExecutionResult:
        """Execute algorithm test with comprehensive monitoring."""
        monitor = PerformanceMonitor(self.max_memory_mb)

        try:
            # Get algorithm info
            algorithm = self.database_manager.get_algorithm_by_id(algorithm_id, token)
            if not algorithm:
                raise AlgorithmExecutionError("Algorithm not found")

            # Load algorithm class
            algorithm_class, error = self.loader.load_algorithm_from_path(algorithm.file_path)
            if not algorithm_class:
                raise AlgorithmExecutionError(f"Failed to load algorithm: {error}")

            # Start monitoring
            monitor.start_monitoring()

            # Generate test stream
            event_log, baseline_scores = self.stream_manager.generate_concept_drift_stream(
                stream_id, num_cases
            )

            # Execute algorithm
            conformance_scores = self._execute_algorithm_on_stream(
                algorithm_class, event_log, monitor
            )

            # Stop monitoring
            monitor.stop_monitoring()

            # Calculate metrics
            metrics = self._calculate_metrics(conformance_scores, baseline_scores)

            # Update algorithm status
            self.database_manager.get_algorithm_by_id(algorithm_id, token)
            # You might want to add a method to update test results

            return ExecutionResult(
                success=True,
                algorithm_id=algorithm_id,
                stream_id=stream_id,
                num_cases=num_cases,
                execution_time=monitor.get_stats().total_execution_time,
                memory_usage_mb=monitor.get_stats().peak_memory_mb,
                conformance_scores=conformance_scores,
                baseline_scores=baseline_scores,
                metrics=metrics,
                performance_data=asdict(monitor.get_stats())
            )

        except Exception as e:
            monitor.stop_monitoring()
            error_msg = f"Algorithm execution failed: {str(e)}"
            logger.error(f"{error_msg}\n{traceback.format_exc()}")

            return ExecutionResult(
                success=False,
                algorithm_id=algorithm_id,
                stream_id=stream_id,
                num_cases=num_cases,
                execution_time=monitor.get_stats().total_execution_time,
                memory_usage_mb=monitor.get_stats().peak_memory_mb,
                conformance_scores=[],
                baseline_scores=[],
                metrics={},
                error_message=error_msg,
                performance_data=asdict(monitor.get_stats())
            )

    def _execute_algorithm_on_stream(self, algorithm_class: Type, event_log: pd.DataFrame,
                                     monitor: PerformanceMonitor) -> List[float]:
        """Execute algorithm on event stream with timeout and monitoring."""
        from src.ui.algorithm_base import EventStream

        # Create algorithm instance
        algorithm = algorithm_class()

        # Create event stream
        event_stream = EventStream(event_log, self.warmup_ratio)

        # Learning phase
        start_time = time.time()
        ground_truth_events = event_stream.get_ground_truth_events()

        algorithm.on_learning_phase_start({'total_events': len(ground_truth_events)})

        for i, event in enumerate(ground_truth_events):
            # Check timeout
            if time.time() - start_time > self.max_execution_time:
                raise AlgorithmTimeoutError(f"Algorithm exceeded timeout during learning phase")

            # Execute learn with timing
            event_start = time.time()
            try:
                algorithm.learn(event)
                had_error = False
            except Exception as e:
                logger.warning(f"Error in learn() at event {i}: {e}")
                had_error = True

            event_end = time.time()
            monitor.record_event(event_end - event_start, had_error)

            # Garbage collection periodically
            if i % 100 == 0:
                gc.collect()

        algorithm.on_learning_phase_end({'learned_events': len(ground_truth_events)})

        # Conformance checking phase
        full_stream = event_stream.get_full_stream()
        conformance_scores = []

        for i, event in enumerate(full_stream):
            # Check timeout
            if time.time() - start_time > self.max_execution_time:
                raise AlgorithmTimeoutError(f"Algorithm exceeded timeout during conformance checking")

            # Execute conformance check with timing
            event_start = time.time()
            try:
                score = algorithm.conformance(event)

                # Validate score
                if not isinstance(score, (int, float)):
                    logger.warning(f"conformance() returned non-numeric value at event {i}: {type(score)}")
                    score = 0.0
                elif not (0.0 <= score <= 1.0):
                    logger.warning(f"conformance() returned out-of-range value at event {i}: {score}")
                    score = max(0.0, min(1.0, float(score)))

                conformance_scores.append(float(score))
                had_error = False

            except Exception as e:
                logger.warning(f"Error in conformance() at event {i}: {e}")
                conformance_scores.append(0.0)  # Default to non-conforming
                had_error = True

            event_end = time.time()
            monitor.record_event(event_end - event_start, had_error)

            # Garbage collection periodically
            if i % 100 == 0:
                gc.collect()

        algorithm.on_stream_complete({'total_events': len(full_stream)})

        return conformance_scores

    def _calculate_metrics(self, conformance_scores: List[float],
                           baseline_scores: List[float]) -> Dict[str, float]:
        """Calculate evaluation metrics."""
        if len(conformance_scores) != len(baseline_scores):
            raise ValueError("Conformance and baseline scores must have same length")

        conf_array = np.array(conformance_scores)
        base_array = np.array(baseline_scores)

        # Core metrics
        mae = float(np.mean(np.abs(conf_array - base_array)))
        rmse = float(np.sqrt(np.mean((conf_array - base_array) ** 2)))
        accuracy = float(np.mean(np.abs(conf_array - base_array) <= 0.1))
        global_errors = int(np.sum(np.abs(conf_array - base_array) > 0.3))

        # Additional metrics
        correlation = float(np.corrcoef(conf_array, base_array)[0, 1]) if len(conf_array) > 1 else 0.0
        if np.isnan(correlation):
            correlation = 0.0

        mean_conformance = float(np.mean(conf_array))
        std_conformance = float(np.std(conf_array))

        return {
            'mae': mae,
            'rmse': rmse,
            'accuracy': accuracy,
            'global_errors': global_errors,
            'correlation': correlation,
            'mean_conformance': mean_conformance,
            'std_conformance': std_conformance,
            'min_conformance': float(np.min(conf_array)),
            'max_conformance': float(np.max(conf_array))
        }

    def get_best_algorithm_for_user(self, token: str) -> Optional[Dict[str, Any]]:
        """Get user's best performing algorithm based on test results."""
        algorithms = self.database_manager.get_user_algorithms(token)

        best_algorithm = None
        best_score = -1.0

        for alg in algorithms:
            if alg.test_results and alg.status == 'tested':
                # Calculate composite score
                metrics = alg.test_results.get('metrics', {})
                score = self._calculate_composite_score(metrics)

                if score > best_score:
                    best_score = score
                    best_algorithm = {
                        'id': alg.id,
                        'name': alg.algorithm_name,
                        'score': score,
                        'metrics': metrics
                    }

        return best_algorithm

    def _calculate_composite_score(self, metrics: Dict[str, float]) -> float:
        """Calculate composite score for ranking."""
        weights = {
            'accuracy': 0.30,
            'mae': 0.25,  # Lower is better, so invert
            'rmse': 0.20,  # Lower is better, so invert
            'correlation': 0.15,
            'global_errors': 0.10  # Lower is better, so invert
        }

        score = 0.0

        # Accuracy (higher is better)
        if 'accuracy' in metrics:
            score += metrics['accuracy'] * weights['accuracy']

        # MAE (lower is better)
        if 'mae' in metrics:
            mae_score = 1.0 / (1.0 + metrics['mae'])
            score += mae_score * weights['mae']

        # RMSE (lower is better)
        if 'rmse' in metrics:
            rmse_score = 1.0 / (1.0 + metrics['rmse'])
            score += rmse_score * weights['rmse']

        # Correlation (higher is better)
        if 'correlation' in metrics:
            score += max(0, metrics['correlation']) * weights['correlation']

        # Global errors (lower is better)
        if 'global_errors' in metrics:
            error_score = 1.0 / (1.0 + metrics['global_errors'])
            score += error_score * weights['global_errors']

        return score


# Export main classes
__all__ = [
    'AlgorithmExecutor',
    'ExecutionResult',
    'AlgorithmStats',
    'PerformanceMonitor',
    'SafeAlgorithmLoader',
    'AlgorithmTimeoutError',
    'AlgorithmMemoryError',
    'AlgorithmExecutionError'
]