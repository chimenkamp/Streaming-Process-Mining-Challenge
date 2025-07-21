"""
Leaderboard and Submission Management System

This module handles algorithm submissions, evaluation, and leaderboard management
for the Streaming Process Mining Challenge.
"""

import json
import sqlite3
import hashlib
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import pandas as pd
import numpy as np
from collections import defaultdict


@dataclass
class SubmissionEntry:
    """Data class for a submission entry."""
    submission_id: str
    team_name: str
    email: str
    algorithm_name: str
    description: str
    submission_time: datetime
    file_hash: str
    file_path: str
    libraries: List[str]
    status: str  # 'pending', 'evaluating', 'completed', 'failed'
    evaluation_results: Optional[Dict[str, Any]] = None
    leaderboard_scores: Optional[Dict[str, float]] = None


@dataclass
class EvaluationResult:
    """Data class for evaluation results."""
    submission_id: str
    stream_id: str
    num_cases: int
    mae: float
    rmse: float
    accuracy: float
    global_errors: int
    avg_processing_time: float
    total_runtime: float
    conformance_scores: List[float]
    baseline_scores: List[float]
    timestamp: datetime


class SubmissionDatabase:
    """Manages submissions in SQLite database."""
    
    def __init__(self, db_path: str = "challenge_submissions.db"):
        """Initialize database connection."""
        self.db_path = Path(db_path)
        self._init_database()
    
    def _init_database(self):
        """Initialize database tables."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS submissions (
                    submission_id TEXT PRIMARY KEY,
                    team_name TEXT NOT NULL,
                    email TEXT NOT NULL,
                    algorithm_name TEXT NOT NULL,
                    description TEXT,
                    submission_time TEXT NOT NULL,
                    file_hash TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    libraries TEXT,
                    status TEXT NOT NULL,
                    evaluation_results TEXT,
                    leaderboard_scores TEXT
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS evaluations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    submission_id TEXT NOT NULL,
                    stream_id TEXT NOT NULL,
                    num_cases INTEGER NOT NULL,
                    mae REAL NOT NULL,
                    rmse REAL NOT NULL,
                    accuracy REAL NOT NULL,
                    global_errors INTEGER NOT NULL,
                    avg_processing_time REAL NOT NULL,
                    total_runtime REAL NOT NULL,
                    conformance_scores TEXT NOT NULL,
                    baseline_scores TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    FOREIGN KEY (submission_id) REFERENCES submissions (submission_id)
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_submissions_time 
                ON submissions (submission_time)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_evaluations_submission 
                ON evaluations (submission_id)
            """)
    
    def add_submission(self, submission: SubmissionEntry) -> bool:
        """Add a new submission to the database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO submissions (
                        submission_id, team_name, email, algorithm_name, description,
                        submission_time, file_hash, file_path, libraries, status,
                        evaluation_results, leaderboard_scores
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    submission.submission_id,
                    submission.team_name,
                    submission.email,
                    submission.algorithm_name,
                    submission.description,
                    submission.submission_time.isoformat(),
                    submission.file_hash,
                    submission.file_path,
                    json.dumps(submission.libraries),
                    submission.status,
                    json.dumps(submission.evaluation_results) if submission.evaluation_results else None,
                    json.dumps(submission.leaderboard_scores) if submission.leaderboard_scores else None
                ))
            return True
        except Exception as e:
            print(f"Error adding submission: {e}")
            return False
    
    def update_submission_status(self, submission_id: str, status: str, 
                                evaluation_results: Optional[Dict] = None,
                                leaderboard_scores: Optional[Dict] = None) -> bool:
        """Update submission status and results."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    UPDATE submissions 
                    SET status = ?, evaluation_results = ?, leaderboard_scores = ?
                    WHERE submission_id = ?
                """, (
                    status,
                    json.dumps(evaluation_results) if evaluation_results else None,
                    json.dumps(leaderboard_scores) if leaderboard_scores else None,
                    submission_id
                ))
            return True
        except Exception as e:
            print(f"Error updating submission: {e}")
            return False
    
    def add_evaluation(self, evaluation: EvaluationResult) -> bool:
        """Add evaluation result to database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO evaluations (
                        submission_id, stream_id, num_cases, mae, rmse, accuracy,
                        global_errors, avg_processing_time, total_runtime,
                        conformance_scores, baseline_scores, timestamp
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    evaluation.submission_id,
                    evaluation.stream_id,
                    evaluation.num_cases,
                    evaluation.mae,
                    evaluation.rmse,
                    evaluation.accuracy,
                    evaluation.global_errors,
                    evaluation.avg_processing_time,
                    evaluation.total_runtime,
                    json.dumps(evaluation.conformance_scores),
                    json.dumps(evaluation.baseline_scores),
                    evaluation.timestamp.isoformat()
                ))
            return True
        except Exception as e:
            print(f"Error adding evaluation: {e}")
            return False
    
    def get_submissions(self, limit: Optional[int] = None) -> List[SubmissionEntry]:
        """Get submissions from database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                query = """
                    SELECT * FROM submissions 
                    ORDER BY submission_time DESC
                """
                if limit:
                    query += f" LIMIT {limit}"
                
                cursor = conn.execute(query)
                rows = cursor.fetchall()
                
                submissions = []
                for row in rows:
                    submission = SubmissionEntry(
                        submission_id=row[0],
                        team_name=row[1],
                        email=row[2],
                        algorithm_name=row[3],
                        description=row[4],
                        submission_time=datetime.fromisoformat(row[5]),
                        file_hash=row[6],
                        file_path=row[7],
                        libraries=json.loads(row[8]) if row[8] else [],
                        status=row[9],
                        evaluation_results=json.loads(row[10]) if row[10] else None,
                        leaderboard_scores=json.loads(row[11]) if row[11] else None
                    )
                    submissions.append(submission)
                
                return submissions
        except Exception as e:
            print(f"Error getting submissions: {e}")
            return []
    
    def get_submission(self, submission_id: str) -> Optional[SubmissionEntry]:
        """Get specific submission by ID."""
        submissions = self.get_submissions()
        for submission in submissions:
            if submission.submission_id == submission_id:
                return submission
        return None
    
    def get_evaluations(self, submission_id: str) -> List[EvaluationResult]:
        """Get evaluations for a specific submission."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT * FROM evaluations 
                    WHERE submission_id = ?
                    ORDER BY timestamp DESC
                """, (submission_id,))
                rows = cursor.fetchall()
                
                evaluations = []
                for row in rows:
                    evaluation = EvaluationResult(
                        submission_id=row[1],
                        stream_id=row[2],
                        num_cases=row[3],
                        mae=row[4],
                        rmse=row[5],
                        accuracy=row[6],
                        global_errors=row[7],
                        avg_processing_time=row[8],
                        total_runtime=row[9],
                        conformance_scores=json.loads(row[10]),
                        baseline_scores=json.loads(row[11]),
                        timestamp=datetime.fromisoformat(row[12])
                    )
                    evaluations.append(evaluation)
                
                return evaluations
        except Exception as e:
            print(f"Error getting evaluations: {e}")
            return []


class EvaluationEngine:
    """Handles algorithm evaluation across multiple streams."""
    
    def __init__(self, stream_manager, upload_manager, algorithm_loader):
        """Initialize evaluation engine."""
        self.stream_manager = stream_manager
        self.upload_manager = upload_manager
        self.algorithm_loader = algorithm_loader
        
        # Evaluation configuration
        self.test_streams = ['simple_sequential']  # Add more as available
        self.test_cases = [100, 250, 500]  # Different case counts for robustness
        self.timeout_seconds = 300  # 5 minutes per test
    
    def evaluate_submission(self, submission: SubmissionEntry) -> Tuple[bool, Dict[str, Any]]:
        """
        Evaluate a submission across multiple test scenarios.
        
        Returns:
            Tuple of (success, results_dict)
        """
        try:
            # Load algorithm
            algorithm_files = self.upload_manager.find_algorithm_files(submission.file_path)
            if not algorithm_files:
                return False, {"error": "No algorithm files found"}
            
            algorithm_class = None
            for file_path in algorithm_files:
                algorithm_class, error = self.algorithm_loader.load_algorithm_from_file(file_path)
                if algorithm_class:
                    break
            
            if not algorithm_class:
                return False, {"error": "Could not load algorithm class"}
            
            # Run evaluations
            results = {}
            overall_scores = defaultdict(list)
            
            for stream_id in self.test_streams:
                stream_results = {}
                
                for num_cases in self.test_cases:
                    test_key = f"{stream_id}_{num_cases}"
                    
                    try:
                        evaluation_result = self._run_single_evaluation(
                            algorithm_class, stream_id, num_cases, submission.submission_id
                        )
                        
                        if evaluation_result:
                            stream_results[test_key] = asdict(evaluation_result)
                            
                            # Collect scores for overall ranking
                            overall_scores['mae'].append(evaluation_result.mae)
                            overall_scores['rmse'].append(evaluation_result.rmse)
                            overall_scores['accuracy'].append(evaluation_result.accuracy)
                            overall_scores['processing_time'].append(evaluation_result.avg_processing_time)
                            overall_scores['global_errors'].append(evaluation_result.global_errors)
                        
                    except Exception as e:
                        stream_results[test_key] = {"error": str(e)}
                
                results[stream_id] = stream_results
            
            # Calculate overall leaderboard scores
            leaderboard_scores = self._calculate_leaderboard_scores(overall_scores)
            
            return True, {
                "detailed_results": results,
                "leaderboard_scores": leaderboard_scores,
                "evaluation_summary": self._create_evaluation_summary(results)
            }
            
        except Exception as e:
            return False, {"error": f"Evaluation failed: {str(e)}"}
    
    def _run_single_evaluation(self, algorithm_class, stream_id: str, 
                              num_cases: int, submission_id: str) -> Optional[EvaluationResult]:
        """Run evaluation for a single stream/case combination."""
        try:
            # Generate stream
            event_log, baseline_results = self.stream_manager.generate_concept_drift_stream(
                stream_id, num_cases
            )
            
            from src.ui.algorithm_base import EventStream
            event_stream = EventStream(event_log)
            
            # Initialize algorithm
            algorithm = algorithm_class()
            start_time = datetime.now()
            
            # Learning phase
            ground_truth_events = event_stream.get_ground_truth_events()
            algorithm.on_learning_phase_start({'total_events': len(ground_truth_events)})
            
            for event in ground_truth_events:
                algorithm.learn(event)
            
            algorithm.on_learning_phase_end({'learned_events': len(ground_truth_events)})
            
            # Conformance checking phase
            full_stream = event_stream.get_full_stream()
            conformance_results = []
            
            for event in full_stream:
                conformance_score = algorithm.conformance(event)
                conformance_results.append(conformance_score)
            
            end_time = datetime.now()
            
            # Calculate metrics
            mae = np.mean([abs(a - b) for a, b in zip(conformance_results, baseline_results)])
            rmse = np.sqrt(np.mean([(a - b) ** 2 for a, b in zip(conformance_results, baseline_results)]))
            accuracy = np.mean([abs(a - b) <= 0.1 for a, b in zip(conformance_results, baseline_results)])
            global_errors = sum(1 for a, b in zip(conformance_results, baseline_results) if abs(a - b) > 0.3)
            
            # Get performance metrics
            performance_metrics = algorithm.get_performance_metrics()
            avg_processing_time = performance_metrics.get('avg_processing_time', 0.0)
            total_runtime = (end_time - start_time).total_seconds()
            
            return EvaluationResult(
                submission_id=submission_id,
                stream_id=stream_id,
                num_cases=num_cases,
                mae=mae,
                rmse=rmse,
                accuracy=accuracy,
                global_errors=global_errors,
                avg_processing_time=avg_processing_time,
                total_runtime=total_runtime,
                conformance_scores=conformance_results,
                baseline_scores=baseline_results,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            print(f"Error in single evaluation: {e}")
            return None
    
    def _calculate_leaderboard_scores(self, overall_scores: Dict[str, List[float]]) -> Dict[str, float]:
        """Calculate normalized scores for leaderboard ranking."""
        scores = {}
        
        if overall_scores['mae']:
            scores['mae_score'] = 1.0 / (1.0 + np.mean(overall_scores['mae']))  # Lower is better
        
        if overall_scores['rmse']:
            scores['rmse_score'] = 1.0 / (1.0 + np.mean(overall_scores['rmse']))  # Lower is better
        
        if overall_scores['accuracy']:
            scores['accuracy_score'] = np.mean(overall_scores['accuracy'])  # Higher is better
        
        if overall_scores['processing_time']:
            # Normalize processing time (lower is better, but cap at reasonable values)
            avg_time = np.mean(overall_scores['processing_time'])
            scores['speed_score'] = 1.0 / (1.0 + min(avg_time * 1000, 100))  # Convert to ms and cap
        
        if overall_scores['global_errors']:
            scores['robustness_score'] = 1.0 / (1.0 + np.mean(overall_scores['global_errors']))
        
        # Calculate composite score (weighted average)
        weights = {
            'mae_score': 0.25,
            'rmse_score': 0.20,
            'accuracy_score': 0.30,
            'speed_score': 0.15,
            'robustness_score': 0.10
        }
        
        composite_score = 0.0
        for metric, weight in weights.items():
            if metric in scores:
                composite_score += scores[metric] * weight
        
        scores['composite_score'] = composite_score
        
        return scores
    
    def _create_evaluation_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Create a summary of evaluation results."""
        total_tests = 0
        successful_tests = 0
        failed_tests = 0
        
        for stream_id, stream_results in results.items():
            for test_key, test_result in stream_results.items():
                total_tests += 1
                if "error" in test_result:
                    failed_tests += 1
                else:
                    successful_tests += 1
        
        return {
            "total_tests": total_tests,
            "successful_tests": successful_tests,
            "failed_tests": failed_tests,
            "success_rate": successful_tests / total_tests if total_tests > 0 else 0.0
        }


class LeaderboardManager:
    """Manages leaderboard generation and ranking."""
    
    def __init__(self, database: SubmissionDatabase):
        """Initialize leaderboard manager."""
        self.database = database
    
    def get_leaderboard(self, limit: Optional[int] = 50) -> List[Dict[str, Any]]:
        """
        Generate current leaderboard.
        
        Returns:
            List of leaderboard entries sorted by composite score
        """
        submissions = self.database.get_submissions()
        
        # Filter completed submissions with scores
        valid_submissions = [
            s for s in submissions 
            if s.status == 'completed' and s.leaderboard_scores
        ]
        
        # Create leaderboard entries
        leaderboard_entries = []
        for submission in valid_submissions:
            entry = {
                'rank': 0,  # Will be set after sorting
                'team_name': submission.team_name,
                'algorithm_name': submission.algorithm_name,
                'submission_time': submission.submission_time,
                'composite_score': submission.leaderboard_scores.get('composite_score', 0.0),
                'accuracy_score': submission.leaderboard_scores.get('accuracy_score', 0.0),
                'mae_score': submission.leaderboard_scores.get('mae_score', 0.0),
                'rmse_score': submission.leaderboard_scores.get('rmse_score', 0.0),
                'speed_score': submission.leaderboard_scores.get('speed_score', 0.0),
                'robustness_score': submission.leaderboard_scores.get('robustness_score', 0.0),
                'submission_id': submission.submission_id
            }
            leaderboard_entries.append(entry)
        
        # Sort by composite score (descending)
        leaderboard_entries.sort(key=lambda x: x['composite_score'], reverse=True)
        
        # Assign ranks
        for i, entry in enumerate(leaderboard_entries):
            entry['rank'] = i + 1
        
        # Apply limit
        if limit:
            leaderboard_entries = leaderboard_entries[:limit]
        
        return leaderboard_entries
    
    def get_team_submissions(self, team_name: str) -> List[Dict[str, Any]]:
        """Get all submissions for a specific team."""
        submissions = self.database.get_submissions()
        
        team_submissions = [
            {
                'submission_id': s.submission_id,
                'algorithm_name': s.algorithm_name,
                'submission_time': s.submission_time,
                'status': s.status,
                'composite_score': s.leaderboard_scores.get('composite_score', 0.0) if s.leaderboard_scores else 0.0
            }
            for s in submissions if s.team_name == team_name
        ]
        
        # Sort by submission time (newest first)
        team_submissions.sort(key=lambda x: x['submission_time'], reverse=True)
        
        return team_submissions
    
    def get_leaderboard_stats(self) -> Dict[str, Any]:
        """Get overall leaderboard statistics."""
        submissions = self.database.get_submissions()
        
        total_submissions = len(submissions)
        completed_submissions = len([s for s in submissions if s.status == 'completed'])
        pending_submissions = len([s for s in submissions if s.status == 'pending'])
        failed_submissions = len([s for s in submissions if s.status == 'failed'])
        
        # Unique teams
        unique_teams = len(set(s.team_name for s in submissions))
        
        # Recent activity (last 7 days)
        week_ago = datetime.now() - timedelta(days=7)
        recent_submissions = len([
            s for s in submissions if s.submission_time >= week_ago
        ])
        
        return {
            'total_submissions': total_submissions,
            'completed_submissions': completed_submissions,
            'pending_submissions': pending_submissions,
            'failed_submissions': failed_submissions,
            'unique_teams': unique_teams,
            'recent_submissions': recent_submissions,
            'completion_rate': completed_submissions / total_submissions if total_submissions > 0 else 0.0
        }


class SubmissionManager:
    """Main class for managing submissions and evaluations."""
    
    def __init__(self, upload_directory: str, database_path: str = "challenge_submissions.db"):
        """Initialize submission manager."""
        from src.ui.algorithm_base import BaseAlgorithm
        from src.file_upload_utils import AlgorithmUploadManager, AlgorithmLoader
        
        self.database = SubmissionDatabase(database_path)
        self.upload_manager = AlgorithmUploadManager(upload_directory)
        self.algorithm_loader = AlgorithmLoader(BaseAlgorithm)
        self.leaderboard_manager = LeaderboardManager(self.database)
        
        # Will be set when stream_manager is available
        self.evaluation_engine = None
    
    def set_stream_manager(self, stream_manager):
        """Set stream manager for evaluation engine."""
        self.evaluation_engine = EvaluationEngine(
            stream_manager, self.upload_manager, self.algorithm_loader
        )
    
    def submit_algorithm(self, team_name: str, email: str, algorithm_name: str,
                        description: str, file_path: str, libraries: List[str]) -> str:
        """
        Submit an algorithm for evaluation.
        
        Returns:
            Submission ID
        """
        # Generate submission ID and hash
        submission_id = str(uuid.uuid4())
        file_hash = self._calculate_file_hash(file_path)
        
        # Create submission entry
        submission = SubmissionEntry(
            submission_id=submission_id,
            team_name=team_name,
            email=email,
            algorithm_name=algorithm_name,
            description=description,
            submission_time=datetime.now(),
            file_hash=file_hash,
            file_path=file_path,
            libraries=libraries,
            status='pending'
        )
        
        # Add to database
        if self.database.add_submission(submission):
            return submission_id
        else:
            raise Exception("Failed to save submission to database")
    
    def evaluate_submission_async(self, submission_id: str) -> bool:
        """
        Start asynchronous evaluation of a submission.
        
        In a real implementation, this would queue the evaluation
        for processing by a background worker.
        """
        if not self.evaluation_engine:
            return False
        
        submission = self.database.get_submission(submission_id)
        if not submission:
            return False
        
        # Update status to evaluating
        self.database.update_submission_status(submission_id, 'evaluating')
        
        try:
            # Run evaluation (in production, this would be async)
            success, results = self.evaluation_engine.evaluate_submission(submission)
            
            if success:
                # Save evaluation results
                leaderboard_scores = results.get('leaderboard_scores', {})
                self.database.update_submission_status(
                    submission_id, 'completed', results, leaderboard_scores
                )
                
                # Save detailed evaluations
                detailed_results = results.get('detailed_results', {})
                for stream_id, stream_results in detailed_results.items():
                    for test_key, test_result in stream_results.items():
                        if 'error' not in test_result:
                            evaluation = EvaluationResult(
                                submission_id=submission_id,
                                stream_id=stream_id,
                                num_cases=test_result['num_cases'],
                                mae=test_result['mae'],
                                rmse=test_result['rmse'],
                                accuracy=test_result['accuracy'],
                                global_errors=test_result['global_errors'],
                                avg_processing_time=test_result['avg_processing_time'],
                                total_runtime=test_result['total_runtime'],
                                conformance_scores=test_result['conformance_scores'],
                                baseline_scores=test_result['baseline_scores'],
                                timestamp=datetime.fromisoformat(test_result['timestamp'])
                            )
                            self.database.add_evaluation(evaluation)
                
                return True
            else:
                # Evaluation failed
                self.database.update_submission_status(submission_id, 'failed', results)
                return False
                
        except Exception as e:
            # Evaluation error
            error_results = {"error": str(e)}
            self.database.update_submission_status(submission_id, 'failed', error_results)
            return False
    
    def _calculate_file_hash(self, file_path: str) -> str:
        """Calculate hash of uploaded file for deduplication."""
        try:
            path = Path(file_path)
            if path.is_file():
                with open(path, 'rb') as f:
                    content = f.read()
            else:
                # For directories, hash all Python files
                content = b""
                for py_file in path.rglob('*.py'):
                    with open(py_file, 'rb') as f:
                        content += f.read()
            
            return hashlib.sha256(content).hexdigest()
        except Exception:
            return "unknown"


# Export main classes
__all__ = [
    'SubmissionEntry',
    'EvaluationResult', 
    'SubmissionDatabase',
    'EvaluationEngine',
    'LeaderboardManager',
    'SubmissionManager'
]