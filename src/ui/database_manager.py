"""
Enhanced Database Manager for AVOCADO Challenge Platform
Handles database operations, migrations, data validation, and default algorithm seeding.
"""

import sqlite3
import json
import hashlib
import uuid
import shutil
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class User:
    """Data class for user information."""
    token: str
    email: str
    institution: Optional[str] = None
    created_at: Optional[datetime] = None
    last_active: Optional[datetime] = None
    total_uploads: int = 0
    total_tests: int = 0

@dataclass
class Algorithm:
    """Data class for algorithm information."""
    id: Optional[int] = None
    token: str = ""
    algorithm_name: str = ""
    filename: str = ""
    file_path: str = ""
    file_hash: str = ""
    upload_time: Optional[datetime] = None
    status: str = "uploaded"
    last_tested: Optional[datetime] = None
    test_results: Optional[Dict[str, Any]] = None
    description: Optional[str] = None
    required_libraries: Optional[List[str]] = None
    file_size: int = 0
    test_count: int = 0
    is_default: bool = False  # New field to mark default algorithms

class DatabaseManager:
    """Enhanced database manager with default algorithm seeding."""

    def __init__(self, db_path: str = "user_algorithms.db"):
        self.db_path = Path(db_path)
        self.current_version = 3  # Incremented for new default algorithm feature

        # Set up default algorithms directory
        self.setup_default_algorithms_directory()

        self._init_database()
        self._run_migrations()

    def setup_default_algorithms_directory(self):
        """Set up directory structure for default algorithms."""
        # Create default algorithms directory
        self.default_algorithms_dir = Path("default_algorithms")
        self.default_algorithms_dir.mkdir(exist_ok=True)

        # Copy example algorithm to default directory if not exists
        example_algo_path = self.default_algorithms_dir / "example_algorithm.py"
        if not example_algo_path.exists():
            self._create_example_algorithm_file(example_algo_path)

    def _create_example_algorithm_file(self, file_path: Path):
        """Create the example algorithm file in the default directory."""
        example_algorithm_content = '''"""
Example Algorithm Implementation for Streaming Process Mining Challenge

This file demonstrates how to implement a conformance checking algorithm
that can be uploaded to the challenge platform.

Required libraries:
pandas
numpy

Author: AVOCADO Platform
Description: Default example algorithms for all users
"""

import sys
from collections import defaultdict, Counter
from typing import Dict, Any, Set, Tuple, Optional
import numpy as np

# Note: BaseAlgorithm will be imported automatically by the platform
from algorithm_base import BaseAlgorithm


class FrequencyBasedConformanceAlgorithm(BaseAlgorithm):
    """
    A frequency-based conformance checking algorithm.
    
    This algorithm learns the frequency of activity transitions during the learning phase
    and uses this information to assess conformance during the checking phase.
    Transitions that occur more frequently during learning are considered more conformant.
    """
    
    def __init__(self):
        """Initialize the algorithm."""
        super().__init__()
        
        # Learning phase data structures
        self._transitions = defaultdict(int)  # Count of each transition
        self._activity_counts = defaultdict(int)  # Count of each activity
        self._case_traces = defaultdict(list)  # Store traces during learning
        
        # Conformance checking data structures
        self._transition_probabilities = {}  # Learned transition probabilities
        self._activity_probabilities = {}  # Learned activity probabilities
        self._case_last_activity = {}  # Track last activity per case
        self._case_conformance_scores = defaultdict(list)  # Store conformance scores per case
        
        # Configuration
        self.min_frequency_threshold = 2  # Minimum frequency to consider a transition valid
        self.smoothing_factor = 0.1  # Laplace smoothing factor
        
        # Constants for event field names
        self.CASE_ID_KEY = 'case:concept:name'
        self.ACTIVITY_KEY = 'concept:name'
    
    def learn(self, event: Dict[str, Any]) -> None:
        """
        Learn patterns from the event during the learning phase.
        
        Args:
            event: Dictionary containing event data with keys:
                  - 'case:concept:name': Case identifier
                  - 'concept:name': Activity name
                  - Additional attributes may be present
        """
        self.learning_events += 1
        
        case_id = event.get(self.CASE_ID_KEY)
        activity = event.get(self.ACTIVITY_KEY)
        
        if not case_id or not activity:
            # Skip events with missing essential information
            return
        
        # Track activity frequency
        self._activity_counts[activity] += 1
        
        # Track transitions
        if case_id in self._case_traces and self._case_traces[case_id]:
            # There's a previous activity in this case
            prev_activity = self._case_traces[case_id][-1]
            transition = (prev_activity, activity)
            self._transitions[transition] += 1
        
        # Add activity to case trace
        self._case_traces[case_id].append(activity)
    
    def conformance(self, event: Dict[str, Any]) -> float:
        """
        Calculate conformance score for the given event.
        
        Args:
            event: Dictionary containing event data
            
        Returns:
            Float between 0.0 and 1.0 representing conformance score
        """
        self.conformance_events += 1
        
        case_id = event.get(self.CASE_ID_KEY)
        activity = event.get(self.ACTIVITY_KEY)
        
        if not case_id or not activity:
            return 0.5  # Neutral score for missing data
        
        # Calculate conformance score
        conformance_score = self._calculate_conformance_score(case_id, activity)
        
        # Update tracking for next event
        self._case_last_activity[case_id] = activity
        self._case_conformance_scores[case_id].append(conformance_score)
        
        return conformance_score
    
    def _calculate_conformance_score(self, case_id: str, activity: str) -> float:
        """
        Calculate conformance score based on learned patterns.
        
        Args:
            case_id: Case identifier
            activity: Current activity name
            
        Returns:
            Conformance score between 0.0 and 1.0
        """
        # For the first activity in a case, use activity frequency
        if case_id not in self._case_last_activity:
            return self._get_activity_conformance(activity)
        
        # For subsequent activities, use transition probability
        prev_activity = self._case_last_activity[case_id]
        transition = (prev_activity, activity)
        
        return self._get_transition_conformance(transition, activity)
    
    def _get_activity_conformance(self, activity: str) -> float:
        """Get conformance score based on activity frequency."""
        if activity in self._activity_probabilities:
            # Use learned probability
            return self._activity_probabilities[activity]
        else:
            # Unknown activity - low conformance
            return 0.1
    
    def _get_transition_conformance(self, transition: Tuple[str, str], activity: str) -> float:
        """Get conformance score based on transition probability."""
        if transition in self._transition_probabilities:
            # Use learned transition probability
            return self._transition_probabilities[transition]
        else:
            # Unknown transition - use activity frequency as fallback
            return self._get_activity_conformance(activity) * 0.5
    
    def on_learning_phase_end(self, stream_info: Dict[str, Any]):
        """
        Called when learning phase ends. Calculate probabilities from learned data.
        
        Args:
            stream_info: Information about the learning phase
        """
        super().on_learning_phase_end(stream_info)
        self._calculate_probabilities()
    
    def _calculate_probabilities(self):
        """Calculate transition and activity probabilities from learned data."""
        # Calculate activity probabilities
        total_activities = sum(self._activity_counts.values())
        if total_activities > 0:
            for activity, count in self._activity_counts.items():
                # Apply Laplace smoothing
                prob = (count + self.smoothing_factor) / (total_activities + self.smoothing_factor * len(self._activity_counts))
                self._activity_probabilities[activity] = min(1.0, prob * 2)  # Scale up for better discrimination
        
        # Calculate transition probabilities
        total_transitions = sum(self._transitions.values())
        if total_transitions > 0:
            # Group transitions by source activity
            source_totals = defaultdict(int)
            for (source, target), count in self._transitions.items():
                source_totals[source] += count
            
            # Calculate conditional probabilities
            for (source, target), count in self._transitions.items():
                if count >= self.min_frequency_threshold:
                    # P(target|source)
                    prob = count / source_totals[source] if source_totals[source] > 0 else 0
                    self._transition_probabilities[(source, target)] = min(1.0, prob)
    
    def get_algorithm_info(self) -> Dict[str, Any]:
        """
        Get information about the algorithm and its current state.
        
        Returns:
            Dictionary with algorithm information
        """
        return {
            'name': 'FrequencyBasedConformanceAlgorithm',
            'description': 'Frequency-based conformance checking using transition probabilities',
            'learned_activities': len(self._activity_counts),
            'learned_transitions': len(self._transitions),
            'valid_transitions': len(self._transition_probabilities),
            'min_frequency_threshold': self.min_frequency_threshold,
            'smoothing_factor': self.smoothing_factor
        }
    
    def reset(self):
        """Reset the algorithm to initial state."""
        super().reset_metrics()
        
        # Clear learning data
        self._transitions.clear()
        self._activity_counts.clear()
        self._case_traces.clear()
        
        # Clear conformance data
        self._transition_probabilities.clear()
        self._activity_probabilities.clear()
        self._case_last_activity.clear()
        self._case_conformance_scores.clear()


class SimpleBaselineAlgorithm(BaseAlgorithm):
    """
    A simple baseline algorithm that returns random conformance scores.
    
    This serves as a basic example and baseline for comparison.
    """
    
    def __init__(self):
        """Initialize the baseline algorithm."""
        super().__init__()
        self._random_seed = 42
        self._activity_seen = set()
        
    def learn(self, event: Dict[str, Any]) -> None:
        """Learn from events by simply tracking seen activities."""
        self.learning_events += 1
        
        activity = event.get('concept:name')
        if activity:
            self._activity_seen.add(activity)
    
    def conformance(self, event: Dict[str, Any]) -> float:
        """Return simple conformance based on whether activity was seen in learning."""
        self.conformance_events += 1
        
        activity = event.get('concept:name')
        if not activity:
            return 0.5
        
        # Simple rule: if activity was seen during learning, return high conformance
        if activity in self._activity_seen:
            return 0.8
        else:
            return 0.2
    
    def get_algorithm_info(self) -> Dict[str, Any]:
        """Get algorithm information."""
        return {
            'name': 'SimpleBaselineAlgorithm',
            'description': 'Simple baseline algorithm for demonstration',
            'activities_learned': len(self._activity_seen)
        }


# The platform will automatically detect and load one of these algorithm classes
# Make sure at least one class inherits from BaseAlgorithm and implements the required methods
'''

        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(example_algorithm_content)

        logger.info(f"Created example algorithm file at {file_path}")

    def _init_database(self):
        """Initialize database with enhanced schema."""
        with sqlite3.connect(self.db_path) as conn:
            # Enable foreign key constraints
            conn.execute("PRAGMA foreign_keys = ON")

            # Users table with enhanced fields
            conn.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    token TEXT PRIMARY KEY,
                    email TEXT NOT NULL,
                    institution TEXT,
                    created_at TEXT NOT NULL,
                    last_active TEXT NOT NULL,
                    total_uploads INTEGER DEFAULT 0,
                    total_tests INTEGER DEFAULT 0,
                    settings TEXT DEFAULT '{}',
                    defaults_seeded BOOLEAN DEFAULT FALSE,
                    UNIQUE(email)
                )
            """)

            # Algorithms table with enhanced fields (including is_default)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS algorithms (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    token TEXT NOT NULL,
                    algorithm_name TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    file_hash TEXT NOT NULL,
                    upload_time TEXT NOT NULL,
                    status TEXT DEFAULT 'uploaded',
                    last_tested TEXT,
                    test_results TEXT,
                    description TEXT,
                    required_libraries TEXT,
                    file_size INTEGER DEFAULT 0,
                    test_count INTEGER DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    is_default BOOLEAN DEFAULT FALSE,
                    FOREIGN KEY (token) REFERENCES users (token) ON DELETE CASCADE
                )
            """)

            # Test sessions table for detailed tracking
            conn.execute("""
                CREATE TABLE IF NOT EXISTS test_sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    algorithm_id INTEGER NOT NULL,
                    stream_id TEXT NOT NULL,
                    num_cases INTEGER NOT NULL,
                    test_start TEXT NOT NULL,
                    test_end TEXT,
                    status TEXT DEFAULT 'running',
                    results TEXT,
                    error_message TEXT,
                    performance_metrics TEXT,
                    FOREIGN KEY (algorithm_id) REFERENCES algorithms (id) ON DELETE CASCADE
                )
            """)

            # User sessions for analytics
            conn.execute("""
                CREATE TABLE IF NOT EXISTS user_sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    token TEXT NOT NULL,
                    session_start TEXT NOT NULL,
                    session_end TEXT,
                    ip_address TEXT,
                    user_agent TEXT,
                    actions_count INTEGER DEFAULT 0,
                    FOREIGN KEY (token) REFERENCES users (token) ON DELETE CASCADE
                )
            """)

            # Database metadata
            conn.execute("""
                CREATE TABLE IF NOT EXISTS db_metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)

            # Create indexes for better performance
            conn.execute("CREATE INDEX IF NOT EXISTS idx_algorithms_token ON algorithms (token)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_algorithms_status ON algorithms (status)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_algorithms_upload_time ON algorithms (upload_time)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_algorithms_is_default ON algorithms (is_default)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_test_sessions_algorithm ON test_sessions (algorithm_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_users_email ON users (email)")

            # Set initial database version
            conn.execute("""
                INSERT OR REPLACE INTO db_metadata (key, value, updated_at)
                VALUES ('schema_version', ?, ?)
            """, (str(self.current_version), datetime.now().isoformat()))

            logger.info("Database initialized successfully")

    def _run_migrations(self):
        """Run database migrations if needed."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT value FROM db_metadata WHERE key = 'schema_version'")
            result = cursor.fetchone()
            current_version = int(result[0]) if result else 1

            if current_version < self.current_version:
                logger.info(f"Running migrations from version {current_version} to {self.current_version}")

                if current_version < 2:
                    self._migrate_to_v2(conn)

                if current_version < 3:
                    self._migrate_to_v3(conn)

                # Update version
                conn.execute("""
                    UPDATE db_metadata SET value = ?, updated_at = ?
                    WHERE key = 'schema_version'
                """, (str(self.current_version), datetime.now().isoformat()))

                logger.info("Migrations completed successfully")

    def _migrate_to_v2(self, conn):
        """Migrate to version 2 - add enhanced fields."""
        try:
            # Add new columns to existing tables if they don't exist
            columns_to_add = [
                ("users", "total_uploads", "INTEGER DEFAULT 0"),
                ("users", "total_tests", "INTEGER DEFAULT 0"),
                ("users", "settings", "TEXT DEFAULT '{}'"),
                ("algorithms", "description", "TEXT"),
                ("algorithms", "required_libraries", "TEXT"),
                ("algorithms", "file_size", "INTEGER DEFAULT 0"),
                ("algorithms", "test_count", "INTEGER DEFAULT 0"),
                ("algorithms", "created_at", "TEXT"),
                ("algorithms", "updated_at", "TEXT")
            ]

            for table, column, definition in columns_to_add:
                try:
                    conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")
                except sqlite3.OperationalError:
                    # Column already exists
                    pass

            # Update existing records with default values
            now = datetime.now().isoformat()
            conn.execute(f"UPDATE algorithms SET created_at = upload_time WHERE created_at IS NULL")
            conn.execute(f"UPDATE algorithms SET updated_at = ? WHERE updated_at IS NULL", (now,))

            logger.info("Migration to v2 completed")

        except Exception as e:
            logger.error(f"Migration to v2 failed: {e}")
            raise

    def _migrate_to_v3(self, conn):
        """Migrate to version 3 - add default algorithm support."""
        try:
            # Add new columns for default algorithm support
            columns_to_add = [
                ("users", "defaults_seeded", "BOOLEAN DEFAULT FALSE"),
                ("algorithms", "is_default", "BOOLEAN DEFAULT FALSE")
            ]

            for table, column, definition in columns_to_add:
                try:
                    conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")
                except sqlite3.OperationalError:
                    # Column already exists
                    pass

            # Create index for is_default
            try:
                conn.execute("CREATE INDEX IF NOT EXISTS idx_algorithms_is_default ON algorithms (is_default)")
            except sqlite3.OperationalError:
                pass

            logger.info("Migration to v3 completed")

        except Exception as e:
            logger.error(f"Migration to v3 failed: {e}")
            raise

    def create_or_update_user(self, token: str, email: str, institution: str = None) -> bool:
        """Create new user or update existing user info, and seed default algorithms."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("PRAGMA foreign_keys = ON")

                # Check if user exists
                cursor = conn.execute("SELECT token, total_uploads, total_tests, defaults_seeded FROM users WHERE token = ?", (token,))
                existing = cursor.fetchone()

                if existing:
                    # Update existing user
                    conn.execute("""
                        UPDATE users SET 
                            email = ?, 
                            institution = ?, 
                            last_active = ?
                        WHERE token = ?
                    """, (email, institution, datetime.now().isoformat(), token))

                    # Check if defaults need to be seeded
                    defaults_seeded = existing[3] if len(existing) > 3 else False
                    if not defaults_seeded:
                        self._seed_default_algorithms_for_user(token, conn)
                else:
                    # Create new user
                    conn.execute("""
                        INSERT INTO users (token, email, institution, created_at, last_active, defaults_seeded)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, (token, email, institution, datetime.now().isoformat(), datetime.now().isoformat(), False))

                    # Seed default algorithms for new user
                    self._seed_default_algorithms_for_user(token, conn)

                logger.info(f"User {'updated' if existing else 'created'}: {email}")
                return True

        except sqlite3.IntegrityError as e:
            if "UNIQUE constraint failed" in str(e):
                logger.warning(f"Email already exists: {email}")
            return False
        except Exception as e:
            logger.error(f"Error creating/updating user: {e}")
            return False

    def _seed_default_algorithms_for_user(self, token: str, conn):
        """Seed default algorithms for a user."""
        try:
            # Check if defaults already seeded
            cursor = conn.execute("SELECT COUNT(*) FROM algorithms WHERE token = ? AND is_default = TRUE", (token,))
            if cursor.fetchone()[0] > 0:
                return  # Already seeded

            # Define default algorithms
            default_algorithms = [
                {
                    'algorithm_name': 'Frequency-Based Conformance Checker',
                    'filename': 'example_frequency_algorithm.py',
                    'description': 'A frequency-based conformance checking algorithm that learns transition patterns during the learning phase and assesses conformance based on learned probabilities. This is a great starting point for understanding the platform.',
                    'required_libraries': ['numpy', 'pandas']
                },
                {
                    'algorithm_name': 'Simple Baseline Algorithm',
                    'filename': 'example_baseline_algorithm.py',
                    'description': 'A simple baseline algorithm that demonstrates the basic structure required for the platform. Returns conformance scores based on whether activities were seen during learning.',
                    'required_libraries': []
                }
            ]

            # Create user-specific algorithm directory
            user_algo_dir = Path("uploaded_algorithms") / token
            user_algo_dir.mkdir(parents=True, exist_ok=True)

            now = datetime.now().isoformat()

            for algo_info in default_algorithms:
                # Copy algorithm file to user directory
                user_algo_file = user_algo_dir / algo_info['filename']
                shutil.copy2(self.default_algorithms_dir / "example_algorithm.py", user_algo_file)

                # Calculate file hash
                file_hash = self._calculate_file_hash(user_algo_file)
                file_size = user_algo_file.stat().st_size

                # Insert into database
                conn.execute("""
                    INSERT INTO algorithms (
                        token, algorithm_name, filename, file_path, file_hash,
                        upload_time, description, required_libraries, file_size,
                        created_at, updated_at, is_default
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    token, algo_info['algorithm_name'], algo_info['filename'],
                    str(user_algo_file), file_hash, now, algo_info['description'],
                    json.dumps(algo_info['required_libraries']), file_size,
                    now, now, True
                ))

            # Mark user as seeded
            conn.execute("UPDATE users SET defaults_seeded = TRUE WHERE token = ?", (token,))

            logger.info(f"Seeded {len(default_algorithms)} default algorithms for user {token}")

        except Exception as e:
            logger.error(f"Error seeding default algorithms for user {token}: {e}")

    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate hash of file for deduplication."""
        try:
            with open(file_path, 'rb') as f:
                content = f.read()
            return hashlib.sha256(content).hexdigest()
        except Exception:
            return "unknown"

    def add_algorithm(self, token: str, algorithm_name: str, filename: str,
                     file_path: str, file_hash: str, description: str = None,
                     required_libraries: List[str] = None, file_size: int = 0,
                     is_default: bool = False) -> Optional[int]:
        """Add algorithm to database with enhanced information."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("PRAGMA foreign_keys = ON")

                now = datetime.now().isoformat()
                libraries_json = json.dumps(required_libraries) if required_libraries else None

                cursor = conn.execute("""
                    INSERT INTO algorithms (
                        token, algorithm_name, filename, file_path, file_hash,
                        upload_time, description, required_libraries, file_size,
                        created_at, updated_at, is_default
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (token, algorithm_name, filename, file_path, file_hash,
                      now, description, libraries_json, file_size, now, now, is_default))

                algorithm_id = cursor.lastrowid

                # Update user statistics (only for non-default algorithms)
                if not is_default:
                    conn.execute("""
                        UPDATE users SET 
                            total_uploads = total_uploads + 1,
                            last_active = ?
                        WHERE token = ?
                    """, (now, token))

                logger.info(f"Algorithm added: {algorithm_name} (ID: {algorithm_id}, Default: {is_default})")
                return algorithm_id

        except Exception as e:
            logger.error(f"Error adding algorithm: {e}")
            return None

    def get_user_algorithms(self, token: str, limit: int = None, include_defaults: bool = True) -> List[Algorithm]:
        """Get algorithms for a user with enhanced information."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                query = """
                    SELECT id, algorithm_name, filename, file_path, file_hash,
                           upload_time, status, last_tested, test_results,
                           description, required_libraries, file_size, test_count,
                           created_at, updated_at, is_default
                    FROM algorithms 
                    WHERE token = ?
                """
                params = [token]

                if not include_defaults:
                    query += " AND is_default = FALSE"

                query += " ORDER BY is_default DESC, upload_time DESC"

                if limit:
                    query += " LIMIT ?"
                    params.append(limit)

                cursor = conn.execute(query, params)

                algorithms = []
                for row in cursor.fetchall():
                    required_libs = json.loads(row[10]) if row[10] else None
                    test_results = json.loads(row[8]) if row[8] else None

                    algorithm = Algorithm(
                        id=row[0],
                        token=token,
                        algorithm_name=row[1],
                        filename=row[2],
                        file_path=row[3],
                        file_hash=row[4],
                        upload_time=datetime.fromisoformat(row[5]),
                        status=row[6],
                        last_tested=datetime.fromisoformat(row[7]) if row[7] else None,
                        test_results=test_results,
                        description=row[9],
                        required_libraries=required_libs,
                        file_size=row[11] or 0,
                        test_count=row[12] or 0,
                        is_default=bool(row[15]) if len(row) > 15 else False
                    )
                    algorithms.append(algorithm)

                return algorithms

        except Exception as e:
            logger.error(f"Error getting user algorithms: {e}")
            return []

    def get_algorithm_by_id(self, algorithm_id: int, token: str) -> Optional[Algorithm]:
        """Get specific algorithm by ID and token."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT id, algorithm_name, filename, file_path, file_hash,
                           upload_time, status, last_tested, test_results,
                           description, required_libraries, file_size, test_count, is_default
                    FROM algorithms 
                    WHERE id = ? AND token = ?
                """, (algorithm_id, token))

                row = cursor.fetchone()
                if not row:
                    return None

                required_libs = json.loads(row[10]) if row[10] else None
                test_results = json.loads(row[8]) if row[8] else None

                return Algorithm(
                    id=row[0],
                    token=token,
                    algorithm_name=row[1],
                    filename=row[2],
                    file_path=row[3],
                    file_hash=row[4],
                    upload_time=datetime.fromisoformat(row[5]),
                    status=row[6],
                    last_tested=datetime.fromisoformat(row[7]) if row[7] else None,
                    test_results=test_results,
                    description=row[9],
                    required_libraries=required_libs,
                    file_size=row[11] or 0,
                    test_count=row[12] or 0,
                    is_default=bool(row[13]) if len(row) > 13 else False
                )

        except Exception as e:
            logger.error(f"Error getting algorithm: {e}")
            return None

    def delete_algorithm(self, algorithm_id: int, token: str) -> bool:
        """Delete an algorithm and all related data (prevents deletion of defaults)."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("PRAGMA foreign_keys = ON")

                # Check if algorithm exists and is not default
                cursor = conn.execute("""
                    SELECT id, is_default FROM algorithms 
                    WHERE id = ? AND token = ?
                """, (algorithm_id, token))
                result = cursor.fetchone()

                if not result:
                    logger.warning(f"Algorithm not found or unauthorized: {algorithm_id}")
                    return False

                is_default = bool(result[1]) if len(result) > 1 else False
                if is_default:
                    logger.warning(f"Cannot delete default algorithm: {algorithm_id}")
                    return False

                # Delete algorithm (cascades to test_sessions)
                conn.execute("DELETE FROM algorithms WHERE id = ?", (algorithm_id,))

                # Update user statistics
                conn.execute("""
                    UPDATE users SET 
                        total_uploads = total_uploads - 1,
                        last_active = ?
                    WHERE token = ?
                """, (datetime.now().isoformat(), token))

                logger.info(f"Algorithm deleted: {algorithm_id}")
                return True

        except Exception as e:
            logger.error(f"Error deleting algorithm: {e}")
            return False

    def force_seed_defaults_for_user(self, token: str) -> bool:
        """Force seed default algorithms for a specific user (for testing/admin)."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("PRAGMA foreign_keys = ON")

                # Remove existing defaults
                conn.execute("DELETE FROM algorithms WHERE token = ? AND is_default = TRUE", (token,))

                # Reset seeded flag
                conn.execute("UPDATE users SET defaults_seeded = FALSE WHERE token = ?", (token,))

                # Seed defaults
                self._seed_default_algorithms_for_user(token, conn)

                return True

        except Exception as e:
            logger.error(f"Error force seeding defaults for {token}: {e}")
            return False

    # ... (keep all other existing methods from the original file)

    def start_test_session(self, algorithm_id: int, stream_id: str, num_cases: int) -> Optional[int]:
        """Start a new test session."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    INSERT INTO test_sessions (algorithm_id, stream_id, num_cases, test_start)
                    VALUES (?, ?, ?, ?)
                """, (algorithm_id, stream_id, num_cases, datetime.now().isoformat()))

                session_id = cursor.lastrowid
                logger.info(f"Test session started: {session_id}")
                return session_id

        except Exception as e:
            logger.error(f"Error starting test session: {e}")
            return None

    def complete_test_session(self, session_id: int, results: Dict[str, Any],
                            performance_metrics: Dict[str, Any] = None) -> bool:
        """Complete a test session with results."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Update test session
                conn.execute("""
                    UPDATE test_sessions SET 
                        test_end = ?,
                        status = 'completed',
                        results = ?,
                        performance_metrics = ?
                    WHERE id = ?
                """, (datetime.now().isoformat(), json.dumps(results),
                      json.dumps(performance_metrics) if performance_metrics else None, session_id))

                # Get algorithm ID
                cursor = conn.execute("SELECT algorithm_id FROM test_sessions WHERE id = ?", (session_id,))
                result = cursor.fetchone()
                if result:
                    algorithm_id = result[0]

                    # Update algorithm with latest test results
                    conn.execute("""
                        UPDATE algorithms SET 
                            status = 'tested',
                            last_tested = ?,
                            test_results = ?,
                            test_count = test_count + 1,
                            updated_at = ?
                        WHERE id = ?
                    """, (datetime.now().isoformat(), json.dumps(results),
                          datetime.now().isoformat(), algorithm_id))

                    # Update user test count (only for non-default algorithms)
                    cursor = conn.execute("SELECT is_default FROM algorithms WHERE id = ?", (algorithm_id,))
                    algo_result = cursor.fetchone()
                    is_default = bool(algo_result[0]) if algo_result and len(algo_result) > 0 else False

                    if not is_default:
                        conn.execute("""
                            UPDATE users SET 
                                total_tests = total_tests + 1,
                                last_active = ?
                            WHERE token = (SELECT token FROM algorithms WHERE id = ?)
                        """, (datetime.now().isoformat(), algorithm_id))

                logger.info(f"Test session completed: {session_id}")
                return True

        except Exception as e:
            logger.error(f"Error completing test session: {e}")
            return False

    def fail_test_session(self, session_id: int, error_message: str) -> bool:
        """Mark a test session as failed."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    UPDATE test_sessions SET 
                        test_end = ?,
                        status = 'failed',
                        error_message = ?
                    WHERE id = ?
                """, (datetime.now().isoformat(), error_message, session_id))

                logger.info(f"Test session failed: {session_id}")
                return True

        except Exception as e:
            logger.error(f"Error failing test session: {e}")
            return False

    def update_algorithm_test_results(self, algorithm_id: int, token: str,
                                      test_results: Dict[str, Any]) -> bool:
        """Update algorithm with test results."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("PRAGMA foreign_keys = ON")

                # Verify ownership
                cursor = conn.execute("SELECT id, is_default FROM algorithms WHERE id = ? AND token = ?",
                                      (algorithm_id, token))
                result = cursor.fetchone()
                if not result:
                    logger.warning(f"Algorithm not found or unauthorized: {algorithm_id}")
                    return False

                is_default = bool(result[1]) if len(result) > 1 else False

                # Update algorithm with test results
                conn.execute("""
                             UPDATE algorithms
                             SET status       = 'tested',
                                 last_tested  = ?,
                                 test_results = ?,
                                 test_count   = test_count + 1,
                                 updated_at   = ?
                             WHERE id = ?
                               AND token = ?
                             """, (
                                 datetime.now().isoformat(),
                                 json.dumps(test_results),
                                 datetime.now().isoformat(),
                                 algorithm_id,
                                 token
                             ))

                # Update user test count (only for non-default algorithms)
                if not is_default:
                    conn.execute("""
                                 UPDATE users
                                 SET total_tests = total_tests + 1,
                                     last_active = ?
                                 WHERE token = ?
                                 """, (datetime.now().isoformat(), token))

                logger.info(f"Test results updated for algorithm {algorithm_id}")
                return True

        except Exception as e:
            logger.error(f"Error updating test results: {e}")
            return False

    def get_user_statistics(self, token: str) -> Dict[str, Any]:
        """Get comprehensive user statistics."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # User basic info
                cursor = conn.execute("""
                    SELECT email, institution, created_at, last_active, total_uploads, total_tests
                    FROM users WHERE token = ?
                """, (token,))
                user_info = cursor.fetchone()

                if not user_info:
                    return {}

                # Algorithm statistics (excluding defaults from upload counts)
                cursor = conn.execute("""
                    SELECT 
                        COUNT(*) as total_algorithms,
                        COUNT(CASE WHEN status = 'tested' THEN 1 END) as tested_algorithms,
                        COUNT(CASE WHEN is_default = FALSE THEN 1 END) as user_algorithms,
                        COUNT(CASE WHEN is_default = TRUE THEN 1 END) as default_algorithms,
                        MAX(upload_time) as last_upload,
                        SUM(file_size) as total_file_size,
                        SUM(test_count) as total_test_runs
                    FROM algorithms WHERE token = ?
                """, (token,))
                alg_stats = cursor.fetchone()

                # Recent test sessions
                cursor = conn.execute("""
                    SELECT COUNT(*) as recent_tests
                    FROM test_sessions ts
                    JOIN algorithms a ON ts.algorithm_id = a.id
                    WHERE a.token = ? AND ts.test_start > datetime('now', '-7 days')
                """, (token,))
                recent_tests = cursor.fetchone()[0]

                return {
                    'email': user_info[0],
                    'institution': user_info[1],
                    'member_since': datetime.fromisoformat(user_info[2]),
                    'last_active': datetime.fromisoformat(user_info[3]),
                    'total_uploads': user_info[4] or 0,
                    'total_tests': user_info[5] or 0,
                    'total_algorithms': alg_stats[0] or 0,
                    'tested_algorithms': alg_stats[1] or 0,
                    'user_algorithms': alg_stats[2] or 0,
                    'default_algorithms': alg_stats[3] or 0,
                    'last_upload': datetime.fromisoformat(alg_stats[4]) if alg_stats[4] else None,
                    'total_file_size': alg_stats[5] or 0,
                    'total_test_runs': alg_stats[6] or 0,
                    'recent_tests': recent_tests
                }

        except Exception as e:
            logger.error(f"Error getting user statistics: {e}")
            return {}

    def get_user_best_algorithm(self, token: str) -> Optional[Algorithm]:
        """Get user's best performing algorithm based on composite score."""
        try:
            algorithms = self.get_user_algorithms(token)

            best_algorithm = None
            best_score = -1.0

            for alg in algorithms:
                if alg.test_results and alg.status == 'tested':
                    metrics = alg.test_results.get('metrics', {})

                    # Calculate composite score (same logic as in AlgorithmExecutor)
                    score = self._calculate_composite_score_db(metrics)

                    if score > best_score:
                        best_score = score
                        best_algorithm = alg

            return best_algorithm

        except Exception as e:
            logger.error(f"Error getting best algorithm: {e}")
            return None

    def _calculate_composite_score_db(self, metrics: Dict[str, float]) -> float:
        """Calculate composite score for database operations."""
        weights = {
            'accuracy': 0.30,
            'mae': 0.25,
            'rmse': 0.20,
            'correlation': 0.15,
            'global_errors': 0.10
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

    def get_algorithm_performance_summary(self, token: str) -> Dict[str, Any]:
        """Get performance summary for user's algorithms."""
        try:
            algorithms = self.get_user_algorithms(token)

            total_algorithms = len(algorithms)
            user_algorithms = len([alg for alg in algorithms if not alg.is_default])
            tested_algorithms = len([alg for alg in algorithms if alg.status == 'tested'])

            if tested_algorithms == 0:
                return {
                    'total_algorithms': total_algorithms,
                    'user_algorithms': user_algorithms,
                    'tested_algorithms': 0,
                    'best_score': 0.0,
                    'avg_score': 0.0,
                    'performance_trend': []
                }

            scores = []
            performance_trend = []

            for alg in algorithms:
                if alg.test_results and alg.status == 'tested':
                    metrics = alg.test_results.get('metrics', {})
                    score = self._calculate_composite_score_db(metrics)
                    scores.append(score)

                    performance_trend.append({
                        'algorithm_name': alg.algorithm_name,
                        'score': score,
                        'test_date': alg.last_tested.isoformat() if alg.last_tested else None,
                        'is_default': alg.is_default
                    })

            # Sort by test date
            performance_trend.sort(key=lambda x: x['test_date'] or '')

            return {
                'total_algorithms': total_algorithms,
                'user_algorithms': user_algorithms,
                'tested_algorithms': tested_algorithms,
                'best_score': max(scores) if scores else 0.0,
                'avg_score': sum(scores) / len(scores) if scores else 0.0,
                'performance_trend': performance_trend
            }

        except Exception as e:
            logger.error(f"Error getting performance summary: {e}")
            return {
                'total_algorithms': 0,
                'user_algorithms': 0,
                'tested_algorithms': 0,
                'best_score': 0.0,
                'avg_score': 0.0,
                'performance_trend': []
            }

    def cleanup_old_data(self, days_old: int = 30) -> Dict[str, int]:
        """Clean up old test sessions and inactive users."""
        cleanup_stats = {'test_sessions': 0, 'inactive_users': 0}

        try:
            with sqlite3.connect(self.db_path) as conn:
                cutoff_date = (datetime.now() - timedelta(days=days_old)).isoformat()

                # Clean up old test sessions
                cursor = conn.execute("""
                    DELETE FROM test_sessions 
                    WHERE test_start < ? AND status IN ('completed', 'failed')
                """, (cutoff_date,))
                cleanup_stats['test_sessions'] = cursor.rowcount

                # Identify inactive users (no activity in specified days and no user uploads)
                cursor = conn.execute("""
                    SELECT token FROM users 
                    WHERE last_active < ? 
                    AND total_uploads = 0
                    AND (SELECT COUNT(*) FROM algorithms WHERE token = users.token AND is_default = FALSE) = 0
                """, (cutoff_date,))
                inactive_tokens = [row[0] for row in cursor.fetchall()]

                # Delete inactive users with no user-created algorithms
                if inactive_tokens:
                    placeholders = ','.join(['?' for _ in inactive_tokens])
                    conn.execute(f"DELETE FROM users WHERE token IN ({placeholders})", inactive_tokens)
                    cleanup_stats['inactive_users'] = len(inactive_tokens)

                logger.info(f"Cleanup completed: {cleanup_stats}")
                return cleanup_stats

        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
            return cleanup_stats

    def export_data(self, token: str, export_path: str = None) -> Optional[str]:
        """Export user data to JSON file."""
        try:
            user_stats = self.get_user_statistics(token)
            algorithms = self.get_user_algorithms(token)

            export_data = {
                'export_date': datetime.now().isoformat(),
                'user_statistics': user_stats,
                'algorithms': [
                    {
                        'algorithm_name': alg.algorithm_name,
                        'filename': alg.filename,
                        'upload_time': alg.upload_time.isoformat() if alg.upload_time else None,
                        'status': alg.status,
                        'description': alg.description,
                        'required_libraries': alg.required_libraries,
                        'test_count': alg.test_count,
                        'last_tested': alg.last_tested.isoformat() if alg.last_tested else None,
                        'test_results': alg.test_results,
                        'is_default': alg.is_default
                    }
                    for alg in algorithms
                ]
            }

            if not export_path:
                export_path = f"user_data_export_{token[:8]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

            with open(export_path, 'w') as f:
                json.dump(export_data, f, indent=2)

            logger.info(f"Data exported to: {export_path}")
            return export_path

        except Exception as e:
            logger.error(f"Error exporting data: {e}")
            return None


# Global database instance
db_manager = DatabaseManager()

# Export for use in main application
__all__ = ['DatabaseManager', 'User', 'Algorithm', 'db_manager']