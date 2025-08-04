"""
Enhanced Database Manager for AVOCADO Challenge Platform
Handles database operations, migrations, and data validation.
"""

import sqlite3
import json
import hashlib
import uuid
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

class DatabaseManager:
    """Enhanced database manager with validation and migrations."""
    
    def __init__(self, db_path: str = "user_algorithms.db"):
        self.db_path = Path(db_path)
        self.current_version = 2  # Database schema version
        self._init_database()
        self._run_migrations()
    
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
                    UNIQUE(email)
                )
            """)
            
            # Algorithms table with enhanced fields
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
    
    def create_or_update_user(self, token: str, email: str, institution: str = None) -> bool:
        """Create new user or update existing user info."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("PRAGMA foreign_keys = ON")
                
                # Check if user exists
                cursor = conn.execute("SELECT token, total_uploads, total_tests FROM users WHERE token = ?", (token,))
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
                else:
                    # Create new user
                    conn.execute("""
                        INSERT INTO users (token, email, institution, created_at, last_active)
                        VALUES (?, ?, ?, ?, ?)
                    """, (token, email, institution, datetime.now().isoformat(), datetime.now().isoformat()))
                
                logger.info(f"User {'updated' if existing else 'created'}: {email}")
                return True
                
        except sqlite3.IntegrityError as e:
            if "UNIQUE constraint failed" in str(e):
                logger.warning(f"Email already exists: {email}")
            return False
        except Exception as e:
            logger.error(f"Error creating/updating user: {e}")
            return False
    
    def add_algorithm(self, token: str, algorithm_name: str, filename: str, 
                     file_path: str, file_hash: str, description: str = None,
                     required_libraries: List[str] = None, file_size: int = 0) -> Optional[int]:
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
                        created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (token, algorithm_name, filename, file_path, file_hash, 
                      now, description, libraries_json, file_size, now, now))
                
                algorithm_id = cursor.lastrowid
                
                # Update user statistics
                conn.execute("""
                    UPDATE users SET 
                        total_uploads = total_uploads + 1,
                        last_active = ?
                    WHERE token = ?
                """, (now, token))
                
                logger.info(f"Algorithm added: {algorithm_name} (ID: {algorithm_id})")
                return algorithm_id
                
        except Exception as e:
            logger.error(f"Error adding algorithm: {e}")
            return None
    
    def get_user_algorithms(self, token: str, limit: int = None) -> List[Algorithm]:
        """Get algorithms for a user with enhanced information."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                query = """
                    SELECT id, algorithm_name, filename, file_path, file_hash,
                           upload_time, status, last_tested, test_results,
                           description, required_libraries, file_size, test_count,
                           created_at, updated_at
                    FROM algorithms 
                    WHERE token = ?
                    ORDER BY upload_time DESC
                """
                params = [token]
                
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
                        test_count=row[12] or 0
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
                           description, required_libraries, file_size, test_count
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
                    test_count=row[12] or 0
                )
                
        except Exception as e:
            logger.error(f"Error getting algorithm: {e}")
            return None
    
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
                    
                    # Update user test count
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
    
    def delete_algorithm(self, algorithm_id: int, token: str) -> bool:
        """Delete an algorithm and all related data."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("PRAGMA foreign_keys = ON")
                
                # Verify ownership
                cursor = conn.execute("SELECT id FROM algorithms WHERE id = ? AND token = ?", 
                                    (algorithm_id, token))
                if not cursor.fetchone():
                    logger.warning(f"Algorithm not found or unauthorized: {algorithm_id}")
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
                
                # Algorithm statistics
                cursor = conn.execute("""
                    SELECT 
                        COUNT(*) as total_algorithms,
                        COUNT(CASE WHEN status = 'tested' THEN 1 END) as tested_algorithms,
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
                    'last_upload': datetime.fromisoformat(alg_stats[2]) if alg_stats[2] else None,
                    'total_file_size': alg_stats[3] or 0,
                    'total_test_runs': alg_stats[4] or 0,
                    'recent_tests': recent_tests
                }
                
        except Exception as e:
            logger.error(f"Error getting user statistics: {e}")
            return {}

    def update_algorithm_test_results(self, algorithm_id: int, token: str,
                                      test_results: Dict[str, Any]) -> bool:
        """Update algorithm with test results."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("PRAGMA foreign_keys = ON")

                # Verify ownership
                cursor = conn.execute("SELECT id FROM algorithms WHERE id = ? AND token = ?",
                                      (algorithm_id, token))
                if not cursor.fetchone():
                    logger.warning(f"Algorithm not found or unauthorized: {algorithm_id}")
                    return False

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

                # Update user test count
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
            tested_algorithms = len([alg for alg in algorithms if alg.status == 'tested'])

            if tested_algorithms == 0:
                return {
                    'total_algorithms': total_algorithms,
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
                        'test_date': alg.last_tested.isoformat() if alg.last_tested else None
                    })

            # Sort by test date
            performance_trend.sort(key=lambda x: x['test_date'] or '')

            return {
                'total_algorithms': total_algorithms,
                'tested_algorithms': tested_algorithms,
                'best_score': max(scores) if scores else 0.0,
                'avg_score': sum(scores) / len(scores) if scores else 0.0,
                'performance_trend': performance_trend
            }

        except Exception as e:
            logger.error(f"Error getting performance summary: {e}")
            return {
                'total_algorithms': 0,
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
                
                # Identify inactive users (no activity in specified days)
                cursor = conn.execute("""
                    SELECT token FROM users 
                    WHERE last_active < ? AND total_uploads = 0
                """, (cutoff_date,))
                inactive_tokens = [row[0] for row in cursor.fetchall()]
                
                # Delete inactive users with no uploads
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
                        'test_results': alg.test_results
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