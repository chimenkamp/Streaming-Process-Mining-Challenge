"""
File Upload Utilities for Streaming Process Mining Challenge

This module provides utilities for handling file uploads, validation,
and algorithm extraction for the challenge platform.
"""

import os
import zipfile
import tempfile
import shutil
import base64
import io
import ast
import inspect
import importlib.util
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Type, Tuple
import json

class AlgorithmUploadManager:
    """Manages algorithm file uploads and validation."""
    
    def __init__(self, upload_directory: str):
        """
        Initialize the upload manager.
        
        Args:
            upload_directory: Base directory for storing uploaded files
        """
        self.upload_directory = Path(upload_directory)
        self.upload_directory.mkdir(exist_ok=True)
        
    def save_uploaded_file(self, contents: str, filename: str, session_id: str) -> Optional[str]:
        """
        Save uploaded file content to disk.
        
        Args:
            contents: Base64 encoded file content
            filename: Original filename
            session_id: Unique session identifier
            
        Returns:
            Path to saved file/directory or None if failed
        """
        if not contents:
            return None
            
        try:
            # Decode the file content
            content_type, content_string = contents.split(',')
            decoded = base64.b64decode(content_string)
            
            # Create session directory
            session_dir = self.upload_directory / session_id
            session_dir.mkdir(exist_ok=True)
            
            file_path = session_dir / filename
            
            if filename.endswith('.zip'):
                return self._handle_zip_upload(decoded, file_path, session_dir)
            else:
                return self._handle_python_upload(decoded, file_path)
                
        except Exception as e:
            print(f"Error saving uploaded file: {e}")
            return None
    
    def _handle_zip_upload(self, decoded_content: bytes, file_path: Path, session_dir: Path) -> str:
        """Handle ZIP file upload and extraction."""
        # Save ZIP file
        with open(file_path, 'wb') as f:
            f.write(decoded_content)
        
        # Extract ZIP contents
        extract_dir = session_dir / 'extracted'
        extract_dir.mkdir(exist_ok=True)
        
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        
        return str(extract_dir)
    
    def _handle_python_upload(self, decoded_content: bytes, file_path: Path) -> str:
        """Handle Python file upload."""
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(decoded_content.decode('utf-8'))
        
        return str(file_path)
    
    def find_algorithm_files(self, directory_path: str) -> List[str]:
        """
        Find Python files that likely contain algorithm implementations.
        
        Args:
            directory_path: Path to search for files
            
        Returns:
            List of file paths containing potential algorithms
        """
        algorithm_files = []
        path = Path(directory_path)
        
        if path.is_file() and path.suffix == '.py':
            return [str(path)]
        
        # Search for Python files recursively
        for py_file in path.rglob('*.py'):
            if self._file_contains_algorithm(py_file):
                algorithm_files.append(str(py_file))
        
        return algorithm_files
    
    def _file_contains_algorithm(self, file_path: Path) -> bool:
        """Check if a Python file likely contains an algorithm implementation."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Parse the file to check for classes
            try:
                tree = ast.parse(content)
                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef):
                        # Check if class inherits from BaseAlgorithm or similar
                        for base in node.bases:
                            if isinstance(base, ast.Name) and 'Algorithm' in base.id:
                                return True
                            if isinstance(base, ast.Attribute) and 'Algorithm' in base.attr:
                                return True
            except SyntaxError:
                # If we can't parse, fall back to string checking
                pass
            
            # Fallback: check for common algorithm-related keywords
            algorithm_keywords = [
                'BaseAlgorithm', 'class', 'def learn', 'def conformance',
                'algorithm', 'process mining', 'streaming'
            ]
            
            return any(keyword in content for keyword in algorithm_keywords)
            
        except Exception:
            return False
    
    def validate_libraries(self, libraries_text: str) -> Tuple[List[str], List[str]]:
        """
        Validate and parse required libraries.
        
        Args:
            libraries_text: Text containing library requirements
            
        Returns:
            Tuple of (valid_libraries, warnings)
        """
        if not libraries_text:
            return [], []
        
        valid_libraries = []
        warnings = []
        
        for line in libraries_text.strip().split('\n'):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
                
            # Basic validation - check if it looks like a valid package name
            if self._is_valid_package_name(line):
                valid_libraries.append(line)
            else:
                warnings.append(f"Invalid package name: {line}")
        
        return valid_libraries, warnings
    
    def _is_valid_package_name(self, package_name: str) -> bool:
        """Check if a string looks like a valid Python package name."""
        # Remove version specifiers
        name = package_name.split('==')[0].split('>=')[0].split('<=')[0].split('>')[0].split('<')[0]
        name = name.strip()
        
        # Basic validation
        if not name:
            return False
        
        # Check for valid characters
        valid_chars = set('abcdefghijklmnopqrstuvwxyz0123456789-_.')
        return all(c.lower() in valid_chars for c in name)
    
    def cleanup_session(self, session_id: str) -> bool:
        """
        Clean up uploaded files for a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            True if cleanup successful
        """
        try:
            session_dir = self.upload_directory / session_id
            if session_dir.exists():
                shutil.rmtree(session_dir)
            return True
        except Exception as e:
            print(f"Error cleaning up session {session_id}: {e}")
            return False

class AlgorithmLoader:
    """Handles loading and validation of algorithm classes from uploaded files."""
    
    def __init__(self, base_algorithm_class):
        """
        Initialize the algorithm loader.
        
        Args:
            base_algorithm_class: The base algorithm class that implementations should inherit from
        """
        self.base_class = base_algorithm_class
    
    def load_algorithm_from_file(self, file_path: str) -> Tuple[Optional[Type], Optional[str]]:
        """
        Load algorithm class from Python file.
        
        Args:
            file_path: Path to Python file
            
        Returns:
            Tuple of (algorithm_class, error_message)
        """
        try:
            # Add the file's directory to Python path temporarily
            file_dir = os.path.dirname(file_path)
            if file_dir not in sys.path:
                sys.path.insert(0, file_dir)
            
            try:
                # Import the module
                module_name = os.path.splitext(os.path.basename(file_path))[0]
                spec = importlib.util.spec_from_file_location(module_name, file_path)
                
                if spec is None or spec.loader is None:
                    return None, f"Could not create module spec for {file_path}"
                
                module = importlib.util.module_from_spec(spec)
                
                # Add necessary imports to module namespace
                self._prepare_module_namespace(module)
                
                # Execute the module
                spec.loader.exec_module(module)
                
                # Find algorithm class
                algorithm_class = self._find_concrete_algorithm_class(module.__dict__)
                
                if algorithm_class:
                    return algorithm_class, None
                else:
                    return None, "No valid algorithm class found in file"
                    
            finally:
                # Remove the added path
                if file_dir in sys.path:
                    sys.path.remove(file_dir)
                    
        except Exception as e:
            return None, f"Error loading algorithm: {str(e)}"
    
    def _prepare_module_namespace(self, module):
        """Prepare module namespace with necessary imports."""
        # Import commonly needed modules and classes
        import typing
        module.BaseAlgorithm = self.base_class
        module.Dict = typing.Dict
        module.Any = typing.Any
        module.List = typing.List
        
        # Import commonly used libraries
        try:
            import pandas as pd
            import numpy as np
            import time
            from collections import defaultdict
            from typing import Dict, Any, List, Optional
            
            module.pd = pd
            module.np = np
            module.time = time
            module.defaultdict = defaultdict
            module.Optional = Optional
        except ImportError:
            pass  # Continue without optional imports
    
    def _find_concrete_algorithm_class(self, namespace: Dict[str, Any]) -> Optional[Type]:
        """
        Find a concrete subclass of base_class in the namespace.
        
        Args:
            namespace: Module namespace to search
            
        Returns:
            First concrete algorithm class found or None
        """
        for obj in namespace.values():
            if not inspect.isclass(obj):
                continue
            
            # Skip the abstract base itself
            if obj is self.base_class:
                continue
            
            # Must be a subclass of the base_class
            if not issubclass(obj, self.base_class):
                continue
            
            # If obj still has abstract methods, skip it
            abstracts = getattr(obj, "__abstractmethods__", set())
            if abstracts:
                continue
            
            return obj
        
        return None
    
    def validate_algorithm_class(self, algorithm_class: Type) -> Tuple[bool, List[str]]:
        """
        Validate that an algorithm class meets requirements.
        
        Args:
            algorithm_class: The algorithm class to validate
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        try:
            # Check if it's a proper subclass
            if not issubclass(algorithm_class, self.base_class):
                issues.append(f"Class {algorithm_class.__name__} does not inherit from BaseAlgorithm")
            
            # Check for abstract methods
            abstracts = getattr(algorithm_class, "__abstractmethods__", set())
            if abstracts:
                issues.append(f"Class has unimplemented abstract methods: {', '.join(abstracts)}")
            
            # Check required methods
            required_methods = ['learn', 'conformance']
            for method_name in required_methods:
                if not hasattr(algorithm_class, method_name):
                    issues.append(f"Missing required method: {method_name}")
                elif not callable(getattr(algorithm_class, method_name)):
                    issues.append(f"Attribute {method_name} is not callable")
            
            # Try to instantiate the class
            try:
                instance = algorithm_class()
                
                # Check if basic methods are callable
                if hasattr(instance, 'learn') and callable(instance.learn):
                    # Test method signature
                    sig = inspect.signature(instance.learn)
                    if len(sig.parameters) < 1:  # Should have at least 'event' parameter
                        issues.append("learn() method should accept an event parameter")
                
                if hasattr(instance, 'conformance') and callable(instance.conformance):
                    sig = inspect.signature(instance.conformance)
                    if len(sig.parameters) < 1:
                        issues.append("conformance() method should accept an event parameter")
                        
            except Exception as e:
                issues.append(f"Cannot instantiate class: {str(e)}")
        
        except Exception as e:
            issues.append(f"Error validating class: {str(e)}")
        
        return len(issues) == 0, issues

class UploadValidator:
    """Validates uploaded files and provides feedback."""
    
    ALLOWED_EXTENSIONS = {'.py', '.zip'}
    MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
    
    def validate_file(self, contents: str, filename: str) -> Tuple[bool, List[str], List[str]]:
        """
        Validate uploaded file.
        
        Args:
            contents: Base64 encoded file content
            filename: Original filename
            
        Returns:
            Tuple of (is_valid, errors, warnings)
        """
        errors = []
        warnings = []
        
        # Check filename
        if not filename:
            errors.append("No filename provided")
            return False, errors, warnings
        
        # Check file extension
        file_ext = Path(filename).suffix.lower()
        if file_ext not in self.ALLOWED_EXTENSIONS:
            errors.append(f"Invalid file type. Allowed: {', '.join(self.ALLOWED_EXTENSIONS)}")
        
        # Check file size (approximate)
        if contents:
            try:
                # Rough estimate of file size from base64
                estimated_size = len(contents) * 3 / 4
                if estimated_size > self.MAX_FILE_SIZE:
                    errors.append(f"File too large. Maximum size: {self.MAX_FILE_SIZE // (1024*1024)}MB")
            except:
                pass
        
        # Additional validation for Python files
        if file_ext == '.py' and contents:
            try:
                content_type, content_string = contents.split(',')
                decoded = base64.b64decode(content_string)
                content = decoded.decode('utf-8')
                
                # Check if it's valid Python
                try:
                    ast.parse(content)
                except SyntaxError as e:
                    errors.append(f"Invalid Python syntax: {str(e)}")
                
                # Check for potential security issues
                if self._has_security_issues(content):
                    warnings.append("File contains potentially unsafe operations")
                    
            except Exception as e:
                errors.append(f"Cannot decode file content: {str(e)}")
        
        return len(errors) == 0, errors, warnings
    
    def _has_security_issues(self, content: str) -> bool:
        """Check for potentially unsafe operations in Python code."""
        # Basic security check - look for dangerous operations
        dangerous_patterns = [
            'import os', 'import sys', 'import subprocess', 'import shutil',
            'open(', 'file(', 'exec(', 'eval(', '__import__',
            'globals()', 'locals()', 'vars()', 'dir(',
            'getattr(', 'setattr(', 'delattr(', 'hasattr(',
        ]
        
        content_lower = content.lower()
        return any(pattern in content_lower for pattern in dangerous_patterns)

# Utility functions for the main application
def create_upload_response(success: bool, message: str, data: Dict = None) -> Dict[str, Any]:
    """Create standardized upload response."""
    response = {
        'success': success,
        'message': message,
        'timestamp': json.dumps(datetime.now(), default=str)
    }
    if data:
        response.update(data)
    return response

def get_file_info(file_path: str) -> Dict[str, Any]:
    """Get information about a file."""
    path = Path(file_path)
    
    info = {
        'name': path.name,
        'size': path.stat().st_size if path.exists() else 0,
        'extension': path.suffix,
        'is_directory': path.is_dir(),
        'modified': path.stat().st_mtime if path.exists() else 0
    }
    
    return info

# Export main classes for use in the application
__all__ = [
    'AlgorithmUploadManager',
    'AlgorithmLoader', 
    'UploadValidator',
    'create_upload_response',
    'get_file_info'
]