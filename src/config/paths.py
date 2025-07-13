"""
Centralized path configuration for the personal finance tracker project.
Provides absolute paths that can be reliably referenced from anywhere in the project.
"""
import os
from pathlib import Path
from typing import Dict, Any


class ProjectPaths:
    """Centralized path configuration for the project."""
    
    # Singleton instance
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ProjectPaths, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        # Get the absolute path to the project root
        # This file is at src/config/paths.py, so we go up 3 levels to get to project root
        self._project_root = Path(__file__).parent.parent.parent.absolute()
        
        # Define all project paths as absolute paths
        self._paths = {
            # Core directories
            'project_root': self._project_root,
            'src': self._project_root / 'src',
            'web': self._project_root / 'src' / 'web',
            'modules': self._project_root / 'src' / 'modules',
            'config': self._project_root / 'src' / 'config',
            
            # Data directories
            'cached_data': self._project_root / 'cached_data',
            'bank_statements': self._project_root / 'bank_statements',
            'logs': self._project_root / 'logs',
            'uploads': self._project_root / 'uploads',
            
            # Test directories
            'tests': self._project_root / 'tests',
            'test_data': self._project_root / 'tests' / 'test_data',
            'test_cached_data': self._project_root / 'tests' / 'test_data' / 'cached_data',
            
            # Cached data subdirectories
            'pdf_cache': self._project_root / 'cached_data' / 'pdf_cache',
        }
        
        # Define all cached data files as absolute paths
        self._cached_data_files = {
            'databank': self._paths['cached_data'] / 'databank.json',
            'merchant_db': self._paths['cached_data'] / 'merchant_db.json',
            'merchant_aliases': self._paths['cached_data'] / 'merchant_aliases.json',
            'uncategorized_merchants': self._paths['cached_data'] / 'uncategorized_merchants.json',
            'uncharacterized_merchants': self._paths['cached_data'] / 'uncharacterized_merchants.json',
            'manual_categories': self._paths['cached_data'] / 'manual_categories.json',
            'category_colors': self._paths['cached_data'] / 'category_colors.json',
            'shopping_keywords': self._paths['cached_data'] / 'shopping_keywords.json',
            'dining_keywords': self._paths['cached_data'] / 'dining_keywords.json',
        }
        
        # Define test cached data files
        self._test_cached_data_files = {
            'databank': self._paths['test_cached_data'] / 'databank.json',
            'merchant_db': self._paths['test_cached_data'] / 'merchant_db.json',
            'category_colors': self._paths['test_cached_data'] / 'category_colors.json',
        }
        
        self._initialized = True
    
    @property
    def project_root(self) -> Path:
        """Get the absolute path to the project root directory."""
        return self._paths['project_root']
    
    @property
    def cached_data(self) -> Path:
        """Get the absolute path to the cached_data directory."""
        return self._paths['cached_data']
    
    @property
    def bank_statements(self) -> Path:
        """Get the absolute path to the bank_statements directory."""
        return self._paths['bank_statements']
    
    @property
    def logs(self) -> Path:
        """Get the absolute path to the logs directory."""
        return self._paths['logs']
    
    @property
    def uploads(self) -> Path:
        """Get the absolute path to the uploads directory."""
        return self._paths['uploads']
    
    @property
    def pdf_cache(self) -> Path:
        """Get the absolute path to the pdf_cache directory."""
        return self._paths['pdf_cache']
    
    def get_cached_data_file(self, filename: str) -> Path:
        """Get the absolute path to a cached data file."""
        if filename in self._cached_data_files:
            return self._cached_data_files[filename]
        else:
            # If not in predefined list, construct path
            return self._paths['cached_data'] / filename
    
    def get_test_cached_data_file(self, filename: str) -> Path:
        """Get the absolute path to a test cached data file."""
        if filename in self._test_cached_data_files:
            return self._test_cached_data_files[filename]
        else:
            # If not in predefined list, construct path
            return self._paths['test_cached_data'] / filename
    
    def ensure_directory_exists(self, directory: str) -> Path:
        """Ensure a directory exists and return its absolute path."""
        path = self._paths.get(directory)
        if path:
            path.mkdir(parents=True, exist_ok=True)
            return path
        else:
            raise ValueError(f"Unknown directory: {directory}")
    
    def ensure_cached_data_exists(self) -> Path:
        """Ensure the cached_data directory exists and return its absolute path."""
        return self.ensure_directory_exists('cached_data')
    
    def ensure_pdf_cache_exists(self) -> Path:
        """Ensure the pdf_cache directory exists and return its absolute path."""
        return self.ensure_directory_exists('pdf_cache')
    
    def get_all_paths(self) -> Dict[str, Path]:
        """Get all project paths as a dictionary."""
        return self._paths.copy()
    
    def get_all_cached_data_files(self) -> Dict[str, Path]:
        """Get all cached data file paths as a dictionary."""
        return self._cached_data_files.copy()
    
    def get_all_test_cached_data_files(self) -> Dict[str, Path]:
        """Get all test cached data file paths as a dictionary."""
        return self._test_cached_data_files.copy()


# Global instance for easy access
paths = ProjectPaths()


# Convenience functions for backward compatibility
def get_project_root() -> Path:
    """Get the absolute path to the project root directory."""
    return paths.project_root


def get_cached_data_dir() -> Path:
    """Get the absolute path to the cached_data directory."""
    return paths.cached_data


def get_cached_data_file(filename: str) -> Path:
    """Get the absolute path to a cached data file."""
    return paths.get_cached_data_file(filename)


def get_bank_statements_dir() -> Path:
    """Get the absolute path to the bank_statements directory."""
    return paths.bank_statements


def get_logs_dir() -> Path:
    """Get the absolute path to the logs directory."""
    return paths.logs


def get_uploads_dir() -> Path:
    """Get the absolute path to the uploads directory."""
    return paths.uploads


def get_pdf_cache_dir() -> Path:
    """Get the absolute path to the pdf_cache directory."""
    return paths.pdf_cache


# Predefined file paths for common cached data files
DATABANK_PATH = paths.get_cached_data_file('databank.json')
MERCHANT_DB_PATH = paths.get_cached_data_file('merchant_db.json')
MERCHANT_ALIASES_PATH = paths.get_cached_data_file('merchant_aliases.json')
UNCATEGORIZED_MERCHANTS_PATH = paths.get_cached_data_file('uncategorized_merchants.json')
UNCHARACTERIZED_MERCHANTS_PATH = paths.get_cached_data_file('uncharacterized_merchants.json')
MANUAL_CATEGORIES_PATH = paths.get_cached_data_file('manual_categories.json')
CATEGORY_COLORS_PATH = paths.get_cached_data_file('category_colors.json')
SHOPPING_KEYWORDS_PATH = paths.get_cached_data_file('shopping_keywords.json')
DINING_KEYWORDS_PATH = paths.get_cached_data_file('dining_keywords.json') 