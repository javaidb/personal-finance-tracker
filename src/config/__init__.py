"""
Configuration package for the personal finance tracker project.
"""

from .paths import (
    ProjectPaths,
    paths,
    get_project_root,
    get_cached_data_dir,
    get_cached_data_file,
    get_bank_statements_dir,
    get_logs_dir,
    get_uploads_dir,
    get_pdf_cache_dir,
    DATABANK_PATH,
    MERCHANT_DB_PATH,
    MERCHANT_ALIASES_PATH,
    UNCATEGORIZED_MERCHANTS_PATH,
    UNCHARACTERIZED_MERCHANTS_PATH,
    MANUAL_CATEGORIES_PATH,
    CATEGORY_COLORS_PATH,
    SHOPPING_KEYWORDS_PATH,
    DINING_KEYWORDS_PATH,
)

__all__ = [
    'ProjectPaths',
    'paths',
    'get_project_root',
    'get_cached_data_dir',
    'get_cached_data_file',
    'get_bank_statements_dir',
    'get_logs_dir',
    'get_uploads_dir',
    'get_pdf_cache_dir',
    'DATABANK_PATH',
    'MERCHANT_DB_PATH',
    'MERCHANT_ALIASES_PATH',
    'UNCATEGORIZED_MERCHANTS_PATH',
    'UNCHARACTERIZED_MERCHANTS_PATH',
    'MANUAL_CATEGORIES_PATH',
    'CATEGORY_COLORS_PATH',
    'SHOPPING_KEYWORDS_PATH',
    'DINING_KEYWORDS_PATH',
] 