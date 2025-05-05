import os
import logging.config
from pathlib import Path

def setup_logging(base_path: Path, config_name: str = 'default') -> None:
    """Configure logging based on the environment."""
    
    # Create logs directory if it doesn't exist
    logs_dir = os.path.join(base_path, 'logs')
    os.makedirs(logs_dir, exist_ok=True)

    # Common logging settings
    common_settings = {
        'version': 1,
        'disable_existing_loggers': True,  # Disable existing loggers to reduce noise
        'formatters': {
            'standard': {
                'format': '%(message)s'  # Simplified format for readability
            },
            'detailed': {
                'format': '%(asctime)s [%(levelname)s] %(message)s'
            }
        },
        'handlers': {
            'console': {
                'class': 'logging.StreamHandler',
                'level': 'INFO',
                'formatter': 'standard',
                'stream': 'ext://sys.stdout'
            },
            'file': {
                'class': 'logging.handlers.RotatingFileHandler',
                'level': 'INFO',
                'formatter': 'detailed',
                'filename': os.path.join(logs_dir, 'app.log'),
                'maxBytes': 10485760,  # 10MB
                'backupCount': 5
            },
            'error_file': {
                'class': 'logging.handlers.RotatingFileHandler',
                'level': 'ERROR',
                'formatter': 'detailed',
                'filename': os.path.join(logs_dir, 'error.log'),
                'maxBytes': 10485760,  # 10MB
                'backupCount': 5
            }
        },
        'loggers': {
            '': {  # Root logger
                'handlers': ['console', 'file', 'error_file'],
                'level': 'INFO',
                'propagate': True
            },
            'src.web': {  # Application logger
                'handlers': ['console', 'file', 'error_file'],
                'level': 'INFO',
                'propagate': False
            },
            # Disable noisy loggers
            'pdfminer': {'level': 'WARNING'},
            'PIL': {'level': 'WARNING'},
            'werkzeug': {'level': 'WARNING'},
            'urllib3': {'level': 'WARNING'}
        }
    }

    # Environment-specific settings
    env_settings = {
        'development': {
            'loggers': {
                '': {'level': 'INFO'},
                'src.web': {'level': 'INFO'}
            },
            'handlers': {
                'console': {'level': 'INFO'},
                'file': {'level': 'DEBUG'}
            }
        },
        'testing': {
            'loggers': {
                '': {'level': 'INFO'},
                'src.web': {'level': 'INFO'}
            },
            'handlers': {
                'console': {'level': 'INFO'},
                'file': {'filename': os.path.join(logs_dir, 'test.log')}
            }
        },
        'production': {
            'loggers': {
                '': {'level': 'WARNING'},
                'src.web': {'level': 'INFO'}
            },
            'handlers': {
                'console': {'level': 'WARNING'},
                'file': {
                    'level': 'INFO',
                    'maxBytes': 52428800,  # 50MB
                    'backupCount': 10
                },
                'error_file': {
                    'maxBytes': 52428800,  # 50MB
                    'backupCount': 10
                }
            }
        }
    }

    # Update settings based on environment
    if config_name in env_settings:
        env_config = env_settings[config_name]
        for logger_name, logger_config in env_config['loggers'].items():
            common_settings['loggers'][logger_name].update(logger_config)
        for handler_name, handler_config in env_config['handlers'].items():
            common_settings['handlers'][handler_name].update(handler_config)

    # Configure logging
    logging.config.dictConfig(common_settings)

    # Log startup message
    logger = logging.getLogger(__name__)
    logger.info(f"Starting application in {config_name} mode") 