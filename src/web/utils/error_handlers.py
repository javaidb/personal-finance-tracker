from typing import Union, Dict, Any, Tuple
from flask import jsonify, render_template
import logging

logger = logging.getLogger(__name__)

class ServiceError(Exception):
    """Custom exception for service-level errors."""
    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.status_code = status_code
        self.message = message

def handle_service_error(error: Exception) -> Union[str, Tuple[Dict[str, Any], int]]:
    """Handle service errors and return appropriate response."""
    if isinstance(error, ServiceError):
        status_code = error.status_code
        message = error.message
    else:
        status_code = 500
        message = "An unexpected error occurred"
        logger.error(f"Unexpected error: {str(error)}", exc_info=True)
    
    # Check if the request was for an API endpoint
    if request.path.startswith('/api/'):
        return jsonify({
            "success": False,
            "error": message
        }), status_code
    
    # For web pages, render error template
    return render_template('error.html',
                         message=message,
                         show_details=status_code == 500,
                         error_details=str(error) if status_code == 500 else None), status_code 