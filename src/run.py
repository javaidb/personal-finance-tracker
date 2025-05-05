"""Entry point for the Flask application."""
import os
import sys

# Add the src directory to the Python path
src_path = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, os.path.dirname(src_path))

from src.web import create_app

app = create_app('development')

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8000, debug=True) 