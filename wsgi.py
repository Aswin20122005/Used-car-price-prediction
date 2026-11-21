"""WSGI entry point for production servers, e.g.

    gunicorn --bind 0.0.0.0:8000 --workers 2 wsgi:app
"""

from app import app  # noqa: F401
