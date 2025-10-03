# app/utils/responses.py
from flask import jsonify

def success(payload, code: int = 200):
    """
    Return a successful JSON response.
    Example: return success({"rows": [...]})
    """
    return jsonify(payload), code


def error(message: str, code: int = 400):
    """
    Return a JSON error with consistent shape.
    Example: return error("Missing parameter", 400)
    """
    return jsonify({"error": message}), code
