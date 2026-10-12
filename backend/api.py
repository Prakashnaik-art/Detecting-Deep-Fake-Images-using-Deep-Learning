from functools import wraps

import cv2
from flask import Blueprint, jsonify, request

from config import API_KEY
from utils import save_uploaded_file


# ============================================================
# API BLUEPRINT
# ============================================================

api = Blueprint(
    "api",
    __name__,
    url_prefix="/api",
)


# ============================================================
# API KEY AUTHENTICATION
# ============================================================

def require_api_key(function):

    @wraps(function)
    def decorated(*args, **kwargs):

        provided_key = request.headers.get("X-API-Key")

        if not provided_key:
            return jsonify({
                "success": False,
                "error": "API key is required.",
            }), 401

        if provided_key != API_KEY:
            return jsonify({
                "success": False,
                "error": "Invalid API key.",
            }), 403

        return function(*args, **kwargs)

    return decorated


# ============================================================
# HEALTH CHECK
# ============================================================

@api.route("/health", methods=["GET"])
def health():

    return jsonify({
        "success": True,
        "status": "DeepGuard API is running.",
    })


# ============================================================
# PREDICT
# ============================================================

@api.route("/predict", methods=["POST"])
@require_api_key
def predict():

    file_path = None

    try:

        # ----------------------------------------------------
        # Check uploaded file
        # ----------------------------------------------------

        if "file" not in request.files:
            return jsonify({
                "success": False,
                "error": "No image file provided.",
            }), 400

        file = request.files["file"]

        if not file.filename:
            return jsonify({
                "success": False,
                "error": "Filename is empty.",
            }), 400

        # ----------------------------------------------------
        # Save uploaded image
        # ----------------------------------------------------

        file_path = save_uploaded_file(file)

        # ----------------------------------------------------
        # Read image
        # ----------------------------------------------------

        image = cv2.imread(str(file_path))

        if image is None:
            return jsonify({
                "success": False,
                "error": "Could not read the uploaded image.",
            }), 400

        # ----------------------------------------------------
        # Load predictor only when prediction is requested
        # ----------------------------------------------------

        from predictor import load_predictor

        predictor = load_predictor()

        # ----------------------------------------------------
        # Run prediction
        # ----------------------------------------------------

        result = predictor.predict(image)

        # ----------------------------------------------------
        # API response
        # ----------------------------------------------------

        return jsonify({
            "success": True,
            "prediction": result["label"],
        })

    except ValueError as error:

        return jsonify({
            "success": False,
            "error": str(error),
        }), 400

    except Exception:
        from flask import current_app

        current_app.logger.exception(
            "DeepGuard prediction failed"
        )

        return jsonify({
            "success": False,
            "error": "An internal prediction error occurred.",
        }), 500

    finally:

        # ----------------------------------------------------
        # Clean up uploaded temporary file
        # ----------------------------------------------------

        if file_path is not None:
            file_path.unlink(missing_ok=True)
