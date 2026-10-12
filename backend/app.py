from pathlib import Path

import cv2

from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    flash,
    send_from_directory,
)

from config import (
    MAX_CONTENT_LENGTH,
    ALLOWED_EXTENSIONS,
    UPLOAD_DIR,
)

from api import api
from external_detector import analyze_image_sync
from utils import save_uploaded_file


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TEMPLATE_DIR = PROJECT_ROOT / "frontend" / "templates"
STATIC_DIR = PROJECT_ROOT / "frontend" / "static"


# ============================================================
# CREATE FLASK APPLICATION
# ============================================================

def create_app():
    app = Flask(
        __name__,
        template_folder=str(TEMPLATE_DIR),
        static_folder=str(STATIC_DIR),
        static_url_path="/static",
    )

    app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH
    app.secret_key = "deepguard-local-development-key"

    # Register the existing REST API.
    app.register_blueprint(api)

    # --------------------------------------------------------
    # HOME PAGE
    # --------------------------------------------------------

    @app.route("/")
    def home():
        return render_template("home.html")

    # --------------------------------------------------------
    # IMAGE UPLOAD PAGE
    # --------------------------------------------------------

    @app.route("/detect")
    def detect_page():
        return render_template("detect.html")

    # --------------------------------------------------------
    # REALITY DEFENDER IMAGE ANALYSIS
    # --------------------------------------------------------

    @app.route("/web-predict", methods=["POST"])
    def web_predict():
        image_path = None

        try:
            if "file" not in request.files:
                flash("Please select an image to analyze.")
                return redirect(url_for("detect_page"))

            uploaded_file = request.files["file"]

            if not uploaded_file.filename:
                flash("Please select an image to analyze.")
                return redirect(url_for("detect_page"))

            extension = uploaded_file.filename.rsplit(".", 1)

            if (
                len(extension) != 2
                or extension[1].lower() not in ALLOWED_EXTENSIONS
            ):
                flash("Please upload a JPG, JPEG, PNG, or WEBP image.")
                return redirect(url_for("detect_page"))

            # Save the uploaded image.
            image_path = save_uploaded_file(uploaded_file)

            # Validate that the saved file is a readable image.
            image = cv2.imread(str(image_path))

            if image is None:
                image_path.unlink(missing_ok=True)
                flash("The uploaded file is not a valid readable image.")
                return redirect(url_for("detect_page"))

            # Reality Defender is the sole prediction source
            # for the website's result page.
            external_result = analyze_image_sync(str(image_path))

            status = external_result.get("status")
            score = external_result.get("score")

            if not status:
                raise RuntimeError(
                    "Reality Defender did not return a detection status."
                )

            # Display the provider's status and score.
            return render_template(
                "result.html",
                prediction=status,
                score=score,
                image_filename=image_path.name,
                result_provider="Reality Defender",
            )

        except ValueError as error:
            flash(str(error))
            return redirect(url_for("detect_page"))

        except Exception:
            app.logger.exception("Reality Defender image analysis failed")
            flash(
                "Reality Defender could not analyze this image. "
                "Please try again later."
            )
            return redirect(url_for("detect_page"))

    # --------------------------------------------------------
    # DISPLAY UPLOADED IMAGE
    # --------------------------------------------------------

    @app.route("/uploads/<path:filename>")
    def uploaded_image(filename):
        return send_from_directory(
            str(UPLOAD_DIR),
            filename,
            as_attachment=False,
        )

    # --------------------------------------------------------
    # NOT FOUND
    # --------------------------------------------------------

    @app.errorhandler(404)
    def page_not_found(error):
        return render_template("home.html"), 404

    return app


# ============================================================
# APPLICATION INSTANCE
# ============================================================

app = create_app()


# ============================================================
# RUN SERVER
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("DeepGuard Backend")
    print("=" * 60)
    print("Home:       http://127.0.0.1:5000/")
    print("Upload:     http://127.0.0.1:5000/detect")
    print("API health: http://127.0.0.1:5000/api/health")
    print("=" * 60)

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True,
    )