
import sys
from pathlib import Path

import cv2
from PIL import Image
from torchvision import transforms

from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    flash,
    send_from_directory,
)

# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

TEMPLATE_DIR = PROJECT_ROOT / "frontend" / "templates"
STATIC_DIR = PROJECT_ROOT / "frontend" / "static"

# ============================================================
# IMPORTS
# ============================================================

from backend.config import (
    MAX_CONTENT_LENGTH,
    ALLOWED_EXTENSIONS,
    UPLOAD_DIR,
)

from backend.api import api
from backend.external_detector import analyze_image_sync
from backend.utils import save_uploaded_file
from backend.model import load_model


# ============================================================
# HEATMAP PREPROCESSING
# ============================================================

IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


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

    app.register_blueprint(api)

    # Load the local explanation model once.
    heatmap_model = load_model()

    # --------------------------------------------------------
    # HOME
    # --------------------------------------------------------

    @app.route("/")
    def home():
        return render_template("home.html")

    # --------------------------------------------------------
    # UPLOAD PAGE
    # --------------------------------------------------------

    @app.route("/detect")
    def detect_page():
        return render_template("detect.html")

    # --------------------------------------------------------
    # PREDICT + GENERATE HEATMAP
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

            image_path = Path(save_uploaded_file(uploaded_file))

            # Validate uploaded image.
            image = cv2.imread(str(image_path))

            if image is None:
                image_path.unlink(missing_ok=True)
                flash("The uploaded file is not a valid readable image.")
                return redirect(url_for("detect_page"))

            # Reality Defender remains the prediction source.
            external_result = analyze_image_sync(str(image_path))
            status = external_result.get("status")

            if not status:
                raise RuntimeError(
                    "Reality Defender did not return a detection status."
                )

            # Generate a separate local-model explanation.
            with Image.open(image_path) as source:
                image_rgb = source.convert("RGB")
                image_tensor = IMAGE_TRANSFORM(image_rgb)

            local_prediction = heatmap_model.predict(image_tensor)

            heatmap_path = (
                Path(UPLOAD_DIR)
                / f"{image_path.stem}_heatmap.jpg"
            )

            heatmap_model.generate_heatmap(
                image_tensor=image_tensor,
                original_image_path=image_path,
                output_path=heatmap_path,
                target_class=local_prediction,
            )

            # Do not pass the numerical Reality Defender score.
            return render_template(
                "result.html",
                prediction=status,
                image_filename=image_path.name,
                heatmap_filename=heatmap_path.name,
                result_provider="Reality Defender",
                local_explanation_label=(
                    "REAL" if local_prediction == 0 else "FAKE"
                ),
            )

        except ValueError as error:
            app.logger.warning("Image analysis input error: %s", error)
            flash(str(error))
            return redirect(url_for("detect_page"))

        except Exception:
            app.logger.exception("DeepGuard image analysis failed")
            flash(
                "Image analysis or heatmap generation failed. "
                "Please try again."
            )
            return redirect(url_for("detect_page"))

    # --------------------------------------------------------
    # SERVE UPLOADED IMAGES AND HEATMAPS
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
