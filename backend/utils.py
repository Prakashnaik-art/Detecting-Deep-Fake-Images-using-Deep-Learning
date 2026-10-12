import uuid
from pathlib import Path

from werkzeug.utils import secure_filename

from config import (
    UPLOAD_DIR,
    RESULT_DIR,
    ALLOWED_EXTENSIONS,
)


# ============================================================
# CHECK FILE EXTENSION
# ============================================================

def allowed_file(filename):

    if not filename:
        return False

    if "." not in filename:
        return False

    extension = filename.rsplit(
        ".", 1
    )[1].lower()

    return extension in ALLOWED_EXTENSIONS


# ============================================================
# GET FILE EXTENSION
# ============================================================

def get_file_extension(filename):

    return Path(filename).suffix.lower()


# ============================================================
# GENERATE UNIQUE FILENAME
# ============================================================

def generate_filename(filename):

    safe_filename = secure_filename(
        filename
    )

    extension = get_file_extension(
        safe_filename
    )

    if not extension:
        raise ValueError(
            "File extension is missing."
        )

    return (
        f"{uuid.uuid4().hex}"
        f"{extension}"
    )


# ============================================================
# SAVE UPLOADED FILE
# ============================================================

def save_uploaded_file(file):

    if file is None:
        raise ValueError(
            "No file was provided."
        )

    if not file.filename:
        raise ValueError(
            "Filename is empty."
        )

    if not allowed_file(
        file.filename
    ):
        raise ValueError(
            "Unsupported file type. "
            "Allowed types: JPG, JPEG, PNG, WEBP."
        )

    filename = generate_filename(
        file.filename
    )

    file_path = UPLOAD_DIR / filename

    file.save(file_path)

    return file_path


# ============================================================
# GENERATE RESULT FILENAME
# ============================================================

def generate_result_filename(
    original_filename
):

    extension = get_file_extension(
        original_filename
    )

    if not extension:
        extension = ".jpg"

    return (
        f"result_"
        f"{uuid.uuid4().hex}"
        f"{extension}"
    )


# ============================================================
# GET RESULT PATH
# ============================================================

def get_result_path(
    original_filename
):

    filename = generate_result_filename(
        original_filename
    )

    return RESULT_DIR / filename


# ============================================================
# CHECK FILE EXISTS
# ============================================================

def file_exists(file_path):

    return Path(file_path).exists()


# ============================================================
# DELETE FILE
# ============================================================

def delete_file(file_path):

    path = Path(file_path)

    if path.exists():
        path.unlink()


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("DeepGuard Utils Test")
    print("=" * 60)

    print(
        "Upload directory:",
        UPLOAD_DIR
    )

    print(
        "Result directory:",
        RESULT_DIR
    )

    print()

    test_files = [
        "photo.jpg",
        "photo.jpeg",
        "photo.png",
        "photo.webp",
        "photo.exe",
    ]

    for filename in test_files:

        print(
            f"{filename:15} "
            f"-> "
            f"{allowed_file(filename)}"
        )

    print("=" * 60)
    print("Utils loaded successfully")
    print("=" * 60)