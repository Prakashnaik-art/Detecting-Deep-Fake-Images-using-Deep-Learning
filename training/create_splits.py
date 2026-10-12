import csv
import json
import random
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_ROOT = Path(
    r"C:\Users\praka\Downloads\FaceForensics++_C23"
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

OUTPUT_DIR = PROJECT_ROOT / "dataset"

RANDOM_SEED = 42

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15


# ============================================================
# DATASET FOLDERS
# ============================================================

ORIGINAL_DIR = DATASET_ROOT / "original"

MANIPULATION_DIRS = [
    DATASET_ROOT / "NeuralTextures",
    DATASET_ROOT / "FaceSwap",
    DATASET_ROOT / "FaceShifter",
    DATASET_ROOT / "Face2Face",
]

DEEPFAKE_DETECTION_DIR = DATASET_ROOT / "DeepFakeDetection"

CSV_DIR = DATASET_ROOT / "csv"


# ============================================================
# HELPERS
# ============================================================

def get_mp4_files(folder):
    """Return MP4 files from a folder."""

    if not folder.exists():
        print(f"WARNING: Folder not found: {folder}")
        return []

    return sorted(
        [
            file
            for file in folder.iterdir()
            if file.is_file()
            and file.suffix.lower() == ".mp4"
        ]
    )


def get_source_id(filename):
    """
    Get the source video ID from a standard
    FaceForensics++ manipulated filename.

    Example:
        000_003.mp4 -> 000, 003
    """

    parts = filename.stem.split("_")

    if len(parts) < 2:
        return None

    first = parts[0]
    second = parts[1]

    if (
        len(first) == 3
        and len(second) == 3
        and first.isdigit()
        and second.isdigit()
    ):
        return first, second

    return None


# ============================================================
# CREATE SOURCE SPLITS
# ============================================================

def create_source_splits():
    """Split the 1000 original source IDs."""

    original_videos = get_mp4_files(ORIGINAL_DIR)

    source_ids = []

    for video in original_videos:

        source_id = video.stem

        if source_id.isdigit():
            source_ids.append(source_id)

    source_ids = sorted(set(source_ids))

    print()
    print(f"Original source videos: {len(source_ids)}")

    random.seed(RANDOM_SEED)

    random.shuffle(source_ids)

    total = len(source_ids)

    train_end = int(total * TRAIN_RATIO)

    val_end = train_end + int(total * VAL_RATIO)

    train_ids = set(source_ids[:train_end])

    val_ids = set(source_ids[train_end:val_end])

    test_ids = set(source_ids[val_end:])

    return train_ids, val_ids, test_ids


# ============================================================
# ORIGINAL REAL VIDEOS
# ============================================================

def create_original_samples(split_ids):
    """Create REAL samples from original videos."""

    samples = []

    for source_id in sorted(split_ids):

        video = ORIGINAL_DIR / f"{source_id}.mp4"

        if not video.exists():
            continue

        samples.append(
            {
                "path": str(video.relative_to(DATASET_ROOT)),
                "label": 0,
                "class": "REAL",
                "source_id": source_id,
                "dataset": "FaceForensics++",
            }
        )

    return samples


# ============================================================
# STANDARD FAKE MANIPULATIONS
# ============================================================

def create_manipulation_samples(folder, split_ids):
    """
    Create FAKE samples from standard FaceForensics++
    manipulation folders.

    A manipulated video such as:

        000_003.mp4

    is included only when BOTH source IDs belong
    to the same split.
    """

    samples = []

    videos = get_mp4_files(folder)

    for video in videos:

        source_pair = get_source_id(video)

        if source_pair is None:
            continue

        source_a, source_b = source_pair

        if (
            source_a in split_ids
            and source_b in split_ids
        ):
            samples.append(
                {
                    "path": str(
                        video.relative_to(DATASET_ROOT)
                    ),
                    "label": 1,
                    "class": "FAKE",
                    "source_id": f"{source_a}_{source_b}",
                    "dataset": folder.name,
                }
            )

    return samples


# ============================================================
# DEEPFAKEDETECTION CSV
# ============================================================

def find_deepfake_detection_csv():
    """Find the DeepFakeDetection CSV file."""

    candidates = list(
        CSV_DIR.glob("DeepFakeDetection*.csv")
    )

    if not candidates:
        raise FileNotFoundError(
            "Could not find DeepFakeDetection CSV "
            f"inside {CSV_DIR}"
        )

    return candidates[0]


def read_deepfake_detection_samples():
    """
    Read DeepFakeDetection metadata.

    Only rows labeled FAKE are used.
    """

    csv_path = find_deepfake_detection_csv()

    print()
    print(f"Reading DeepFakeDetection CSV:")
    print(csv_path)

    samples = []

    with open(
        csv_path,
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as file:

        reader = csv.DictReader(file)

        if not reader.fieldnames:
            raise RuntimeError(
                "DeepFakeDetection CSV has no headers."
            )

        # Find columns ignoring case and spaces.
        columns = {
            column.strip().lower(): column
            for column in reader.fieldnames
        }

        if "file path" not in columns:
            raise RuntimeError(
                "Could not find 'File Path' column "
                "in DeepFakeDetection CSV."
            )

        if "label" not in columns:
            raise RuntimeError(
                "Could not find 'Label' column "
                "in DeepFakeDetection CSV."
            )

        path_column = columns["file path"]
        label_column = columns["label"]

        for row in reader:

            label = (
                row[label_column]
                .strip()
                .upper()
            )

            if label != "FAKE":
                continue

            file_path = (
                row[path_column]
                .strip()
            )

            if not file_path:
                continue

            filename = Path(file_path).name

            video = (
                DEEPFAKE_DETECTION_DIR
                / filename
            )

            if not video.exists():
                continue

            samples.append(
                {
                    "path": str(
                        video.relative_to(DATASET_ROOT)
                    ),
                    "label": 1,
                    "class": "FAKE",
                    "source_id": filename,
                    "dataset": "DeepFakeDetection",
                }
            )

    # Remove duplicate paths.
    unique = {}

    for sample in samples:
        unique[sample["path"]] = sample

    return list(unique.values())


# ============================================================
# SPLIT DEEPFAKEDETECTION
# ============================================================

def split_dfd_samples(samples):
    """
    Split DeepFakeDetection samples separately.

    These videos don't use the normal FF++ pair naming,
    so they are split at the video level.
    """

    random.seed(RANDOM_SEED)

    random.shuffle(samples)

    total = len(samples)

    train_end = int(total * TRAIN_RATIO)

    val_end = train_end + int(total * VAL_RATIO)

    train = samples[:train_end]

    val = samples[train_end:val_end]

    test = samples[val_end:]

    return train, val, test


# ============================================================
# SAVE JSON
# ============================================================

def save_json(filename, samples):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = OUTPUT_DIR / filename

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            samples,
            file,
            indent=4
        )

    print(
        f"Created: {output_path}"
    )


# ============================================================
# STATISTICS
# ============================================================

def print_statistics(name, samples):

    real_count = sum(
        1
        for sample in samples
        if sample["label"] == 0
    )

    fake_count = sum(
        1
        for sample in samples
        if sample["label"] == 1
    )

    print()
    print(name)
    print("-" * 40)
    print(f"Total : {len(samples)}")
    print(f"REAL  : {real_count}")
    print(f"FAKE  : {fake_count}")


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("DeepGuard - Creating New Dataset Splits")
    print("=" * 60)

    print()
    print(f"Dataset root:")
    print(DATASET_ROOT)

    # --------------------------------------------------------
    # CHECK DATASET
    # --------------------------------------------------------

    if not DATASET_ROOT.exists():

        print()
        print("ERROR:")
        print("Dataset folder does not exist.")

        return

    # --------------------------------------------------------
    # SOURCE SPLIT
    # --------------------------------------------------------

    train_ids, val_ids, test_ids = (
        create_source_splits()
    )

    print()
    print(
        f"TRAIN source IDs: {len(train_ids)}"
    )

    print(
        f"VAL source IDs  : {len(val_ids)}"
    )

    print(
        f"TEST source IDs : {len(test_ids)}"
    )

    # --------------------------------------------------------
    # ORIGINAL REAL
    # --------------------------------------------------------

    train_samples = create_original_samples(
        train_ids
    )

    val_samples = create_original_samples(
        val_ids
    )

    test_samples = create_original_samples(
        test_ids
    )

    print()
    print(
        f"REAL train: {len(train_samples)}"
    )

    print(
        f"REAL val: {len(val_samples)}"
    )

    print(
        f"REAL test: {len(test_samples)}"
    )

    # --------------------------------------------------------
    # STANDARD MANIPULATIONS
    # --------------------------------------------------------

    for folder in MANIPULATION_DIRS:

        print()
        print(
            f"Processing {folder.name}..."
        )

        train_fake = create_manipulation_samples(
            folder,
            train_ids
        )

        val_fake = create_manipulation_samples(
            folder,
            val_ids
        )

        test_fake = create_manipulation_samples(
            folder,
            test_ids
        )

        train_samples.extend(train_fake)
        val_samples.extend(val_fake)
        test_samples.extend(test_fake)

        print(
            f"  TRAIN: {len(train_fake)}"
        )

        print(
            f"  VAL:   {len(val_fake)}"
        )

        print(
            f"  TEST:  {len(test_fake)}"
        )

    # --------------------------------------------------------
    # DEEPFAKEDETECTION
    # --------------------------------------------------------

    print()
    print(
        "Processing DeepFakeDetection..."
    )

    dfd_samples = (
        read_deepfake_detection_samples()
    )

    print(
        f"DeepFakeDetection FAKE videos: "
        f"{len(dfd_samples)}"
    )

    dfd_train, dfd_val, dfd_test = (
        split_dfd_samples(dfd_samples)
    )

    train_samples.extend(dfd_train)
    val_samples.extend(dfd_val)
    test_samples.extend(dfd_test)

    print(
        f"  TRAIN: {len(dfd_train)}"
    )

    print(
        f"  VAL:   {len(dfd_val)}"
    )

    print(
        f"  TEST:  {len(dfd_test)}"
    )

    # --------------------------------------------------------
    # SHUFFLE FINAL SPLITS
    # --------------------------------------------------------

    random.seed(RANDOM_SEED)

    random.shuffle(train_samples)
    random.shuffle(val_samples)
    random.shuffle(test_samples)

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    save_json(
        "train.json",
        train_samples
    )

    save_json(
        "val.json",
        val_samples
    )

    save_json(
        "test.json",
        test_samples
    )

    # --------------------------------------------------------
    # STATISTICS
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("FINAL DATASET SUMMARY")
    print("=" * 60)

    print_statistics(
        "TRAIN",
        train_samples
    )

    print_statistics(
        "VALIDATION",
        val_samples
    )

    print_statistics(
        "TEST",
        test_samples
    )

    print()
    print("=" * 60)
    print("DeepGuard dataset creation completed.")
    print("=" * 60)


if __name__ == "__main__":
    main()