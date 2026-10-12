
import asyncio
import os

from dotenv import load_dotenv
from realitydefender import RealityDefender

# Load environment variables from the project root.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))


async def analyze_image(image_path):
    """
    Submit an image to Reality Defender and retrieve its analysis.

    Returns the provider's status and score.
    """
    api_key = os.getenv("REALITY_DEFENDER_API_KEY")

    if not api_key:
        raise RuntimeError(
            "REALITY_DEFENDER_API_KEY is missing from the project .env file."
        )

    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")

    client = RealityDefender(api_key=api_key)

    upload = await client.upload(file_path=image_path)

    if not upload or not upload.get("request_id"):
        raise RuntimeError(
            "Reality Defender did not return an analysis request ID."
        )

    result = await client.get_result(upload["request_id"])

    if not result or not result.get("status"):
        raise RuntimeError(
            "Reality Defender returned an empty or invalid analysis result."
        )

    return {
        "status": result["status"],
        "score": result.get("score"),
    }


def analyze_image_sync(image_path):
    """Synchronous wrapper for use in a Flask route."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(analyze_image(image_path))

    raise RuntimeError(
        "Cannot use this synchronous wrapper inside an active "
        "asyncio event loop."
    )
