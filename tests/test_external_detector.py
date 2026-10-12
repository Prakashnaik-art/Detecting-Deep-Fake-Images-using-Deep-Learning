
import asyncio
import os
from pathlib import Path

from dotenv import load_dotenv
from realitydefender import RealityDefender

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")


async def main():
    api_key = os.getenv("REALITY_DEFENDER_API_KEY")

    if not api_key:
        raise RuntimeError(
            "REALITY_DEFENDER_API_KEY is missing from the root .env file."
        )

    image_path = input(
        "Enter the full path to a JPG or PNG image: "
    ).strip().strip('"')

    image_path = Path(image_path)

    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    client = RealityDefender(api_key=api_key)

    print("Uploading image to Reality Defender...")
    upload = await client.upload(file_path=str(image_path))
    request_id = upload["request_id"]

    print("Analysis submitted. Waiting for the result...")
    result = await client.get_result(request_id)

    print("\nExternal analysis completed.")
    print("Status:", result.get("status"))
    print("Score:", result.get("score"))


if __name__ == "__main__":
    asyncio.run(main())
