# utils.py

from urllib.parse import urlparse, parse_qs


def extract_video_id(url: str) -> str:
    """
    Extracts video ID from different YouTube URL formats.

    Supported formats:
    - https://www.youtube.com/watch?v=VIDEO_ID
    - https://youtube.com/watch?v=VIDEO_ID
    - https://m.youtube.com/watch?v=VIDEO_ID
    - https://youtu.be/VIDEO_ID
    - https://youtube.com/shorts/VIDEO_ID
    - https://youtube.com/embed/VIDEO_ID
    """

    parsed_url = urlparse(url)

    # Short URL format
    if parsed_url.hostname == "youtu.be":
        return parsed_url.path.lstrip("/")

    # Standard YouTube URLs
    if parsed_url.hostname in [
        "www.youtube.com",
        "youtube.com",
        "m.youtube.com"
    ]:

        # watch?v=
        if parsed_url.path == "/watch":
            query_params = parse_qs(parsed_url.query)

            if "v" not in query_params:
                raise ValueError("No video ID found in URL")

            return query_params["v"][0]

        # shorts
        if parsed_url.path.startswith("/shorts/"):
            return parsed_url.path.split("/")[2]

        # embed
        if parsed_url.path.startswith("/embed/"):
            return parsed_url.path.split("/")[2]

    raise ValueError("Invalid YouTube URL")