"""One HTTP session for every Memory API call, carrying the internal bearer token."""
import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "service"))
from runtime_config import required  # noqa: E402

api = requests.Session()
api.headers["Authorization"] = f"Bearer {required('MEMORY_INTERNAL_TOKEN')}"
