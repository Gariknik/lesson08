import time
from dotenv import load_dotenv
from google import genai
from google.genai import errors, types

load_dotenv()
client = genai.Client(
    http_options=types.HttpOptions(
        retry_options=types.HttpRetryOptions(attempts=2, initial_delay=0.5, max_delay=2.0)
    )
)

MODEL = "gemini-3.5-flash-lite"        # перевірте актуальну назву в AI Studio
EMBED_MODEL = "gemini-embedding-001"
EMBED_DIM = 768

_FALLBACK_MODELS = [
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemini-3.8-flash",
    "gemini-3.7-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-4-hydrogen",
    "gemma-4-26b-a4b-it",
]

_orig_generate_content = client.models.generate_content


def _generate_content_with_fallback(*args, **kwargs):
    primary = kwargs.get("model", MODEL)
    candidates = [primary] + [m for m in _FALLBACK_MODELS if m != primary]
    last_err = None
    for attempt in range(2):
        for candidate in candidates:
            kwargs["model"] = candidate
            try:
                return _orig_generate_content(*args, **kwargs)
            except (errors.ServerError, errors.ClientError) as e:
                if getattr(e, "code", None) in (429, 503):
                    last_err = e
                    continue
                raise
        time.sleep(1.5 * (attempt + 1))
    raise last_err


client.models.generate_content = _generate_content_with_fallback
