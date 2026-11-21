import json, os, re, threading, time
from openai import OpenAI, RateLimitError, AuthenticationError, BadRequestError
from . import config

_client = OpenAI(base_url=config.LLM_BASE_URL, api_key=config.LLM_API_KEY or "none",
                 timeout=90, max_retries=0)
EFFORT = os.getenv("LLM_REASONING_EFFORT")
MIN_INTERVAL = float(os.getenv("LLM_MIN_INTERVAL", "0"))
MAX_WAIT = 120
_lock = threading.Lock()
_last = [0.0]

class FatalLLMError(Exception):
    """Stop the whole run: retrying will not help."""
class RateLimitExhausted(FatalLLMError):
    pass
class AuthFailed(FatalLLMError):
    pass

def _throttle():
    if MIN_INTERVAL <= 0:
        return
    with _lock:
        wait = _last[0] + MIN_INTERVAL - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        _last[0] = time.monotonic()

def _wait_from(msg: str) -> float:
    m = re.search(r"(?:try again|retry) in ((?:[\d.]+(?:ms|h|m|s)\s*)+)", msg, re.I)
    if not m:
        return 10.0
    total = 0.0
    for num, unit in re.findall(r"([\d.]+)(ms|h|m|s)", m.group(1)):
        v = float(num)
        total += v / 1000 if unit == "ms" else v * 3600 if unit == "h" else v * 60 if unit == "m" else v
    return total + 1.0

def chat(system: str, user: str, json_mode: bool = False, retries: int = 8) -> str:
    kwargs = dict(model=config.LLM_MODEL, temperature=0,
                  messages=[{"role": "system", "content": system},
                            {"role": "user", "content": user}])
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    if EFFORT:
        kwargs["reasoning_effort"] = EFFORT
    for attempt in range(retries):
        _throttle()
        try:
            return _client.chat.completions.create(**kwargs).choices[0].message.content or ""
        except AuthenticationError as e:
            raise AuthFailed(f"Authentication failed - check LLM_API_KEY / LLM_BASE_URL in .env: {e}")
        except RateLimitError as e:
            wait = _wait_from(str(e))
            if wait > MAX_WAIT:
                raise RateLimitExhausted(f"Provider asks to wait {wait/60:.0f} min: {e}")
            time.sleep(wait)
        except BadRequestError as e:
            if "response_format" in kwargs and "response_format" in str(e):
                kwargs.pop("response_format")       # provider lacks JSON mode: fall back
                continue
            if "reasoning_effort" in kwargs and "reasoning_effort" in str(e):
                kwargs.pop("reasoning_effort")
                continue
            raise
        except Exception as e:
            print(f"LLM request failed: {e}")
            if attempt >= 2:
                raise
            time.sleep(2 ** attempt)
    raise RateLimitExhausted("LLM retries exhausted")

def chat_json(system: str, user: str):
    """Returns parsed JSON (dict or list), or None if the model produced invalid JSON."""
    raw = chat(system, user, json_mode=True)
    try:
        return json.loads(raw)
    except Exception:
        m = re.search(r"[\{\[].*[\}\]]", raw, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                return None
    return None
