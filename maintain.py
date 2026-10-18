"""Weekly upkeep for the free-llm tool. Runs on your Mac with no Claude usage.

1. Tests every provider that has a key. If its model is retired, picks a
   working replacement from the provider's own model list and saves it.
2. Reads the awesome-freellm-apis list and adds new providers (no key yet).
3. Notifies you only when a key stopped working (needs you) or new free
   providers appeared (optional: add a key to use one).

Writes providers.json (read by server.py) and REPORT.md next to this file.
"""
import json
import re
import subprocess
import sys
import urllib.request
from datetime import date
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import server  # noqa: E402  (shares keys file, client and built-in defaults)

PROVIDERS_FILE = HERE / "providers.json"
REPORT_FILE = HERE / "REPORT.md"
LIST_URL = "https://raw.githubusercontent.com/open-free-llm-api/awesome-freellm-apis/main/README.md"

# Model names containing these are not general chat models.
SKIP_WORDS = (
    "embed", "guard", "safety", "whisper", "tts", "audio", "transcribe", "image",
    "rerank", "orpheus", "lyria", "banana", "robotics", "live", "deplot", "kosmos",
    "fuyu", "detector", "retriever", "calibration", "computer-use", "research",
    "aqa", "ocr", "moderation", "vision", "vl-", "-vl", "omni", "antigravity",
)
# Earlier word = preferred family when choosing a replacement model.
PREFER = (
    "flash-latest", "gpt-oss-120b", "nemotron-3-ultra", "nemotron-3-super", "kimi",
    "deepseek", "qwen", "glm", "gemini", "mistral-medium", "mistral-small",
    "llama", "gemma", "gpt-oss", "mistral",
)
RETIRED_HINTS = ("404", "410", "does not exist", "archived", "not found", "gone", "decommission", "deprecated")
KEY_HINTS = ("401", "403", "invalid api key", "unauthorized", "forbidden")


def load_providers() -> dict:
    if PROVIDERS_FILE.exists():
        return json.loads(PROVIDERS_FILE.read_text())
    return {name: dict(p) for name, p in server.PROVIDERS.items()}


def call(name: str, p: dict, model: str, keys: dict) -> str:
    """Return '' on success, else the error text."""
    try:
        client = server.OpenAI(base_url=p["base_url"], api_key=keys[p["key"]], timeout=60, max_retries=0)
        r = client.chat.completions.create(
            model=model, messages=[{"role": "user", "content": "Reply with just: hello"}], max_tokens=300
        )
        return "" if (r.choices[0].message.content or "").strip() else "empty answer"
    except Exception as error:
        return str(error)[:300]


def ranked_models(p: dict, keys: dict) -> list[str]:
    client = server.OpenAI(base_url=p["base_url"], api_key=keys[p["key"]], timeout=60, max_retries=0)
    ids = [m.id.removeprefix("models/") for m in client.models.list()]
    ids = [i for i in ids if not any(w in i.lower() for w in SKIP_WORDS)]

    def score(model_id: str):
        low = model_id.lower()
        rank = next((n for n, w in enumerate(PREFER) if w in low), len(PREFER))
        return (rank, "preview" in low, len(model_id))

    return sorted(ids, key=score)


def check_providers(providers: dict, keys: dict, report: list, human: list) -> None:
    for name, p in providers.items():
        if not keys.get(p["key"]):
            continue
        error = call(name, p, p["model"], keys) if p.get("model") else "no model set"
        if not error:
            report.append(f"- {name}: OK ({p['model']})")
            continue
        low = error.lower()
        if any(h in low for h in KEY_HINTS):
            report.append(f"- {name}: key rejected")
            human.append(f"Replace the {p['key']} key in {server.KEYS_FILE} ({name} rejected it).")
            continue
        if not (any(h in low for h in RETIRED_HINTS) or error in ("empty answer", "no model set")):
            report.append(f"- {name}: temporary problem, left as is ({error[:120]})")
            continue
        try:
            candidates = ranked_models(p, keys)[:6]
        except Exception as list_error:
            report.append(f"- {name}: model retired and model list failed ({str(list_error)[:120]})")
            continue
        for model in candidates:
            if not call(name, p, model, keys):
                report.append(f"- {name}: model {p.get('model')} retired, switched to {model}")
                p["model"] = model
                break
        else:
            report.append(f"- {name}: no working model found among {candidates}")


def discover(providers: dict, keys: dict, report: list, optional: list) -> None:
    try:
        try:  # use the Mac's own certificate store (python.org Python lacks one)
            import ssl
            import truststore
            context = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        except ImportError:
            context = None
        text = urllib.request.urlopen(LIST_URL, timeout=60, context=context).read().decode()
    except Exception as error:
        report.append(f"- Could not read the provider list: {error}")
        return
    start = text.find("## Quick Reference")
    if start < 0:
        report.append("- Provider list format changed; discovery skipped this week.")
        return
    section = text[start:]
    end = section.find("\n## ", 3)  # stop at the next heading
    section = section[:end] if end > 0 else section
    known_hosts = {p["base_url"].split("/")[2] for p in providers.values()}
    for row in section.splitlines():
        m = re.match(r"\|\s*([^|]+?)\s*\|\s*`(https://[^`]+)`\s*\|\s*[^|]*?(?:href=\"([^\"]+)\"|\]\(([^)]+)\))", row)
        if not m:
            continue
        title, base_url, signup = m.group(1), m.group(2), m.group(3) or m.group(4)
        host = base_url.split("/")[2]
        if "{" in base_url or host in known_hosts:
            continue  # needs extra setup, or already known
        slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
        key = slug.upper().replace("-", "_") + "_API_KEY"
        providers[slug] = {"base_url": base_url, "key": key, "model": None, "signup": signup,
                           "added": date.today().isoformat()}
        known_hosts.add(host)
        report.append(f"- New provider added: {title} ({base_url})")
        optional.append(f"{title}: sign up at {signup}, then paste the key after {key}= in {server.KEYS_FILE}.")
    # Make sure every provider has a line in the keys file, ready to paste into.
    existing = server.KEYS_FILE.read_text() if server.KEYS_FILE.exists() else ""
    missing = [p["key"] for p in providers.values() if f"{p['key']}=" not in existing]
    if missing:
        server.KEYS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with server.KEYS_FILE.open("a") as f:
            f.write("".join(f"\n{k}=" for k in missing) + "\n")


def notify(message: str) -> None:
    if sys.platform == "darwin":
        script = f'display notification "{message}" with title "free-llm needs you"'
        subprocess.run(["osascript", "-e", script], check=False)


def main() -> None:
    providers = load_providers()
    keys = server.load_keys()
    report, human, optional = [], [], []
    check_providers(providers, keys, report, human)
    discover(providers, keys, report, optional)
    PROVIDERS_FILE.write_text(json.dumps(providers, indent=2) + "\n")

    ready = sum(1 for p in providers.values() if keys.get(p["key"]) and p.get("model"))
    lines = [f"# free-llm weekly check, {date.today().isoformat()}", "",
             f"{ready} providers ready out of {len(providers)} known.", "", "## What happened", *report, ""]
    lines += ["## Needs you", *(f"- {h}" for h in human)] if human else ["## Needs you", "- Nothing. All automatic."]
    if optional:
        lines += ["", "## New free providers (optional: add a key to use one)", *(f"- {o}" for o in optional)]
    REPORT_FILE.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    if human:
        notify(f"{len(human)} item(s) need you. See REPORT.md in free-llm-tool.")
    elif optional:
        notify(f"{len(optional)} new free provider(s) found. Optional. See REPORT.md in free-llm-tool.")


if __name__ == "__main__":
    main()
