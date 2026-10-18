"""free-llm: a Claude Code tool that hands work to free-tier AI models.

Claude Code calls `ask_free_model`. This server reads any files you name on
your own computer, sends the prompt to a free provider, and returns only the
answer. So Claude never reads those files and never writes the long answer
itself, which saves your Claude Max usage.

Keys live in ~/.config/free-llm/keys.env, one per line, for example:
    GROQ_API_KEY=gsk_...
A provider with no key is skipped.
"""
import json
import os
from pathlib import Path

from openai import OpenAI

try:  # mcp 2.x
    from mcp.server.mcpserver import MCPServer as Server
except ImportError:  # mcp 1.x
    from mcp.server.fastmcp import FastMCP as Server

KEYS_FILE = Path.home() / ".config" / "free-llm" / "keys.env"
MAX_FILE_CHARS = 400_000  # about 100k tokens across all files in one call

# Order = try order for provider="auto". All use the OpenAI chat format.
# Default models come from the awesome-freellm-apis list (Oct 2026).
# If one is retired, call list_models and pass a new one as `model`.
PROVIDERS = {
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "key": "GROQ_API_KEY",
        "model": "openai/gpt-oss-120b",
    },
    "cerebras": {
        "base_url": "https://api.cerebras.ai/v1",
        "key": "CEREBRAS_API_KEY",
        "model": "gpt-oss-120b",
    },
    "nvidia": {
        "base_url": "https://integrate.api.nvidia.com/v1",
        "key": "NVIDIA_API_KEY",
        "model": "nvidia/nemotron-3-super-120b-a12b",
    },
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "key": "GEMINI_API_KEY",
        "model": "gemini-flash-latest",
    },
    "mistral": {
        "base_url": "https://api.mistral.ai/v1",
        "key": "MISTRAL_API_KEY",
        "model": "mistral-small-latest",
    },
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "key": "OPENROUTER_API_KEY",
        "model": "nvidia/nemotron-3-ultra-550b-a55b:free",
    },
}

# The weekly maintain.py job keeps this file current (new providers, model fixes).
PROVIDERS_FILE = Path(__file__).with_name("providers.json")
if PROVIDERS_FILE.exists():
    PROVIDERS = json.loads(PROVIDERS_FILE.read_text())


def load_keys() -> dict:
    """Keys from the environment, overridden by the keys file if present."""
    keys = {p["key"]: os.environ.get(p["key"], "") for p in PROVIDERS.values()}
    if KEYS_FILE.exists():
        for line in KEYS_FILE.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                name, value = line.split("=", 1)
                keys[name.strip()] = value.strip().strip('"').strip("'")
    return keys


def client_for(name: str, keys: dict) -> OpenAI:
    p = PROVIDERS[name]
    return OpenAI(base_url=p["base_url"], api_key=keys[p["key"]], timeout=120, max_retries=0)


def read_files(file_paths: list[str]) -> str:
    parts, total = [], 0
    for raw in file_paths:
        path = Path(raw).expanduser()
        text = path.read_text(errors="replace")
        total += len(text)
        if total > MAX_FILE_CHARS:
            raise ValueError(f"Files are too large (over {MAX_FILE_CHARS} characters). Send fewer files.")
        parts.append(f"===== FILE: {path} =====\n{text}")
    return "\n\n".join(parts)


server = Server(
    "free-llm",
    instructions=(
        "Use ask_free_model to save Claude usage on bulky, low-risk work: summarising "
        "long files or logs, first drafts, boilerplate code, translations, bulk "
        "rewrites. Pass file_paths instead of pasting file contents. Check the answer "
        "before relying on it. NEVER send confidential company or client code or "
        "data, credentials, or personal data: free tiers may log or train on inputs."
    ),
)


@server.tool()
def ask_free_model(
    prompt: str,
    file_paths: list[str] | None = None,
    provider: str = "auto",
    model: str | None = None,
    max_tokens: int = 4000,
) -> str:
    """Send a task to a free-tier model and return its answer.

    prompt: the full instruction. file_paths: local files the model should read
    (read here, so Claude does not spend tokens on them). provider: "auto" (first
    working provider with a key) or one of groq, cerebras, nvidia, gemini,
    mistral, openrouter. model: override the provider's default model.
    Do not send confidential, client, credential or personal data.
    """
    keys = load_keys()
    content = prompt
    if file_paths:
        content = f"{prompt}\n\n{read_files(file_paths)}"

    if provider == "auto":
        order = [n for n, p in PROVIDERS.items() if keys.get(p["key"]) and p.get("model")]
    elif provider in PROVIDERS:
        order = [provider]
    else:
        return f"Unknown provider '{provider}'. Choose from: auto, {', '.join(PROVIDERS)}."
    if not order or not keys.get(PROVIDERS[order[0]]["key"]):
        return f"No API key found. Add one to {KEYS_FILE} (e.g. GROQ_API_KEY=...)."

    errors = []
    for name in order:
        use_model = model or PROVIDERS[name]["model"]
        try:
            response = client_for(name, keys).chat.completions.create(
                model=use_model,
                messages=[{"role": "user", "content": content}],
                max_tokens=max_tokens,
            )
            answer = response.choices[0].message.content or ""
            if answer.strip():
                return f"[answered by {name} / {use_model}]\n{answer}"
            errors.append(f"{name}: empty answer")
        except Exception as error:  # rate limit, retired model, outage: try next
            errors.append(f"{name} ({use_model}): {str(error)[:200]}")
    return "All providers failed:\n" + "\n".join(errors)


@server.tool()
def list_free_providers() -> str:
    """Show which free providers have a key set and their default models."""
    keys = load_keys()
    lines = [
        f"{name}: {'ready' if keys.get(p['key']) else 'no key (' + p['key'] + ')'}, default model {p['model']}"
        for name, p in PROVIDERS.items()
    ]
    return f"Keys file: {KEYS_FILE}\n" + "\n".join(lines)


@server.tool()
def list_models(provider: str) -> str:
    """List the model IDs a provider currently offers to your key."""
    keys = load_keys()
    if provider not in PROVIDERS:
        return f"Unknown provider '{provider}'. Choose from: {', '.join(PROVIDERS)}."
    if not keys.get(PROVIDERS[provider]["key"]):
        return f"No key for {provider}. Add {PROVIDERS[provider]['key']}=... to {KEYS_FILE}."
    try:
        ids = sorted(m.id for m in client_for(provider, keys).models.list())
    except Exception as error:
        return f"Could not list models: {error}"
    return "\n".join(ids)


def main() -> None:
    server.run()


if __name__ == "__main__":
    main()
