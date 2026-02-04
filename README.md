# Amplifier Codex Provider Module

Codex CLI integration for Amplifier using `codex exec --json` (subscription-based).

## Purpose

Provides a Codex-backed LLM provider for Amplifier without API keys by calling the
Codex CLI in non-interactive mode and translating JSONL output into Amplifier responses.

## Contract

**Module Type:** Provider
**Mount Point:** `providers`
**Entry Point:** `amplifier_module_provider_codex:mount`

## Install

```bash
amplifier module add provider-codex --source git+https://github.com/shohei81/amplifier-module-provider-codex@main
```

## Prerequisites

- **Python 3.11+**
- **Codex CLI** installed and authenticated
- **Node.js** (for the CLI install)

```bash
npm i -g @openai/codex
codex login
```

## Supported Models

- `gpt-5.2-codex` (default)
- `gpt-5.2`

## Configuration

```toml
[[providers]]
module = "provider-codex"
name = "codex"
config = {
    default_model = "gpt-5.2-codex",
    timeout = 300,
    skip_git_repo_check = true,
    profile = null,     # Optional Codex CLI profile name
    sandbox = "read-only",     # Optional CLI sandbox mode
    full_auto = false,  # Allow Codex to run commands and edit files
    search = false,     # Optional: enable web search
    ask_for_approval = null, # Optional: Codex CLI approval policy
    network_access = null,   # Optional: override network access (bool)
    add_dir = [],       # Optional: additional writable directories
    reasoning_effort = "medium" # Optional: none | low | medium | high | xhigh
}
```

## Permissions

Codex CLI defaults are determined by your Codex CLI profile when `sandbox` is unset.
We recommend **read-only** for non-interactive runs and **do not rely on Codex CLI
approvals** in non-interactive execution. For side-effectful actions, rely on
Amplifier tools and (optionally) the `hooks-approval` module rather than Codex CLI
approvals.

Recommended defaults:

- `sandbox = "read-only"`
- `full_auto = false`
- `search = false`
- `network_access = null` (unset → use Codex CLI/profile defaults)

Use a Codex CLI profile (`profile`) to manage more advanced sandbox and network
settings, and only escalate to `workspace-write` or `full_auto` when required.

## Notes

- Uses Codex CLI sessions for caching when available (`~/.amplifier-codex/sessions`).
- Reuses Amplifier session IDs for persisted session files when available, improving
  resume/caching stability across provider restarts.
- Tool calls are emitted as `<tool_use>...</tool_use>` blocks and parsed from JSONL.
- Codex CLI runs in read-only mode by default; set `sandbox` or `full_auto` only if intended.
- This provider supports GPT-5.2 models only. Unsupported model settings are rejected (or defaulted to `gpt-5.2-codex` for `default_model`).
- `reasoning_effort` maps to Codex's `model_reasoning_effort` and supports: `none`, `low`, `medium`, `high`, `xhigh`.
