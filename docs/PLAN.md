# Codex Provider Permission Model Plan

## Goals

- Keep the Codex provider "natural" to Codex CLI usage: use Codex CLI sandbox + profile config rather than emulating Claude's `--tools ""` behavior.
- Preserve Amplifier as the execution authority: Codex only proposes tool calls; Amplifier decides execution.
- Ensure predictable, safe defaults without relying on Codex approval UI (not available in Amplifier non-interactive runs).
- Make the permission model explicit and stable so another agent can implement without ambiguity.

## Non-Goals

- Implement an interactive approval UI inside the provider.
- Change Amplifier core behavior.
- Depend on Codex CLI undocumented flags.

## Constraints and Assumptions

- Provider is invoked via `codex exec --json` (non-interactive).
- Amplifier does not ship a built-in approval UI; approval can be added via hooks.
- Codex CLI has its own sandbox and approval policies; approval prompts require a UI and are unsafe to enable in non-interactive execution.
- "Full control" in Codex is best achieved by keeping Codex in read-only and restricting tools in Amplifier (tool list).

## Architectural Summary

### Principle

Codex provider should **delegate all side-effectful actions to Amplifier tools**. Codex CLI should be used in read-only by default. Any escalation of filesystem/network permissions should be explicit in the provider config and kept minimal.

### Controls

There are two control planes:

1. **Codex CLI permissions** (sandbox, network, search).
2. **Amplifier tool permissions** (which tools are passed in the request).

Codex CLI permissions are used for *Codex-side* operations. Tool permissions are used for *Amplifier-side* operations.

## Permission Levels (Provider Design)

### Level 0 (Default / Safe)

- Codex CLI: `--sandbox read-only`
- Codex CLI: **do not** set `--full-auto`
- Codex CLI: **avoid** `--ask-for-approval on-request` (non-interactive)
- Codex CLI: `network_access = false` (config default)
- Codex CLI: `--search` disabled
- Amplifier tools: allow only explicitly requested tools

**Intent**: Codex cannot write or execute; Amplifier is the only executor.

### Level 1 (Workspace Write, No Network)

- Codex CLI: `--sandbox workspace-write`
- Codex CLI: `network_access = false`
- Codex CLI: `--search` disabled
- Amplifier tools: allow only explicitly requested tools

**Intent**: Allow Codex to modify files under the workspace, but keep network off.

### Level 2 (Workspace Write + Search)

- Codex CLI: `--sandbox workspace-write`
- Codex CLI: `network_access = false`
- Codex CLI: enable search (CLI `--search` or config `features.web_search_request = true`)
- Amplifier tools: allow only explicitly requested tools

**Intent**: Allow Codex to use web search without full network access.

### Level 3 (Workspace Write + Network)

- Codex CLI: `--sandbox workspace-write`
- Codex CLI: `network_access = true`
- Codex CLI: optional `--search` if needed
- Amplifier tools: allow only explicitly requested tools

**Intent**: Allow Codex to access network for package installs or external APIs. Use sparingly.

### Level 4 (Full Auto, Isolated Only)

- Codex CLI: `--full-auto` (exact meaning to confirm via `codex --help`)
- Codex CLI: `--sandbox workspace-write` (implied by `--full-auto`)
- Amplifier tools: allow only explicitly requested tools

**Intent**: Fully automated runs in CI or isolated environments only.

### Level 5 (Dangerous / Do Not Use)

- Codex CLI: `--sandbox danger-full-access` or `--yolo`

**Intent**: Only for isolated, disposable environments. Not supported by default.

## Provider Configuration (Proposed)

Add explicit configuration fields for clarity and determinism:

- `sandbox` (existing): `read-only | workspace-write | danger-full-access`
- `full_auto` (existing): `true|false`
- `profile` (existing): Codex CLI profile name
- `search` (new): `true|false` to toggle `--search`
- `ask_for_approval` (new): `untrusted | on-failure | on-request | never`
  - Default: **unset** (do not pass)
  - Warning: non-interactive execution should **not** use `on-request`
- `network_access` (new, optional): `true|false`
  - If present, pass via CLI `--config` or rely on profile config
- `add_dir` (new, optional list): additional writable roots

Note: any "approval" behavior should be handled by Amplifier hooks (e.g., hooks-approval),
not by Codex CLI, because CLI approvals need a UI.

## CLI Command Construction (Changes)

### Current

- Uses `--sandbox` and `--full-auto` if configured
- Uses `--profile` if configured

### Proposed

Build command in the following order:

1. base: `codex exec --json --model <model>`
2. session resume args (inserted immediately after `exec`, before flags; existing logic)
3. `--profile <profile>` if set
4. `--sandbox <sandbox>` if set
5. `--full-auto` if set
6. `--ask-for-approval <policy>` if set (avoid on-request in non-interactive)
7. `--search` if `search` is true
8. `--add-dir <path>` for each entry in `add_dir`
9. `--skip-git-repo-check` if configured (existing)

### Config Overrides (Optional)

If `network_access` is set, pass:

- `--config sandbox_workspace_write.network_access=true|false`

If `writable_roots` are configured in provider config (optional extension),
pass via:

- `--config sandbox_workspace_write.writable_roots=[...]`

This keeps profile compatibility while allowing explicit overrides.

## Tool Permissions (Amplifier Side)

No changes required in Codex provider. Continue:

- Build `_valid_tool_names` from `request.tools`
- Filter out tool calls not in that list
- Inject a tool result message to notify rejection (existing pattern)

## Approval Flow (Where It Lives)

- Codex CLI approval prompts are **not reliable** in non-interactive execution.
- If approval is required, it should be implemented in Amplifier via hooks.
- Recommend: add `hooks-approval` to Amplifier profile when deploying.

## README Updates (Planned)

Add a "Permissions" section:

- Clarify default read-only + no network
- Explain `sandbox`, `full_auto`, `search`, `profile`
- Explicitly state that CLI approvals are not used in non-interactive mode
- Recommend hooks-approval for human gating

## Implementation Steps (Agent-Friendly)

1. Add `docs/PLAN.md` (this document).
2. Update `README.md` with a new "Permissions" section referencing this plan.
3. Extend provider config parsing:
   - `search`, `ask_for_approval`, `network_access`, `add_dir`.
4. Update `_build_command` to include new flags (preserve existing order).
5. Add validation warnings:
   - If `ask_for_approval == "on-request"` and `full_auto` is false: log a warning.
   - If `sandbox == "danger-full-access"`: log a warning.
6. Add tests:
   - Verify command includes `--search` when enabled.
   - Verify command includes `--ask-for-approval <policy>` when set.
   - Verify `--add-dir` entries are passed.
   - Verify `--config ...network_access` when provided.

## Acceptance Criteria

- Default behavior remains read-only and non-interactive safe.
- No CLI approval prompts block execution in default config.
- Operator can select a profile or flags to change permissions.
- Tool filtering remains enforced; invalid tool calls are rejected and reported.
