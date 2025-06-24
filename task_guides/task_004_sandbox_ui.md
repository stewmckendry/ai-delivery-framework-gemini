# 🏝️ Task 004: Sandbox Environment UI Support

## Objective
Enable users to initialize and switch between sandbox environments via UI using the `sandboxInit` MCP tool.

## Context
Backed supports `sandboxInit`, but no frontend UI flow exists. Sandboxes let users experiment in forks or throwaway branches securely.

## Scope
- Let user specify upstream repo or fork settings
- Display current sandbox status (branch, repo)
- Call `sandboxInit` tool and store response (e.g. `reuse_token`)
- Allow switching or reusing sandbox branches
- Update prompt and commit flow to reflect current sandbox

## Inputs
- GitHub identity (see Task 003)
- Desired sandbox type (fork, temp repo, branch name)

## Outputs
- New sandbox created or reused
- UI reflects current working repo/branch

## Target Files
- `script.js`, optional new settings UI modal

## Deliverables
- Visible UI control for sandbox management
- Display of current working environment
- Seamless integration with existing file fetch/commit flows

## Related
- Backend tool: `sandbox_init_mcp_tool`
- Depends on: GitHub identity from Task 003

---

Create a PR and reference this task in the title or body.