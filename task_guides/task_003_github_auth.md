# 🔐 Task 003: GitHub OAuth + Identity

## Objective
Enable GitHub authentication so users interact with ProductPod using their GitHub identity and repos.

## Context
Currently, all tool calls are anonymous and require manual token injection. A proper OAuth flow allows users to grant access securely and links actions to their GitHub account.

## Scope
- Add OAuth flow in frontend
- Create backend OAuth redirect/callback handler
- Store access token (in session, localStorage, or cookie)
- Use token in all backend tool calls (pass via headers)
- Update Gemini handler to respect identity (already has token plumbing)

## Inputs
- GitHub OAuth App (client ID and secret)
- Redirect URI (e.g. `/auth/callback`)

## Outputs
- Authenticated GitHub token usable in tool requests
- User identity (GitHub login) tracked for context/logging

## Target Files
- `app/frontend/index.html`, `script.js`
- `app/backend/main.py` (add OAuth route)

## Deliverables
- Working login/logout buttons in UI
- Secure token storage and forwarding
- Graceful auth error handling

## Related
- GitHub OAuth docs: https://docs.github.com/en/apps/oauth-apps/building-oauth-apps/authorizing-oauth-apps
- Existing backend expects `Authorization: Bearer <token>` or `X-GitHub-Token`

---

Create a PR and reference this task in the title or body.