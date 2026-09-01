# Security policy

## Supported versions

Security fixes target the latest `main` branch and the latest published release.

## Report a vulnerability

Use GitHub private vulnerability reporting for sensitive reports.

Do not open a public issue with credentials, private prompts, session files, or
vulnerability details that enable exploitation.

## API keys

MACQWEN stores managed API keys at:

```text
~/Library/Application Support/MACQWEN/api_keys.json
```

The directory uses mode `0700`. The file uses mode `0600`.

Use `/keys set SERVICE` to enter a key without echo. Do not pass keys inline or
store them in the repository.

Tool, checker, editor, cache, and status child processes receive an environment
with secret-like variables removed.

## Saved sessions

Saved sessions can contain private prompt state, repository text, and model
cache data. Keep these directories private:

```text
~/.cache/flashnext/sessions/
~/.frankenstein/sessions/
```

Do not publish session files.

## Agent tools

The agent profile can read and modify files inside its selected workspace.
Modifying tools require approval by default.

Review the workspace path before approving a command. Use the plain profile
when repository access is unnecessary.

## Web terminal

`web_terminal.py` exposes the local terminal on the network. Its URL contains
an access token.

Use it only on a trusted local network. Do not remove the token or expose the
port to the public internet.

## Local API server

Server mode binds to `127.0.0.1` by default. Localhost mode does not require
authentication. Any local process can send prompts to the loaded model.

A non-local bind requires `MACQWEN_SERVER_API_KEY` or `--server-api-key`.
Prefer the environment variable because command arguments can appear in process
lists and shell history. The API key is compared in constant time.

Use the server only on a trusted network. Do not expose it directly to the
public internet.

### Browser pages

A page on any website can send a request to a server bound to localhost. The
firewall does not stop it because the request comes from the local browser.

The server refuses a request with an `Origin` header unless startup settings
allow that origin:

```bash
./chat.sh --server --allow-origin http://localhost:3000
```

Without `--allow-origin`, a website cannot use the model or read its replies.
Clients that send no `Origin`, including command line tools and SDKs, remain
unaffected.

`--allow-origin '*'` accepts any page. Do not use it for normal operation.

The server returns tool calls to the client instead of running them. A web page
cannot use this endpoint to execute a tool. Tools run only in the chat agent
profile, which requires approval.

## Models and dependencies

This repository does not contain model weights. Review each checkpoint license
and source before downloading it.

Python dependencies and model code execute locally with the operator's permissions.
Install them only from trusted sources.
