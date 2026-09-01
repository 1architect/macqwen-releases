"""One command table for every model and profile.

The two chats grew separate command sets, so a command added to one was
missing from the other: Flash-Next had /thinking and /max-tokens, the 27B
chat had /effort and /stream, and neither had the other's. This is the
union, declared once.

A command is a name, its aliases, a usage line and a handler. The handler
takes the session and the argument text and returns what to print. It
never prints directly, so the same table can drive a terminal, a test, or
anything else.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from macqwen.preferences import (
    DEFAULT_ANSWER_TOKENS,
    DEFAULT_PLAIN_ANSWER_TOKENS,
)
from macqwen.ui import token_limit_text


@dataclass
class Command:
    names: tuple[str, ...]
    usage: str
    summary: str
    handler: Callable[[Any, str], str]
    profiles: tuple[str, ...] = ("plain", "agent")

    @property
    def name(self) -> str:
        return self.names[0]


def _on_off(value: str) -> bool | None:
    if value in ("on", "enable", "yes"):
        return True
    if value in ("off", "disable", "no"):
        return False
    return None


def _thinking(session, argument: str) -> str:
    prefs = session.preferences
    if not argument:
        return (f"thinking: {'on' if prefs['thinking_enabled'] else 'off'}, "
                f"display: {'show' if prefs['show_thinking'] else 'hide'}")
    state = _on_off(argument)
    if state is not None:
        prefs["thinking_enabled"] = state
    elif argument == "show":
        prefs["show_thinking"] = True
    elif argument == "hide":
        prefs["show_thinking"] = False
    else:
        return "usage: /thinking on|off|show|hide"
    session.save_preferences()
    return (f"thinking: {'on' if prefs['thinking_enabled'] else 'off'}, "
            f"display: {'show' if prefs['show_thinking'] else 'hide'}")


def _max_tokens(session, argument: str) -> str:
    prefs = session.preferences
    if argument in ("off", "none", "unlimited", "-1"):
        prefs["max_tokens"] = -1
    elif argument:
        try:
            limit = int(argument)
        except ValueError:
            limit = 0
        if limit <= 0:
            return "usage: /max-tokens N|off"
        prefs["max_tokens"] = limit
    session.save_preferences()
    shown = token_limit_text(prefs["max_tokens"])
    if prefs["max_tokens"] < 0:
        default = (
            DEFAULT_PLAIN_ANSWER_TOKENS
            if session.profile == "plain"
            else DEFAULT_ANSWER_TOKENS
        )
        shown = f"default ({default})"
    return f"max answer tokens: {shown}"


def _think_budget(session, argument: str) -> str:
    prefs = session.preferences
    if argument in ("off", "none", "-1"):
        prefs["think_budget"] = -1
    elif argument:
        try:
            limit = int(argument)
        except ValueError:
            limit = 0
        if limit <= 0:
            return "usage: /think-budget N|off"
        prefs["think_budget"] = limit
    session.save_preferences()
    return f"thinking tokens: {token_limit_text(prefs['think_budget'])}"


def _effort(session, argument: str) -> str:
    prefs = session.preferences
    if not argument:
        return f"effort: {prefs['effort']}"
    if argument not in ("low", "medium", "xhigh"):
        return "usage: /effort low|medium|xhigh"
    prefs["effort"] = argument
    session.save_preferences()
    return f"effort: {argument}"


def _stream(session, argument: str) -> str:
    prefs = session.preferences
    state = _on_off(argument) if argument else None
    if argument and state is None:
        return "usage: /stream on|off"
    if state is not None:
        prefs["stream_answers"] = state
        session.save_preferences()
    return f"stream: {'on' if prefs['stream_answers'] else 'off'}"


def _animate(session, argument: str) -> str:
    prefs = session.preferences
    state = _on_off(argument) if argument else None
    if argument and state is None:
        return "usage: /animate on|off"
    if state is not None:
        prefs["animate"] = state
        session.save_preferences()
    return f"animate: {'on' if prefs['animate'] else 'off'}"


def _approval(session, argument: str) -> str:
    prefs = session.preferences
    if argument and argument not in ("ask", "auto"):
        return "usage: /approval ask|auto"
    if argument:
        prefs["approval"] = argument
        session.save_preferences()
    return f"approval: {prefs['approval']}"


def _workspace(session, argument: str) -> str:
    if not argument:
        return f"workspace: {session.preferences['workspace']}"
    session.preferences["workspace"] = argument
    session.save_preferences()
    return f"workspace: {argument}  (takes effect on the next /reset)"


def _profile(session, argument: str) -> str:
    if not argument:
        return f"profile: {session.profile}"
    if argument not in ("plain", "agent"):
        return "usage: /profile plain|agent"
    changed = session.set_profile(argument)
    return f"profile: {argument}" + ("; conversation reset" if changed else "")


def _prompt(session, argument: str) -> str:
    if not argument:
        return (
            f"{session.current_system_prompt()}\n\n"
            f"prompt file: {session.system_prompt_path()}\n"
            "edit this file, then use /reset"
        )
    if argument == "default":
        session.set_system_prompt("")
        return "prompt back to default" + ("  /reset to apply" if session.opened else "")
    if argument == "edit":
        session.edit_system_prompt()
    else:
        session.set_system_prompt(argument)
    return "prompt updated" + ("  /reset to apply" if session.opened else "")


def _keys(session, argument: str) -> str:
    parts = argument.split()
    if not parts or parts == ["list"]:
        return session.list_api_keys()
    if len(parts) == 2 and parts[0] == "set":
        return session.set_api_key(parts[1])
    if len(parts) == 2 and parts[0] in ("delete", "remove"):
        return session.delete_api_key(parts[1])
    return "usage: /keys [list|set SERVICE|delete SERVICE]"


def _status(session, argument: str) -> str:
    return session.status()


def _settings(session, argument: str) -> str:
    return session.model_settings(argument)


def _server(session, argument: str) -> str:
    if argument:
        return "usage: /server"
    session.start_server()
    return "starting the local API server"


def _reset(session, argument: str) -> str:
    session.reset()
    return "conversation reset, model stayed loaded"


def _save(session, argument: str) -> str:
    return session.save_session(argument.strip() or "last")


def _load(session, argument: str) -> str:
    return session.load_session(argument.strip() or "last")


def _sessions(session, argument: str) -> str:
    return session.list_sessions()


def _delete(session, argument: str) -> str:
    if not argument.strip():
        return "usage: /delete NAME"
    return session.delete_session(argument.strip())


def _help(session, argument: str) -> str:
    return render_help(session.profile)


def _quit(session, argument: str) -> str:
    session.stop()
    return ""


COMMANDS: tuple[Command, ...] = (
    Command(("/thinking", "/think"), "/thinking on|off|show|hide",
            "enable or hide model reasoning", _thinking),
    Command(("/max-tokens",), "/max-tokens N|off",
            "set the answer limit", _max_tokens),
    Command(("/think-budget",), "/think-budget N|off",
            "set extra reasoning capacity", _think_budget),
    Command(("/effort",), "/effort low|medium|xhigh",
            "how hard the model should think", _effort),
    Command(("/stream",), "/stream on|off",
            "stream the answer as it is written", _stream),
    Command(("/animate",), "/animate on|off",
            "fade complete words in or show them immediately", _animate),
    Command(("/approval",), "/approval ask|auto",
            "confirm before a tool changes anything", _approval, ("agent",)),
    Command(("/workspace",), "/workspace PATH",
            "which repository the tools may touch", _workspace, ("agent",)),
    Command(("/profile",), "/profile plain|agent",
            "view or change the chat profile", _profile),
    Command(("/prompt",), "/prompt [text|edit|default]",
            "view or change the system prompt", _prompt),
    Command(("/keys", "/api-keys"), "/keys [list|set SERVICE|delete SERVICE]",
            "manage private API keys", _keys),
    Command(("/status",), "/status", "settings, context and memory", _status),
    Command(("/settings",), "/settings [NAME VALUE|defaults]",
            "view or change model settings", _settings),
    Command(("/server",), "/server",
            "leave chat and start the local API server", _server),
    Command(("/reset",), "/reset", "forget the conversation, keep the model", _reset),
    Command(("/save",), "/save [name]", "save the live model state", _save),
    Command(("/load",), "/load [name]", "restore a saved state", _load),
    Command(("/sessions",), "/sessions", "list saved states", _sessions),
    Command(("/delete",), "/delete NAME", "delete a saved state", _delete),
    Command(("/help",), "/help", "show this list", _help),
    Command(("/quit", "/exit", "/q"), "/quit", "leave the chat", _quit),
)

BY_NAME: dict[str, Command] = {
    name: command for command in COMMANDS for name in command.names
}


def available(profile: str) -> tuple[Command, ...]:
    return tuple(c for c in COMMANDS if profile in c.profiles)


def render_help(profile: str) -> str:
    commands = available(profile)
    usage_width = max(len(command.usage) for command in commands) + 2
    lines = []
    for command in commands:
        lines.append(f"  {command.usage:<{usage_width}}{command.summary}")
    return "\n".join(lines)


def dispatch(session, text: str) -> str | None:
    """Run a command. Returns what to print, or None if this is not a command.

    A multi-line paste is never a command, however it starts: pasted code
    beginning with a slash must reach the model, not the command table.
    """
    if "\n" in text or not text.strip().startswith("/"):
        return None
    parts = text.strip().split(maxsplit=1)
    command = BY_NAME.get(parts[0])
    if command is None:
        return f"unknown command {parts[0]}  use /help"
    if session.profile not in command.profiles:
        return f"{parts[0]} is only available in the agent profile"
    return command.handler(session, parts[1].strip() if len(parts) > 1 else "")
