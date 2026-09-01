"""Routing a tool call to whatever serves it.

The nine tools come from three places: seven are filesystem work in Repo,
api_docs is a documentation lookup, and web_search reaches the internet.
The agent loop should not know that, so it calls one `call` here.

A provider that is missing or unconfigured returns an error to the model
rather than raising. The model can then say it could not look something up,
which is far better than a crashed turn or, worse, a guessed API signature.
"""
from __future__ import annotations

from typing import Any


class Toolbox:
    def __init__(self, repo, docs: Any = None, web: Any = None):
        self.repo = repo
        self.docs = docs
        self.web = web

    @classmethod
    def build(cls, repo, want_docs: bool = True, want_web: bool = True):
        """Attach the providers that import cleanly, skip the rest."""
        docs = web = None
        if want_docs:
            try:
                from macqwen.tools.context7 import Context7

                docs = Context7()
            except Exception:
                docs = None
        if want_web:
            try:
                from macqwen.tools.free_search import FreeSearch
                from macqwen.tools.tavily_search import TavilySearch

                paid = TavilySearch()
                web = paid if getattr(paid, "configured", False) else FreeSearch()
            except Exception:
                web = None
        return cls(repo, docs, web)

    @property
    def missing(self) -> tuple[str, ...]:
        absent = []
        if self.docs is None:
            absent.append("api_docs")
        if self.web is None:
            absent.append("web_search")
        return tuple(absent)

    def call(self, name: str, args: dict):
        if name == "api_docs":
            if self.docs is None:
                return {"error": "api_docs is unavailable in this session; "
                                 "say so rather than recalling a signature"}
            return self.docs.docs(args.get("library", ""), args.get("topic"))
        if name == "web_search":
            if self.web is None:
                return {"error": "web_search is unavailable in this session"}
            return self.web.search(**args)
        return self.repo.call(name, args)
