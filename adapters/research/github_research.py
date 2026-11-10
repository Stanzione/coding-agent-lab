"""Bounded public GitHub code-search evidence adapter."""

import json
from datetime import datetime, timezone
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

_SEARCH_URL = "https://api.github.com/search/code"
_MAX_REQUESTS = 3
_MAX_RESULTS = 3
_MAX_QUERY_LENGTH = 180
_MAX_SNIPPET_LENGTH = 500


class GitHubResearch:
    """Search public GitHub code with per-instance request and result limits."""

    def __init__(self, token: str | None = None) -> None:
        self._token = token
        self._request_count = 0
        self._result_count = 0

    def search_github(self, query: str) -> dict[str, object]:
        """Return small public-source evidence for one validated query."""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("GitHub search query must not be empty")
        if "\n" in query or "\r" in query:
            raise ValueError("GitHub search query must be a single line")
        if len(query) > _MAX_QUERY_LENGTH:
            raise ValueError("GitHub search query must be at most 180 characters")
        if self._token and self._token in query:
            raise ValueError("GitHub search query must not contain the access token")
        if self._request_count >= _MAX_REQUESTS:
            raise RuntimeError("GitHub research request budget exhausted")
        remaining_results = _MAX_RESULTS - self._result_count
        if remaining_results <= 0:
            raise RuntimeError("GitHub research result budget exhausted")

        self._request_count += 1
        parameters = urlencode({"q": query, "per_page": remaining_results})
        headers = {
            "Accept": "application/vnd.github.text-match+json",
            "User-Agent": "coding-agent-lab",
        }
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"
        request = Request(
            f"{_SEARCH_URL}?{parameters}",
            headers=headers,
            method="GET",
        )
        try:
            with urlopen(request, timeout=15) as response:
                data = json.loads(response.read())
        except HTTPError as error:
            raise RuntimeError(f"GitHub code search failed with HTTP {error.code}") from error
        except (URLError, TimeoutError) as error:
            raise RuntimeError("GitHub code search request failed") from error
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            raise RuntimeError("GitHub code search returned invalid JSON") from error

        items = data.get("items") if isinstance(data, dict) else None
        if not isinstance(items, list):
            raise RuntimeError("GitHub code search response has no result list")

        sources: list[dict[str, str]] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            repository = item.get("repository")
            if not isinstance(repository, dict) or repository.get("private") is True:
                continue
            repository_name = repository.get("full_name")
            path = item.get("path")
            url = item.get("html_url")
            if (
                not isinstance(repository_name, str)
                or not isinstance(path, str)
                or not isinstance(url, str)
                or not url.startswith("https://github.com/")
            ):
                continue
            if self._token and self._token in f"{repository_name}{path}{url}":
                continue
            text_matches = item.get("text_matches", [])
            snippet = ""
            if isinstance(text_matches, list):
                for match in text_matches:
                    if isinstance(match, dict) and isinstance(match.get("fragment"), str):
                        snippet = match["fragment"][:_MAX_SNIPPET_LENGTH]
                        break
            if self._token:
                snippet = snippet.replace(self._token, "[redacted]")
            sources.append(
                {
                    "repository": repository_name,
                    "path": path,
                    "url": url,
                    "snippet": snippet,
                }
            )
            if len(sources) >= remaining_results:
                break

        self._result_count += len(sources)
        return {
            "query": query,
            "retrieved_at": datetime.now(timezone.utc).isoformat(),
            "sources": sources,
        }
