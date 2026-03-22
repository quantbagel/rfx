"""Minimal control-plane client for robot registration."""

from __future__ import annotations

import json
import os
from typing import Any
from urllib import error, request


def _base_url(url: str | None) -> str:
    resolved = (url or os.environ.get("RFX_PLATFORM_URL", "")).strip().rstrip("/")
    if not resolved:
        raise ValueError("Platform URL is required. Set --url or RFX_PLATFORM_URL.")
    return resolved


def _api_key(value: str | None) -> str:
    resolved = (value or os.environ.get("RFX_PLATFORM_API_KEY", "")).strip()
    if not resolved:
        raise ValueError("Platform API key is required. Set --api-key or RFX_PLATFORM_API_KEY.")
    return resolved


def _request_json(
    base_url: str,
    path: str,
    payload: dict[str, Any] | None = None,
    *,
    api_key: str,
    method: str = "POST",
) -> dict[str, Any] | list[dict[str, Any]]:
    body = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = request.Request(
        f"{base_url}{path}",
        data=body,
        method=method,
        headers={
            "X-API-Key": api_key,
            "User-Agent": "rfx-sdk-connect/0.2.0",
            **({"Content-Type": "application/json"} if payload is not None else {}),
        },
    )
    try:
        with request.urlopen(req, timeout=10) as response:
            raw = response.read().decode("utf-8")
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{exc.code} {detail}".strip()) from exc
    except error.URLError as exc:
        raise RuntimeError(f"Failed to reach platform: {exc.reason}") from exc
    return json.loads(raw) if raw else {}


def register_robot(
    *,
    url: str | None,
    api_key: str | None,
    robot_id: str,
    robot_kind: str,
    display_name: str,
    transport: str,
    hostname: str,
    sdk_version: str,
    metadata: dict[str, str],
    org_id: str = "",
) -> dict[str, Any]:
    return _request_json(
        _base_url(url),
        "/api/robots/register",
        {
            "robot_id": robot_id,
            "org_id": org_id,
            "display_name": display_name,
            "robot_kind": robot_kind,
            "transport": transport,
            "hostname": hostname,
            "sdk_version": sdk_version,
            "metadata": metadata,
        },
        api_key=_api_key(api_key),
    )


def heartbeat_robot(
    *,
    url: str | None,
    api_key: str | None,
    robot_id: str,
    transport: str,
    sdk_version: str,
    metadata: dict[str, str],
) -> dict[str, Any]:
    return _request_json(
        _base_url(url),
        "/api/robots/heartbeat",
        {
            "robot_id": robot_id,
            "transport": transport,
            "sdk_version": sdk_version,
            "metadata": metadata,
        },
        api_key=_api_key(api_key),
    )


def disconnect_robot(*, url: str | None, api_key: str | None, robot_id: str) -> dict[str, Any]:
    return _request_json(
        _base_url(url),
        "/api/robots/disconnect",
        {"robot_id": robot_id},
        api_key=_api_key(api_key),
    )


def fetch_region_candidates(*, url: str | None, api_key: str | None) -> list[dict[str, Any]]:
    response = _request_json(
        _base_url(url),
        "/api/regions/candidates",
        None,
        api_key=_api_key(api_key),
        method="GET",
    )
    return response if isinstance(response, list) else []


def submit_probe_results(
    *,
    url: str | None,
    api_key: str | None,
    robot_id: str,
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    response = _request_json(
        _base_url(url),
        "/api/robots/probe-results",
        {
            "robot_id": robot_id,
            "results": results,
        },
        api_key=_api_key(api_key),
    )
    return response if isinstance(response, dict) else {}
