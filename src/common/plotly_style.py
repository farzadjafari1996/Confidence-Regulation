"""Shared Plotly layout helpers (title wrapping / full-page HTML)."""

from __future__ import annotations

import pathlib
import re

import plotly.graph_objects as go


def wrap_plotly_title(text: str, *, max_line: int = 64) -> str:
    """Insert <br> breaks so long figure titles fit the plot width.

    Prefers natural breaks after an em/en dash separator and before a trailing
    parenthetical; falls back to wrapping on spaces near ``max_line``.
    """
    if not text or "<br>" in text.lower():
        return text

    plain = re.sub(r"<[^>]+>", "", text)
    if len(plain) <= max_line:
        return text

    for sep in (" — ", " – ", " - "):
        if sep in text:
            left, right = text.split(sep, 1)
            right = right.lstrip()
            if " (" in right and right.rstrip().endswith(")"):
                main, paren = right.rsplit(" (", 1)
                right = f"{main}<br>({paren}"
            return f"{left}<br>{right}"

    parts: list[str] = []
    buf = ""
    plain_len = 0
    i = 0
    while i < len(text):
        if text[i] == "<":
            j = text.find(">", i)
            chunk = text[i : (j + 1 if j >= 0 else len(text))]
            buf += chunk
            i += len(chunk)
            continue
        ch = text[i]
        buf += ch
        plain_len += 1
        if plain_len >= max_line and ch == " ":
            parts.append(buf.rstrip())
            buf = ""
            plain_len = 0
        i += 1
    if buf:
        parts.append(buf)
    return "<br>".join(parts) if len(parts) > 1 else text


def apply_readable_title(fig: go.Figure, *, max_line: int = 64) -> go.Figure:
    """Wrap + center the layout title and reserve top margin for multi-line text."""
    layout = fig.layout
    raw = layout.title.text if layout.title is not None else None
    if not raw:
        # Still fix grouped box overlays even when there is no title.
        _ensure_grouped_boxmode(fig)
        fig.update_layout(autosize=True)
        fig.layout.width = None
        fig.layout.height = None
        return fig

    wrapped = wrap_plotly_title(str(raw), max_line=max_line)
    n_lines = wrapped.lower().count("<br>") + 1
    top = max(int(layout.margin.t or 0), 56 + 26 * n_lines)
    _ensure_grouped_boxmode(fig)
    fig.update_layout(
        title=dict(text=wrapped, x=0.5, xanchor="center"),
        margin_t=top,
        autosize=True,
    )
    # Drop fixed pixel size so the figure can fill the browser window.
    fig.layout.width = None
    fig.layout.height = None
    return fig


def _ensure_grouped_boxmode(fig: go.Figure) -> None:
    """Keep Total/Direct (etc.) box traces side-by-side instead of overlaid."""
    box_traces = [t for t in fig.data if getattr(t, "type", None) == "box"]
    if len(box_traces) < 2:
        return
    groups = {getattr(t, "offsetgroup", None) or getattr(t, "name", None) for t in box_traces}
    if len(groups) >= 2 and not fig.layout.boxmode:
        fig.update_layout(boxmode="group")


def save_plotly_html(fig: go.Figure, path: pathlib.Path | str, *, config: dict | None = None) -> None:
    """Write a full-window responsive HTML plot with a wrapped title."""
    apply_readable_title(fig)
    cfg = {"responsive": True, **(config or {})}
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(
        str(path),
        config=cfg,
        include_plotlyjs=True,
        full_html=True,
        default_width="100%",
        default_height="100vh",
    )
    # Ensure the page itself uses the viewport (Plotly div is already 100%/100vh).
    html = path.read_text(encoding="utf-8")
    if "html, body { margin:0; height:100%; }" not in html:
        html = html.replace(
            "<head><meta charset=\"utf-8\" /></head>",
            "<head><meta charset=\"utf-8\" />"
            "<style>html, body { margin:0; height:100%; background:#fff; }</style></head>",
            1,
        )
        path.write_text(html, encoding="utf-8")
