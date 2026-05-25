"""Optional global CLI entry point.

``maps blindsight ...``, ``maps agl ...``, etc. — a top-level Typer app
that dispatches to each domain's ``cli.py``.

To be decided in Sprint 11+: keep this layer or only co-located CLIs
inside ``maps.domains.<domain>.cli``.

Empty during Sprint 10.
"""
