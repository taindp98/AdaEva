"""Docstring-insensitive canonical key for a heuristic's source code.

Why this exists: the tiny/reprod EoH profilers log a candidate's source as ``str(func)``
in ``register_function`` (WITH the docstring the LLM/template carried), but the surviving
``Function`` objects captured from ``pop.population`` in ``_snapshot_population`` have their
docstring stripped, so ``str(f)`` differs. Matching survivors back to heuristics.json by the
raw source string then fails (cand_id=None) for exactly the candidates whose logged source
kept a docstring.

``canon_source`` normalizes a source string to a docstring-insensitive key (function name +
signature + body, no docstring), so a survivor and its heuristics.json entry map to the same
key regardless of whether either side carried a docstring. Falls back to the raw string when
parsing fails, so an unparseable source still keys consistently to itself.
"""

from __future__ import annotations


def canon_source(source: str) -> str:
    """Return a docstring-insensitive canonical key for ``source``."""
    if not source:
        return ""
    try:
        from llm4ad.base.code import TextFunctionProgramConverter as _C
        f = _C.text_to_function(source)
        if f is None:
            return source
        # Drop the docstring so a survivor (docstring stripped) and its logged heuristic
        # (docstring kept) canonicalize identically.
        try:
            f.docstring = None
        except Exception:
            pass
        return str(f)
    except Exception:
        return source
