"""Homogeneizar tablenotes a Nota[i]: en italica. v2 con balance de llaves."""

import re
from pathlib import Path

P = Path(r"c:\MyLibs\fall_n\PhD_Thesis\capitulos\9_modelos_especificos_y_validacion.tex")
text = P.read_text(encoding="utf-8")
lines = text.split("\n")

starts = [i for i, l in enumerate(lines) if l.lstrip().startswith(r"\begin{tablenotes}")]
ends = [i for i, l in enumerate(lines) if l.lstrip().startswith(r"\end{tablenotes}")]
assert len(starts) == len(ends)


def unwrap_nota(content: str) -> str:
    """If content begins with `\\textit{\\textbf{Nota...:} ` strip that wrapper.

    Finds the matching `}` of the outer `\\textit{...}` by brace counting and
    returns the inner text. If no such wrapper, returns content unchanged.
    """
    s = content.lstrip()
    lead = content[: len(content) - len(s)]
    m = re.match(r"\\textit\{", s)
    if not m:
        return content
    # Check this is actually a `\textit{\textbf{Nota...:} ...}` opener.
    if not re.match(r"\\textit\{\\textbf\{Nota(?:\s*\d+)?:\}", s):
        return content
    # Walk braces to find matching close.
    depth = 0
    i = 0
    open_idx = None
    while i < len(s):
        ch = s[i]
        if ch == "\\":
            i += 2  # skip escaped char (incl. \{ or \})
            continue
        if ch == "{":
            depth += 1
            if open_idx is None:
                open_idx = i
        elif ch == "}":
            depth -= 1
            if depth == 0:
                # `i` is the matching close of the outer \textit{...}.
                inner = s[open_idx + 1 : i]
                # Strip the `\textbf{Nota...:}` prefix from inner.
                inner = re.sub(r"^\\textbf\{Nota(?:\s*\d+)?:\}\s*", "", inner)
                tail = s[i + 1 :]
                return lead + inner + tail
        i += 1
    return content  # unbalanced; bail.


changed_blocks = 0
# Reverse to preserve indices.
for s_idx, e_idx in reversed(list(zip(starts, ends))):
    block = lines[s_idx + 1 : e_idx]
    # Find \item line indices.
    item_idxs = [k for k, l in enumerate(block) if l.lstrip().startswith(r"\item ")]
    if not item_idxs:
        continue
    n = len(item_idxs)
    spans = [
        (item_idxs[j], item_idxs[j + 1] if j + 1 < n else len(block))
        for j in range(n)
    ]
    # Reconstruct each item.
    new_items = []  # list of (indent, label, content_text)
    for j, (a, b) in enumerate(spans):
        first = block[a]
        m = re.match(r"^(\s*)\\item\s+(.*)$", first, flags=re.DOTALL)
        if not m:
            new_items = None
            break
        indent, first_rest = m.group(1), m.group(2)
        full = first_rest
        for rl in block[a + 1 : b]:
            full += "\n" + rl
        unwrapped = unwrap_nota(full)
        label = "Nota:" if n == 1 else f"Nota {j+1}:"
        new_items.append((indent, label, unwrapped))
    if new_items is None:
        continue
    # Detect if already canonical: all items begin with `\textit{\textbf{<label>} `.
    canonical = True
    for j, (a, b) in enumerate(spans):
        first = block[a]
        label = "Nota:" if n == 1 else f"Nota {j+1}:"
        needle = f"\\item \\textit{{\\textbf{{{label}}} "
        if needle not in first:
            canonical = False
            break
    if canonical:
        continue
    # Build new block lines.
    new_block_lines = block[: spans[0][0]]
    for j, (indent, label, content) in enumerate(new_items):
        wrapped = f"{indent}\\item \\textit{{\\textbf{{{label}}} " + content + "}"
        new_block_lines.extend(wrapped.split("\n"))
    # Trailing portion after last item (should be empty inside tablenotes typically).
    after = block[spans[-1][1] :]
    new_block_lines.extend(after)
    lines[s_idx + 1 : e_idx] = new_block_lines
    changed_blocks += 1

P.write_text("\n".join(lines), encoding="utf-8", newline="\n")
print("blocks:", len(starts), "changed:", changed_blocks)
