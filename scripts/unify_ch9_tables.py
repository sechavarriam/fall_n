import re, sys

p = r"c:\MyLibs\fall_n\PhD_Thesis\capitulos\9_modelos_especificos_y_validacion.tex"
with open(p, encoding="utf-8") as f:
    text = f.read()
lines = text.split("\n")

starts = [i for i, l in enumerate(lines) if l.strip().startswith(r"\begin{ValidationWideTable}")]
ends   = [i for i, l in enumerate(lines) if l.strip().startswith(r"\end{ValidationWideTable}")]
assert len(starts) == len(ends), (len(starts), len(ends))
pairs = list(zip(starts, ends))

mods_a = mods_b = mods_c = 0
for s, e in reversed(pairs):
    block = lines[s:e+1]
    label_idx_rel = None
    for k, ln in enumerate(block):
        if re.search(r"\\label\{tab:validacion-", ln):
            label_idx_rel = k
            break
    if label_idx_rel is None:
        continue

    tail = "\n".join(block[-4:])
    if (r"\addtocounter{table}{-1}" not in tail
            and r"\setcounter{table}{\value{table}-1}" not in tail):
        lines.insert(e, "  \\addtocounter{table}{-1}")
        mods_b += 1

    abs_label = s + label_idx_rel
    nxt = lines[abs_label+1] if abs_label+1 < len(lines) else ""
    if r"\vspace{-5mm}" not in nxt:
        lines.insert(abs_label+1, "  \\vspace{-5mm}")
        mods_a += 1

    if s > 0:
        prev = lines[s-1].strip()
        if not prev.startswith(r"\vspace{5mm}"):
            lines.insert(s, r"\vspace{5mm}")
            mods_c += 1

with open(p, "w", encoding="utf-8", newline="\n") as f:
    f.write("\n".join(lines))

print("tables:", len(pairs), "a:", mods_a, "b:", mods_b, "c:", mods_c)
