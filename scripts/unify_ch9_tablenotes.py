"""Homogeneizar tablenotes del capitulo 9 al patron Nota[i]: en itálica.

Reglas:
- 1 item   -> \textit{\textbf{Nota:} <contenido>}
- >1 items -> \textit{\textbf{Nota 1:} ...}, \textit{\textbf{Nota 2:} ...}, ...
- Si ya esta en el patron Nota[i]:, renumera por si acaso.
- Preserva indentacion del primer caracter no-blanco del item.
"""

import re
from pathlib import Path

P = Path(r"c:\MyLibs\fall_n\PhD_Thesis\capitulos\9_modelos_especificos_y_validacion.tex")
lines = P.read_text(encoding="utf-8").split("\n")

# Localiza bloques tablenotes.
starts = [i for i, l in enumerate(lines) if l.lstrip().startswith(r"\begin{tablenotes}")]
ends = [i for i, l in enumerate(lines) if l.lstrip().startswith(r"\end{tablenotes}")]
assert len(starts) == len(ends), (len(starts), len(ends))

NOTA_RE = re.compile(r"\\textit\{\\textbf\{Nota(?:\s*\d+)?:\}\s*")

modified_blocks = 0

# Procesar en orden inverso para no perturbar indices.
for s, e in reversed(list(zip(starts, ends))):
    block = lines[s + 1 : e]  # lineas internas

    # Localiza indices (relativos a `block`) de cada \item.
    item_idxs = [k for k, l in enumerate(block) if l.lstrip().startswith(r"\item ")]
    if not item_idxs:
        continue

    n_items = len(item_idxs)

    # Construye rangos [item_idx, next_item_idx) por item.
    spans = []
    for j, idx in enumerate(item_idxs):
        end_rel = item_idxs[j + 1] if j + 1 < n_items else len(block)
        spans.append((idx, end_rel))

    new_block = list(block)  # copia mutable
    changed_local = False

    for j, (a, b) in enumerate(spans):
        prefix_label = "Nota:" if n_items == 1 else f"Nota {j+1}:"
        # Extrae el contenido completo del item (texto de varias lineas).
        # Conserva la indentacion de la linea \item.
        first = new_block[a]
        m = re.match(r"^(\s*)\\item\s+(.*)$", first, flags=re.DOTALL)
        if not m:
            continue
        indent, first_rest = m.group(1), m.group(2)
        rest_lines = new_block[a + 1 : b]

        # Junta el contenido completo del item para inspeccion.
        full_text = first_rest
        for rl in rest_lines:
            full_text += "\n" + rl

        # Elimina cualquier envoltura previa "\textit{\textbf{Nota...:} " al inicio.
        stripped = full_text.lstrip()
        cleaned_prefix_len = len(full_text) - len(stripped)
        leading_ws = full_text[:cleaned_prefix_len]
        m2 = NOTA_RE.match(stripped)
        if m2:
            stripped = stripped[m2.end():]
            # Si la envoltura previa terminaba con '}' final que cerraba el \textit{...},
            # debemos quitarlo del final del contenido. Buscamos la ultima '}' que cierre
            # el \textit{\textbf{Nota...:} ... }. Heuristica: quitar la ultima '}' no balanceada.
            # Cuenta de llaves para ubicar el cierre del \textit{...}.
            # Aproximacion: quitamos exactamente un '}' al final si el contenido termina con '}'
            # tras posibles espacios.
            tmp = stripped.rstrip()
            if tmp.endswith("}"):
                # Quita ese '}' y mantiene el resto.
                # Reconstruye stripped sin la '}' final, preservando trailing whitespace.
                idx_close = stripped.rfind("}")
                stripped = stripped[:idx_close] + stripped[idx_close + 1 :]

        content = leading_ws + stripped

        # Si el contenido ya esta envuelto con prefijo correcto, salta.
        # (Lo limpiamos arriba; reenvolvemos siempre.)
        new_item_text = (
            f"{indent}\\item \\textit{{\\textbf{{{prefix_label}}} " + content + "}"
        )

        # Reescribe las lineas del item.
        new_item_lines = new_item_text.split("\n")
        # Reemplaza new_block[a:b] con new_item_lines.
        new_block[a:b] = new_item_lines
        changed_local = True

        # Re-localiza spans posteriores porque la longitud cambio.
        # Recalcula item_idxs y spans tras la modificacion.
        item_idxs = [k for k, l in enumerate(new_block) if l.lstrip().startswith(r"\item ")]
        spans = []
        for jj, idx2 in enumerate(item_idxs):
            end_rel = item_idxs[jj + 1] if jj + 1 < len(item_idxs) else len(new_block)
            spans.append((idx2, end_rel))
        # Continua el bucle desde el item j+1 con los nuevos spans:
        # como iteramos sobre `spans` original, abortamos y reiniciamos en este bloque
        # con un loop manual:
        break  # salimos del for j; haremos un re-loop manual abajo.

    if changed_local:
        # Re-loop manual hasta procesar todos los items.
        while True:
            item_idxs = [k for k, l in enumerate(new_block) if l.lstrip().startswith(r"\item ")]
            n_items = len(item_idxs)
            # Determina el primer item NO envuelto correctamente.
            target = None
            for j, idx in enumerate(item_idxs):
                first = new_block[idx]
                expected = "Nota:" if n_items == 1 else f"Nota {j+1}:"
                if (
                    f"\\textit{{\\textbf{{{expected}}}" in first
                ):
                    continue
                target = j
                break
            if target is None:
                break
            # Procesa target.
            a = item_idxs[target]
            b = item_idxs[target + 1] if target + 1 < n_items else len(new_block)
            prefix_label = "Nota:" if n_items == 1 else f"Nota {target+1}:"
            first = new_block[a]
            m = re.match(r"^(\s*)\\item\s+(.*)$", first, flags=re.DOTALL)
            if not m:
                break
            indent, first_rest = m.group(1), m.group(2)
            rest_lines = new_block[a + 1 : b]
            full_text = first_rest
            for rl in rest_lines:
                full_text += "\n" + rl
            stripped = full_text.lstrip()
            leading_ws = full_text[: len(full_text) - len(stripped)]
            m2 = NOTA_RE.match(stripped)
            if m2:
                stripped = stripped[m2.end():]
                tmp = stripped.rstrip()
                if tmp.endswith("}"):
                    idx_close = stripped.rfind("}")
                    stripped = stripped[:idx_close] + stripped[idx_close + 1 :]
            content = leading_ws + stripped
            new_item_text = (
                f"{indent}\\item \\textit{{\\textbf{{{prefix_label}}} " + content + "}"
            )
            new_item_lines = new_item_text.split("\n")
            new_block[a:b] = new_item_lines

        lines[s + 1 : e] = new_block
        modified_blocks += 1

P.write_text("\n".join(lines), encoding="utf-8", newline="\n")
print("blocks:", len(starts), "modified:", modified_blocks)
