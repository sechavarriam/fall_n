# Estrategia de documentación de `fall_n` — propuesta para decisión

> Documento de decisión. Presenta opciones de sistema de documentación de
> **uso** del software (distinta de la bitácora LaTeX y del CHANGELOG), con
> ventajas y contras, para que el autor escoja. Ninguna se ha implementado.

## 1. Qué queremos y contra qué se decide

La documentación de uso debe:

- **Separar tres registros** que hoy se confunden:
  1. *Bitácora de desarrollo* — el documento LaTeX actual (`doc/`), que pasa
     a ser el registro histórico de teoría/formulaciones/campañas (tesis),
     no el manual del código.
  2. *Changelog* — `CHANGELOG.md`, hitos por commit.
  3. *Documentación de usuario/desarrollador* — **lo nuevo**: cómo usar y
     extender la librería.
- Combinar **narrativa** (tutoriales, guías conceptuales) con **referencia
  de API** extraída del código C++23 (concepts, plantillas, type-erasure).
- Renderizar **ecuaciones** (semántica física/numérica: trabajo virtual,
  pares conjugados, retorno plástico, arc-length…).
- Incluir **diagramas de clases y de flujo** (idealmente versionables como
  texto, no imágenes binarias).
- Tener **bibliografía** con `.bib` (Ko-Bathe, Menegotto-Pinto, Kent-Park,
  Crisfield, Battini-Pacoste, Küttler-Wall, Reynolds…).
- Ser **legible, moderna y extensible** por futuros desarrolladores, y
  **hospedable** (GitHub Pages / Read the Docs) y reproducible en CI.

Criterio transversal: el proyecto ya vive de metadatos en el tipo (concepts,
catálogos `constexpr`). La doc debe poder **citar el código como fuente de
verdad** y no divergir de él.

## 2. Opciones

### Opción A — Sphinx + Breathe + Doxygen (+ MyST-Markdown)

Doxygen extrae la API a XML; Breathe la inyecta en Sphinx; la narrativa se
escribe en MyST (Markdown) o reStructuredText.

- **Pros**
  - El estándar de facto en documentación científica; combina de forma
    natural **tutoriales + referencia de API** en un solo sitio navegable.
  - Ecuaciones de primera: `mathjax`/`katex`, sintaxis LaTeX directa.
  - **Bibliografía** madura con `sphinxcontrib-bibtex` (consume el mismo
    `.bib` de la tesis).
  - Diagramas como texto: `sphinxcontrib-mermaid` o PlantUML; Graphviz para
    grafos de clases automáticos vía Doxygen.
  - Temas modernos (Furo, PyData, Book) con búsqueda, modo oscuro, versiones.
  - Hospedaje trivial en **Read the Docs** (build reproducible, versionado
    por tag/branch).
  - Ecosistema enorme; cualquier desarrollador científico lo reconoce.
- **Contras**
  - Cadena de herramientas más pesada: Python + Doxygen + Breathe + tema.
  - Breathe puede atragantarse con **C++ moderno muy templatizado / concepts**
    (a veces hay que documentar a mano ciertos símbolos).
  - reStructuredText tiene curva; MyST lo suaviza pero añade otra pieza.
- **Encaje con fall_n**: alto. Es la opción con mejor balance narrativa+API+
  math+bib+diagramas+hosting. Riesgo concentrado en la fidelidad de Breathe
  con los concepts.

### Opción B — MkDocs + Material for MkDocs (+ Doxygen vía `mkdoxy`)

Documentación Markdown-nativa con el tema Material (muy pulido); la API se
puentea desde Doxygen con el plugin `mkdoxy`.

- **Pros**
  - **La mejor UX lista para usar**: navegación, búsqueda instantánea, modo
    oscuro, tabs, admoniciones, todo elegante sin tocar CSS.
  - Markdown puro: **la barrera de entrada más baja** para que futuros
    desarrolladores contribuyan.
  - Math con KaTeX/MathJax; **Mermaid integrado** para diagramas de clases/
    secuencia como texto.
  - Bibliografía con `mkdocs-bibtex` (consume `.bib`).
  - Build rápido; hospedaje directo en GitHub Pages.
- **Contras**
  - La **extracción de API C++** es menos madura que Sphinx+Breathe;
    `mkdoxy` funciona pero es menos expresivo con plantillas/concepts.
  - Menos convención científica establecida (aunque crece rápido).
- **Encaje con fall_n**: alto si se prioriza **narrativa hermosa + Markdown
  fácil** y se acepta una referencia de API más ligera (o escrita a mano
  para las piezas insignia).

### Opción C — Doxygen + m.css (front-end moderno de mosra)

`m.css` reemplaza el front-end de Doxygen por uno pensado para **C++
moderno**, con salida bellísima, búsqueda rápida y math LaTeX real.

- **Pros**
  - **Nativo C++**, el que mejor entiende plantillas/sobrecargas; salida
    limpia y rápida.
  - Math LaTeX renderizado por m.css; búsqueda excelente; un solo paso de
    build (Doxygen → m.css), **cadena ligera**.
  - Cero dependencia de Python/Sphinx.
- **Contras**
  - Orientado a **referencia de API**, no a tutoriales largos; la narrativa
    se hace con “pages” de Doxygen (menos cómodo que Markdown/MyST).
  - Bibliografía vía `\cite` de Doxygen (funciona, menos flexible que
    sphinxcontrib-bibtex).
  - Comunidad más pequeña; menos temas/plugins.
- **Encaje con fall_n**: bueno si la prioridad #1 es una **referencia de API
  C++ impecable** con poco tooling, y la narrativa es secundaria.

### Opción D — Doxygen “clásico” + doxygen-awesome-css

Doxygen directo con un tema moderno (doxygen-awesome-css).

- **Pros**: cero pegamento, un solo binario; MathJax; grafos Graphviz;
  `\cite`+`.bib`; se ve decente con el tema; **mínimo esfuerzo**.
- **Contras**: la UX y la narrativa son las más pobres del lote; se siente
  “documentación de los 2010s”; poco atractivo para nuevos contribuidores.
- **Encaje**: es el **piso mínimo** — útil como paso 0 barato si se quiere
  algo hoy y migrar luego.

## 3. Comparativa rápida

| Criterio | A. Sphinx+Breathe | B. MkDocs Material | C. Doxygen+m.css | D. Doxygen clásico |
|---|---|---|---|---|
| Narrativa/tutoriales | Excelente | Excelente | Media | Baja |
| Referencia de API C++ | Alta (Breathe) | Media (mkdoxy) | **Muy alta** | Alta |
| C++ moderno/concepts | Media | Media | **Alta** | Media |
| Ecuaciones LaTeX | Excelente | Excelente | Alta | Buena |
| Diagramas como texto | Mermaid/PlantUML | **Mermaid nativo** | Graphviz | Graphviz |
| Bibliografía `.bib` | **Excelente** | Buena | Media (`\cite`) | Media (`\cite`) |
| UX / estética | Alta | **Muy alta** | Alta | Media |
| Barrera para contribuir | Media | **Baja (Markdown)** | Media | Media |
| Peso del tooling | Alto | Medio | **Bajo** | **Bajo** |
| Hosting | RTD/Pages | Pages | Pages | Pages |

## 4. Recomendación

Dos finalistas según la prioridad:

- **Si la prioridad es una referencia de API C++ fiel + narrativa rica en un
  solo sitio científico estándar → Opción A (Sphinx + Breathe + Doxygen).**
  Es la que mejor honra “el código como fuente de verdad” y la bibliografía
  compartida con la tesis. Recomendada como objetivo.

- **Si la prioridad es adopción y estética con mínima fricción para futuros
  desarrolladores → Opción B (MkDocs Material + mkdoxy).** Markdown baja la
  barrera de contribución; la referencia de API es algo más ligera pero
  suficiente si se escriben a mano las páginas de las piezas insignia
  (`MultiscaleAnalysis`, la capa de concepts, la familia de solvers).

Camino híbrido sugerido (mi recomendación operativa): **empezar por B** para
tener ya un sitio bello y contribuible con las guías conceptuales y los
diagramas Mermaid, y **evaluar complementar con A** cuando la referencia de
API automática sea el cuello de botella. En ambos casos:

- La bibliografía sale del mismo `.bib` de la tesis (una sola fuente).
- Los diagramas de clases se escriben en **Mermaid** (texto versionable),
  no como imágenes.
- El sitio se construye en CI y se publica en GitHub Pages por tag.

## 5. Esqueleto de contenidos propuesto (independiente de la herramienta)

```
Introducción            — qué es fall_n, alcance, estado
Instalación             — dependencias, build por plataforma
Guía de inicio          — primer modelo, primer análisis
Conceptos               — la arquitectura de concepts; formulación×ruta;
                          disciplina de estado commit/rollback
Elementos               — beam/shell/truss/continuum; políticas cinemáticas
Materiales              — Ko-Bathe, Menegotto-Pinto, Kent-Park; commit chains
Solvers                 — SNES, Newmark, arc-length, LM/Newton, TAO
Multiescala (FE²)       — downscaling, two-way, homogenización, XFEM local
Referencia de API       — extraída del código (Doxygen)
Bibliografía            — .bib compartido con la bitácora
Contribuir              — estilo, concepts sobre herencia, cómo extender
```

## 6. Decisión pendiente del autor

Escoger A, B, C, D o el híbrido B→A. Una vez elegido, el primer entregable
sería el andamiaje del sitio + las páginas “Introducción”, “Instalación” y
“Guía de inicio”, más un diagrama Mermaid de la arquitectura de módulos.
