# Content Moderation Pipeline — Academic Report

This directory contains the LaTeX source for the project's academic report.

## Prerequisites

- **TeX Distribution**: [TeX Live](https://www.tug.org/texlive/) (recommended) or [MiKTeX](https://miktex.org/)
- **Required packages**: All packages are listed in `main.tex` preamble. Install them via your TeX distribution's package manager if missing.

### Required LaTeX Packages

| Package | Purpose |
|---------|---------|
| `tikz` | Architecture and flow diagrams |
| `pgfplots` | Charts and plots |
| `booktabs` | Professional tables |
| `hyperref` | Cross-references and links |
| `natbib` | Bibliography management |
| `listings` | Code snippets |
| `geometry` | Page layout |
| `fancyhdr` | Headers and footers |
| `graphicx` | Figure inclusion |
| `caption`, `subcaption` | Figure/table captions |
| `enumitem` | Customized lists |
| `amsmath`, `amssymb` | Mathematical notation |
| `float` | Float placement control |
| `tabularx` | Flexible-width tables |
| `multicol` | Multi-column layouts |

## Compilation

### Using latexmk (recommended)

```bash
cd report
latexmk -pdf main.tex
```

This will automatically run `pdflatex`, `bibtex`, and `pdflatex` again as needed to resolve all cross-references and bibliography.

### Manual compilation

```bash
cd report
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

Three passes of `pdflatex` are needed:
1. First pass: generates `.aux` files with citation and reference data
2. `bibtex`: processes bibliography from `.aux` and `.bib` files
3. Second pass: incorporates bibliography and resolves cross-references
4. Third pass: ensures all page numbers and references are correct

### Output

The compiled PDF will be generated at `report/main.pdf`.

## File Structure

```
report/
├── main.tex              # Main document (preamble + includes)
├── references.bib        # Bibliography database
├── README.md             # This file
├── chapters/
│   ├── titlepage.tex     # Title page with placeholders
│   ├── abstract.tex      # Abstract
│   ├── introduction.tex  # Chapter 1: Introduction
│   ├── state_of_art.tex  # Chapter 2: State of the Art
│   ├── requirements.tex  # Chapter 3: Requirements Analysis
│   ├── architecture.tex  # Chapter 4: System Architecture
│   ├── text_moderation.tex  # Chapter 5: Text Moderation
│   ├── image_moderation.tex # Chapter 6: Image Moderation
│   ├── implementation.tex   # Chapter 7: Implementation
│   ├── experiments.tex      # Chapter 8: Experiments
│   ├── results.tex          # Chapter 9: Results
│   ├── limitations.tex      # Chapter 10: Limitations & Future Work
│   └── conclusion.tex       # Chapter 11: Conclusion
├── figures/              # (place diagrams and screenshots here)
└── tables/               # (place standalone table files here)
```

## Customization

### Adding Figures

Place PNG, JPG, or PDF files in the `figures/` directory. Reference them in the chapter files:

```latex
\begin{figure}[H]
\centering
\includegraphics[width=0.8\textwidth]{figures/my-figure.png}
\caption{Figure description.}
\label{fig:my-figure}
\end{figure}
```

### Adding More References

Add BibTeX entries to `references.bib` and cite them with `\cite{key}`.

### Compiling to HTML

To generate HTML output (using TeX4ht or latex2html):

```bash
latex4ht main.tex
# or
latex2html main.tex
```

## Troubleshooting

- **Missing packages**: Use your TeX distribution's package manager (`tlmgr` for TeX Live, MiKTeX Package Manager for MiKTeX)
- **Font issues**: Ensure `lmodern` package is installed
- **TikZ errors**: Make sure `tikz` and `pgfplots` are up to date
- **Bibliography issues**: Ensure `natbib` is compatible with `plainnat` style
