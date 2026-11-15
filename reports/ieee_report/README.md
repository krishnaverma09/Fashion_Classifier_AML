# IEEE project report

`Fashion_Classifier_IEEE_Report.pdf` is the project report in the IEEE conference format
(`IEEEtran`, two columns, 5 pages). `main.tex` is its source, and the figures it uses are in
`figures/`:

- `dataset_examples.png`: a copy of `reports/figures/training_examples.png`;
- `learning_curves.png`: a copy of `reports/v3/learning_curves.png`;
- `per_class_recall.png`: new, built from `reports/final_evaluation/per_class_recall.csv` and
  `reports/v3/test_results.csv`.

Every number in the paper comes from the committed result files in `reports/`.

## Compile

**Overleaf:** New Project → Upload Project → upload a zip of this folder, then Recompile.
`IEEEtran` is built into Overleaf.

**Locally** (needs a LaTeX installation, for example `brew install tectonic`):

```bash
tectonic main.tex        # or: pdflatex main.tex && pdflatex main.tex
mv main.pdf Fashion_Classifier_IEEE_Report.pdf
```

pdfLaTeX uses IEEEtran's Times font. Tectonic (XeTeX) uses TeX Gyre Termes, a Times clone,
which `main.tex` selects automatically.
