# MiKTeX's bibtex resolves .bib files relative to the output directory when
# latexmk passes an absolute aux path (e.g. -outdir=build), so references.bib
# in this directory is missed and a system-installed references.bib matches
# instead. Prepend this directory (and the output dir's parent) to BIBINPUTS.
$ENV{'BIBINPUTS'} = '.;..;' . $ENV{'BIBINPUTS'};
