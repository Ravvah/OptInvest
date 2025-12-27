# Build the report PDF using pdflatex (two passes)

TEX=rapport_optinvest.tex
PDF=$(TEX:.tex=.pdf)

all: $(PDF)

$(PDF): $(TEX)
	pdflatex -interaction=nonstopmode -halt-on-error $(TEX)
	pdflatex -interaction=nonstopmode -halt-on-error $(TEX)

clean:
	rm -f *.aux *.log *.out *.toc *.lof *.lot *.bbl *.blg *.fls *.fdb_latexmk *.synctex.gz $(PDF)

.PHONY: all clean
