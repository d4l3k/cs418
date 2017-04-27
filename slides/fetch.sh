rm *.pdf*
wget -i urls.txt
# pdfjam --checkfiles $(ls -v | grep slides.pdf) -o all-slides.pdf
gs -dBATCH -dNOPAUSE -q -sDEVICE=pdfwrite -dPDFSETTINGS=/prepress -sOutputFile=all-slides.pdf $(ls -v | grep slides.pdf)
pdfnup --nup 4x7 --paper letter --no-landscape all-slides.pdf -o slides-grid.pdf
