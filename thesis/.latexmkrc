# Toda la compilación vive fuera de la fuente, en una única carpeta desechable:
# el PDF final y los auxiliares (.aux, .log, .bbl, .bcf, ...) van a thesis/render/.
# Nunca se genera nada en la raíz de thesis/.
$pdf_mode = 1;        # usar pdflatex
$out_dir  = 'render'; # destino de todo el resultado de compilación

# Que `latexmk -c` deje render/ con solo el PDF: barre también el .bbl de biber
# y el residuo .bcf-SAVE-ERROR que biber deja a veces en builds multi-pasada.
$clean_ext = 'bbl bcf-SAVE-ERROR synctex.gz';
