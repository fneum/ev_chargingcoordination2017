set table "edengths.pgf-plot.table"; set format "%.5f"
set format "%.7e";; set samples 100; set dummy x; plot [x=0:1440] 1-0.5*(1+(erf((x-1285)/(130*sqrt(2)))));
