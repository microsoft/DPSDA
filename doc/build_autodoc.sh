sphinx-apidoc -e -f --module-first -d 7 -o source/api ../pe ../pe/*/test* ../pe/*/*/test* ../pe/*/bk*
make clean html