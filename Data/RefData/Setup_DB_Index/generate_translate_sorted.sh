#!/bin/bash
#Usage: ./generate_translate_sorted.sh path-to-db_info.txt
#generate a translation file (Accession Number to TaxId) for use with MetaTrinity-cs from underlying db_info.txt
set -e
if [ -z "$1" ]; then
    echo "Supply full path to db_info.txt"
    exit 1
fi
echo "making new translate-file"
awk 'NR>3{print $1,$3}' $(realpath $1) > translate.csv
echo "sorting new translate-file"
LC_ALL=C sort -n -k 1b,1 translate.csv > translate_sorted.csv
rm translate.csv
echo "done translate-file"
exit 0