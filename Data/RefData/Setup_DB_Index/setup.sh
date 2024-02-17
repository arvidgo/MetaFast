#!/bin/bash
#Usage: ./setup.sh [-s fna_target_size(GB)=0.8] [-k kmer_length=28] [-t max_threads=16] [-f infile] [-d (flag, debug concatenation)]
#Downloads data and generates
# 1. a translate_sorted.csv file (for usage with containment search)
# 2. merged FNA files [listed in infile, or all] of size ~800MB (0.8GB) each
# 3. the index (MMI-files) with a default of kmer-length 28 (while using up to 16 threads)

usage() {
    echo "Usage: ./setup.sh [-s fna_target_size(GB)=0.8] [-k kmer_length=28] [-t max_threads=16] [-f infile] [-d (flag, debug concatenation)]"
    exit 1
}

set -e
fnasize=0.8
kmerlen=28
usableThreadCount=16
debugConcat=0;
while getopts ':s:k:t:f:d' OPTION; do case $OPTION in 
    s) fnasize=$OPTARG;;
    k) kmerlen=$OPTARG;;
    t) usableThreadCount=$OPTARG;;
    f) infile=$(realpath "$OPTARG");;
    d) debugConcat=1;;
    ?) usage;;
esac; done;
shift "$(($OPTIND -1))"
if ! [ -z ${1+x} ]; then
    usage
fi
SCRIPT_DIR=$(dirname -- "$(readlink -f -- "${BASH_SOURCE[0]}")")
cd $SCRIPT_DIR
mkdir -p ../DB
mkdir -p ../Index

echo MetaTrinity - SETUP - downloading, this will likely take VERY long
#/bin/bash download_DB.sh ../DB/

echo MetaTrinity - SETUP - generating translate_sorted.csv
/bin/bash generate_translate_sorted.sh ../DB/db_info.txt
mv translate_sorted.csv ../DB

echo MetaTrinity - SETUP - generating Index
cd ../Index
if [ -z ${infile+x} ]; then
    echo no infile given, converting entire db to mmis
    ls -1U ../DB/organism_files > mergeallfnas.txt
fi
/bin/bash $SCRIPT_DIR/generate_fnas_and_mmis.sh "${infile:-mergeallfnas.txt}" ../DB/organism_files $fnasize $kmerlen $usableThreadCount $debugConcat
if [ -z ${infile+x} ]; then
    rm mergeallfnas.txt
fi

echo MetaTrinity - SETUP - ALL DONE
exit 0

