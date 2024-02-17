#!/bin/bash
#Usage: ./generate_fnas_and_mmis.sh fna_list_file fna_dir fna_target_size [kmer_length=15] [max_threads=3] [concat_debugmode=0]
#concat all filenames listed in 'fna_list_file' from 'fna_dir' into fna chunks of size 'fna_target_size' and
#convert concatenated fnas to mmis with specified kmer_length
#NOTE:
# 1. For our readsets, the MMI-size was about 3.8x to 6.3x the concatenated FNA-size, depending on the dataset. Your mileage may vary significantly.
# 2. max_threads has to be at least 3, since mm2 uses 3 threads for indexing (https://lh3.github.io/minimap2/minimap2.html see -t option)

set -e #Exit immediately if a command exits with a non-zero status.
SCRIPT_DIR=$(dirname -- "$(readlink -f -- "${BASH_SOURCE[0]}")")
INDEXING_MM2_THREAD_COUNT=3 #https://lh3.github.io/minimap2/minimap2.html explanation see -t option
#cd $SCRIPT_DIR

fnalist=$(realpath $1)
fnas=$(realpath $2)
fnasize=$3
kmerlen=${4:-15} # set k to 15 (minimap default) if arg not present
usableThreadCount=${5:-$INDEXING_MM2_THREAD_COUNT} # set max_threads to standard if arg not present
concatOutput=${6:-1} #give debug output by default
currdate=$(date '+%Y-%m-%d_%H_%M_%S')
echo using fna size $fnasize
outFNA=concatenatedFnas_$currdate
mkdir $outFNA


echo "FNA+MMI: starting (this will likely take long)"

echo recompiling: g++ -o $SCRIPT_DIR/concatfnas.o $SCRIPT_DIR/concatfnas.cpp
g++ -o $SCRIPT_DIR/concatfnas.o $SCRIPT_DIR/concatfnas.cpp

echo concatenating FNAs from file: $SCRIPT_DIR/concatfnas.o $fnalist $fnas $(realpath $outFNA) $fnasize $concatOutput
"$SCRIPT_DIR/concatfnas.o" $fnalist $fnas $(realpath $outFNA) $fnasize $concatOutput

echo making to mmis
/bin/bash $SCRIPT_DIR/generate_mmis.sh $(realpath $outFNA) $kmerlen $usableThreadCount

echo FNA+MMI: DONE
exit 0