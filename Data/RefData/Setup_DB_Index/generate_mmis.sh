#!/bin/bash
#Usage: ./generate_mmis.sh concatenated_fna_gz_folder [kmer_length=15] [max_threads=1]
#convert concatenated fnas to mmis with specified kmer_length
#NOTE:
# 1. For our testcases, the MMI-size was about 3.8x to 6.3x the merged FNA-size, depending on the dataset. Your mileage may vary significantly.
# 2. max_threads has to be at least 3, since mm2 uses 3 threads for indexing (https://lh3.github.io/minimap2/minimap2.html see -t option)

set -e #Exit immediately if a command exits with a non-zero status.
SCRIPT_DIR=$(dirname -- "$(readlink -f -- "${BASH_SOURCE[0]}")")
INDEXING_MM2_THREAD_COUNT=3 #https://lh3.github.io/minimap2/minimap2.html -t option
#cd $SCRIPT_DIR

fnas=$(realpath $1)
kmerlen=${2:-15} # set k to 15 (minimap default) if arg not present
usableThreadCount=${3:-$INDEXING_MM2_THREAD_COUNT} # set to standard (=will be essentially sequential) if arg not present
if [ $usableThreadCount < $INDEXING_MM2_THREAD_COUNT ]; then
    echo "adjusting usable thread count to 3 (mm2 minimum for index generation)"
    usableThreadCount=$INDEXING_MM2_THREAD_COUNT
fi
outMMI=generatedMMIs_$(date '+%Y-%m-%d_%H_%M_%S')_k$kmerlen
mkdir $outMMI

#start actual script
echo starting MMI conversion
N=$(($usableThreadCount/$INDEXING_MM2_THREAD_COUNT)) #integer division intended
echo making $fnas to mmis with k=$kmerlen and using up to $usableThreadCount threads, so N=$N
count=1
for f in $(ls -1v $fnas -I "*.txt"); do
    echo minimap2 --idx-no-seq -H -k $kmerlen -d $outMMI/mmi$count.mmi $fnas/$f
    minimap2 --idx-no-seq -H -k $kmerlen -d $outMMI/mmi$count.mmi $fnas/$f & 
    ((count++))
    if [[ $(jobs -r -p | wc -l) -ge $N ]]; then
        wait -n
    fi
done
wait
echo DONE MMI conversion
exit 0