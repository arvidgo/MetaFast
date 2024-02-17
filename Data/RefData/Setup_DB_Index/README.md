# MetaTrinity: Enabling Fast Metagenomic Classification via Seed Counting and Edit Distance Approximation

We introduce MetaTrinity, a tool based on heuristics, to achieve a fundamental improvement in accuracy-runtime tradeoff over existing methods. MetaTrinity delivers accuracy comparable to the alignment-based and highly accurate tool Metalign but with significantly enhanced efficiency. In MetaTrinity, we accelerate memory-frugal reference database indexing and filtering. We further employ heuristics to accelerate read mapping. Our evaluation demonstrates that MetaTrinity achieves a 3.9x speedup over Metalign without compromising accuracy.  
MetaTrinity has been rigorously debugged and validated using GCC 12.1.0-16.

Described by Arvid E. Gollwitzer et al. (current version at https://doi.org/10.48550/arXiv.2311.02029).

## Table of Contents
- [Setup the Database and Index Structure ](#install)
- [Getting Help](#contact)
- [Citing MetaTrinity](#cite)


## <a name="install"></a>Installation
We provide a one-click-setup script:
```bash
./setup.sh
```
The setup has three stages: 
1. Download of the dataset used by Metalign
2. Generation of a sorted translation file used in our containment search implementation, saved to [DB](../DB/)
3. Concatenation of the organism files in the DB to generate index files of a certain size, both saved to [Index](../Index/)

By default, the setup will concatenate **all** organism files tp a size of about 800MB, and generate index files with kmer-length 28.
During index generation, the script will by default use up to 16 threads. 
To change these values, call ```setup.sh``` with different parameters.  
Example: for concatenation size 2.5GB, with kmer-length 15, using up to 10 threads, and only limited to organism files listed in *infile.txt*, with debug output for concatenation
```bash
#Usage: ./setup.sh [-s fna_target_size(GB)=0.8] [-k kmer_length=28] [-t max_threads=16] [-i infile] [-d (flag, debug concatenation)]
./setup.sh -s 2.5 -k 15 -t 10 -i infile.txt -d
```
To execute individual stages of the setup independently, refer to the corresponding scripts for specific usage information

To (re)generate index files from already concatenated organism files (e.g. with different kmer-length), use ```generate_mmis.sh```.


##  <a name="contact"></a>Getting Help
If you have suggestions for improvement, new applications, or wish to collaborate, please contact arvid.gollwitzer@safari.ethz.ch.  
If you encounter bugs or have further questions or requests, you may raise an issue on the issue page.


## <a name="cite"></a>Citing MetaTrinity
If you use MetaTrinity in your work, please cite:

> Gollwitzer, Arvid E., et al. "MetaTrinity: Enabling Fast Metagenomic Classification via Seed Counting and Edit Distance Approximation." 
> arXiv:2311.02029 (2023).

You may use the following BibTeX:

```bibtex
@misc{gollwitzer2023MetaTrinity,
      title={MetaTrinity: Enabling Fast Metagenomic Classification via Seed Counting and Edit Distance Approximation}, 
      author={Arvid E. Gollwitzer and Mohammed Alser and Joel Bergtholdt and Joel Lindegger and Maximilian-David Rumpf and Can Firtina and Serghei Mangul and Onur Mutlu},
      year={2023},
      eprint={2311.02029},
      archivePrefix={arXiv},
      primaryClass={q-bio.GN}
}

