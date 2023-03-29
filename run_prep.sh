source PATH_TO_CONDA/conda.sh

cd PATH_TO_PROJ_DIR
conda activate ENV

subjs=$(< mindboggle101_list.txt)

for f in $subjs; 
do 
    echo $f
    python spectral_align.py -r data/mindboggle/HLN-12-4 -s data/mindboggle/$f -o data/after_alignment --robust --gpu --verbose
done
