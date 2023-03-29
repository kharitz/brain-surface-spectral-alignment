source /autofs/space/ballarat_001/users/kg149/anaconda3/etc/profile.d/conda.sh

cd /autofs/space/ballarat_001/users/kg149/proj_gcn_parcellation
conda activate proj_gcn_parcellation


subjs=$(< mindboggle101_list.txt)

for f in $subjs; 
do 
    echo $f
    python spectral_align.py -r data/mindboggle/HLN-12-4 -s data/mindboggle/$f -o data/after_alignment --robust --gpu --verbose
done