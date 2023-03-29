# This file may be used to create an environment for running this project:

export LD_LIBRARY_PATH=/usr/pubsw/packages/11.3/lib64:$LD_LIBRARY_PATH
export PATH=/usr/pubsw/packages/CUDA/11.3/bin:$PATH
source /autofs/space/ballarat_001/users/kg149/anaconda3/etc/profile.d/conda.sh
# export LD_LIBRARY_PATH=/PATH_TO_CUDA/11.3/lib64:$LD_LIBRARY_PATH
# export PATH=/PATH_TO_CUDA/CUDA/11.3/bin:$PATH
conda create -n proj_alignment python=3.9
conda activate proj_alignment
conda install -c pytorch pytorch=1.11.0 torchvision cudatoolkit=11.3
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
conda install pytorch3d -c pytorch3d
conda install pyg -c pyg
pip install nibabel