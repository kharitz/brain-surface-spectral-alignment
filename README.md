# Spectral Embedding alignment
This repository impliments the python version of the graph spectral alignment. 

![Alt Text](https://github.com/kharitz/brain-surface-spectral-alignment/fig1.png)

If you find this work useful, please cite:
```
@article{gopinath2019graph,
  title={Graph convolutions on spectral embeddings for cortical surface parcellation},
  author={Gopinath, Karthik and Desrosiers, Christian and Lombaert, Herve},
  journal={Medical image analysis},
  volume={54},
  pages={297--305},
  year={2019},
  publisher={Elsevier}
}
```
## Installation

### What does the reopositery do?

- main.py
  1. FreeSurfer brain surfaces is read from the "dataset" folder.
  2. Spectral embedding of the brain graph is computed.
  3. Spectral basis of each subject is aligned to a common reference from the dataset.
  4. Computed embeddings, transformation and aligned spectral embeddings are saved in "output" folder in  ".pt" for DeepLearning algorithms later use in PyTorch.

### Where to find the dataset?

- The MindBoggle brain surfaces dataset is available to download [here](https://osf.io/nhtur/).
- The FreeSurfer processed ADNI surfaces dataset is available to download [here](http://adni.loni.ucla.edu).

### Package Requirements
- math
- matplotlib 
- mne
- nibabel
- numpy
- python3, pytorch>1.0 
- pickle
- pandas 
- scipy
- time, timeit
- vtk

### Installation
- To use the *requirements.txt* file to create an identical environment on the same machine or another machine:
```
conda create --name myenv --file requirements_linux.txt
```
- To use the *requirements.txt* file to install its listed packages into an existing environment:
```
conda install --name myenv --file requirements_linux.txt
```


### Usage
```
python3 main.py
```

#### REFERENCE 

Please cite our papers if you use this code in your own work:

```
@article{gopinath2019graph,
  title={Graph convolutions on spectral embeddings for cortical surface parcellation},
  author={Gopinath, Karthik and Desrosiers, Christian and Lombaert, Herve},
  journal={Medical image analysis},
  volume={54},
  pages={297--305},
  year={2019},
  publisher={Elsevier}
}
```
# test_1
