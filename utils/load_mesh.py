import os
import torch
import nibabel.freesurfer.io as fsio
from utils.utils import *

class LoadMesh:
        
    def load_mesh(self, main_path, id, hemi, device):
        """
        Loads the mesh using free surfer

        path    : path of the dataset
        id      : patient id
        hemi    : left/right hemisphere
        
        returns coordinates, faces, sulcal depth, cortical thickness, parcellation

        """
        self.device = device
    
        #mesh
        path = os.path.join(main_path,id,'surf',hemi+".white")
        
        try:
            self.coords,self.faces = fsio.read_geometry(path)
            self.coords = torch.from_numpy(self.coords).to(device=device)
            self.faces = torch.from_numpy(self.faces.astype('float')).to(device=device)
            
        except FileNotFoundError:
            print('Mesh File not found')

        #Sulcal depth
        path = path.replace('white','sulc')
        try:
            self.depth = fsio.read_morph_data(path)
            self.depth = torch.from_numpy(self.depth.astype('float')).to(device=device)
            
            
        except FileNotFoundError:
            print('Depth File not found')
        
        #mesh cortical thickness
        path = path.replace('sulc','thickness')

        try:
            self.thickness = fsio.read_morph_data(path)
            self.thickness = torch.from_numpy(self.thickness.astype('float')).to(device=device)
            
        except FileNotFoundError:
            print('Thickness File not found')
        
        path = path.replace('thickness','curv')
        try:
            self.curv = fsio.read_morph_data(path)
            self.curv = torch.from_numpy(self.curv.astype('float')).to(device=device)
            
        except FileNotFoundError:
            print('Thickness File not found')
        
        try:
            path = os.path.join(main_path, id, 'label', hemi + ".labels.DKT31.manual.2.annot")

            # build lookup table to compact labels into consecutive indices
            parcs = fsio.read_annot(path)
            all_labels = flatten_lists([list(np.where(parcs[0]<0, 0, parcs[0]))])
            al = [l if l >= 0 else 0 for l in all_labels]
            lab_to_ind, ind_to_lab = rebase_labels(al)
            parc = lab_to_ind[np.where(parcs[0]<0, 0, parcs[0])]
            self.P  = torch.from_numpy(parc.astype('float32')).to(device=device)

        except FileNotFoundError:
            print(id + ' - Parcellation file not found')




  
