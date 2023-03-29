import torch 
import torch_geometric as tg
from pytorch3d.ops import iterative_closest_point as icp
from utils.weight_adjaceny import weight_adjacency_matrix
from utils.graph_spectrum import eigen_values_spectrum
from utils.flip_eigen import flip_eigen_sign

class Embedding:

    def __init__(self,obj):
        
        self.coords, self.faces = obj.coords, obj.faces
        self.depth = obj.depth
        self.thickness = obj.thickness
        self.curv = obj.curv
        self.device = obj.device
        if hasattr(obj, 'P'):
            self.P = obj.P
        else:
            self.P=[]

    def spectral(self, ne):
        
        """
        Computes the spectral embedding of the graph

        self.weights: weight adjacency matrix
        self.D = diagonal degree matrix
        self.Dinv = D^-1
        self.Lambda = eigen values
        self.vectors = eigen vectors

        returns: 
            self.edge_index : Adj matrix edge index
            self.edge_attr  : Adj matrix edge weights
            self.X          : Spectral embedding (K* ne)
            self.eig_vals   : Eigen values  (real + sorted)
            self.eig_vecs   : Eigen vectors  (real + sorted)
            self.device     : CPU / GPU
        
        """
        # compute the weighted adjacency matrix from the triangulated mesh(weight affinities)            
        self.edge_index, self.edge_attr = weight_adjacency_matrix(self.coords, self.faces) 

        # compute the randomwalk graphlaplacian 
        laplace_edge_index, laplace_edge_weight = tg.utils.get_laplacian(self.edge_index, self.edge_attr, normalization='rw')
        laplace = tg.utils.to_scipy_sparse_matrix(laplace_edge_index, laplace_edge_weight)
        self.eig_vals, self.eig_vecs = eigen_values_spectrum(laplace, ne)
        self.eig_vals, self.eig_vecs = self.eig_vals.to(device=self.device), self.eig_vecs.to(device=self.device)
        self.X = torch.matmul(self.eig_vecs, torch.diag(self.eig_vals ** (-0.5))) 
    
    def align(self, ref, krot, matching_samples, w_sulcal, two_step, matching_mode, verbose):
        """
        Performs spectral alignment of brain surfaces 

        ref              : reference embedding
            ref.depth    : sulcal depth (surface data)
        to_align         : embedding to be aligned to ref
        krot             : number of eigen vectors used for transformation
        matching_samples : number of points used to find transformation
        matching_mode    : partial or complete
        two_step         : start with 3 eigen vectors(less ambiguous)
        w_sulcal         : use/not use sulcal weights 

        Returns the aligned spectral embedding
        Mw               : aligned embedding
        """
        max_iterations = 100# * krot
        Mo = ref
        Mw = self        
        Mo.n = ref.coords.shape[0]
        Mw.n = Mw.coords.shape[0]
        

        Mw = flip_eigen_sign(Mo, Mw, krot, verbose)

        if matching_mode=='complete': #complete mesh
            if w_sulcal:
                E1 = torch.hstack((w_sulcal*Mo.depth.unsqueeze(1), Mo.X[:,0:krot])).unsqueeze(0)
                E2 = torch.hstack((w_sulcal*Mw.depth.unsqueeze(1), Mw.X[:,0:krot])).unsqueeze(0)
                if two_step: #start with 3 (less ambiguous)
                    init_trans = icp(E2[:, :, 1:4].to(torch.float32), E1[:, :, 1:4].to(torch.float32))
                    E2[:, :, 1:4] = init_trans.Xt   

                best_trans = icp(E2.to(torch.float32), E1.to(torch.float32), max_iterations=max_iterations, verbose=verbose)
                
                Mw.X[:,0:krot] = best_trans.Xt.squeeze()[:, 1:]     

            else:
                E1 = Mo.X[:,0:krot].unsqueeze(0)
                E2 = Mw.X[:,0:krot].unsqueeze(0)

                if two_step: #start with 3 (less ambiguous)
                    init_trans = icp(E2[:, :, 1:4].to(torch.float32), E1[:, :, 1:4].to(torch.float32))
                    E2[:, :, 1:4] = init_trans.Xt   

                best_trans = icp(E2.to(torch.float32), E1.to(torch.float32), max_iterations=max_iterations, verbose=verbose)
                
                Mw.X = best_trans.Xt.squeeze()
           
        
        elif matching_mode=='partial': #partial mesh
            
            n = torch.Tensor((matching_samples,Mo.n,Mw.n)).int().min()
            idx1 = torch.randperm(Mo.n)
            idx1 = idx1[0:n]
            idx2 = torch.randperm(Mw.n)
            idx2 = idx2[0:n]

            if w_sulcal:
                E1 = torch.hstack((w_sulcal*Mo.depth[idx1].unsqueeze(1), Mo.X[idx1,0:krot])).unsqueeze(0)
                E2 = torch.hstack((w_sulcal*Mw.depth[idx2].unsqueeze(1), Mw.X[idx2,0:krot])).unsqueeze(0)

                if two_step: # start with 3 (less ambiguous)
                    init_trans = icp(E2[:, :, 1:4].to(torch.float32), E1[:, :, 1:4].to(torch.float32))
                    E2[:, :, 1:4] = init_trans.Xt  

                best_trans = icp(E2.to(torch.float32), E1.to(torch.float32), verbose=verbose)

                E2 = torch.hstack((w_sulcal*Mw.depth.unsqueeze(1), Mw.X[:,0:krot])).unsqueeze(0)
                X2 =  best_trans.RTs.s[:, None, None] * torch.bmm(E2, best_trans.RTs.R.to(torch.float64)) + best_trans.RTs.T[:, None, :]
                Mw.X[:,0:krot] = X2.squeeze()[:, 1:]
                
            else:            
                E1 = Mo.X[idx1,0:krot]
                E2 = Mw.X[idx2,0:krot]

                if two_step: # start with 3 (less ambiguous)
                    init_trans = icp(E2[:, :, 1:4], E1[:, :, 1:4])
                    E2[:, :, 1:4] = init_trans.Xt  

                best_trans = icp(E2.to(torch.float32), E1.to(torch.float32), verbose=verbose)

                E2 = torch.hstack((w_sulcal*Mw.depth.unsqueeze(1), Mw.X[:,0:krot])).unsqueeze(0)
                X2 =  best_trans.RTs.s[:, None, None] * torch.bmm(E2, best_trans.RTs.R.to(torch.float64)) + best_trans.RTs.T[:, None, :]
                Mw.X = X2.squeeze()            
        
        self = Mw
        del(Mw); del(Mo)



