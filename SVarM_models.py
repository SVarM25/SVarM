import torch
import torch.nn as nn

#Varifold Computations
def mesh_varifold(mesh):
    V = mesh[0]
    F = mesh[1]
    V0, V1, V2 = V.index_select(0, F[:, 0]), V.index_select(0, F[:, 1]), V.index_select(0, F[:, 2])
    C, N =  (V0 + V1 + V2)/3, .5 * torch.cross(V1 - V0, V2 - V0)
    L = (N ** 2).sum(dim=1)[:, None].clamp_(min=1e-6).sqrt()    
    return C,N/L,L

def graph_varifold(mesh):
    V = mesh[0]
    E = mesh[1]
    V0,V1=V.index_select(0,E[:,0]),V.index_select(0,E[:,1])
    N=V1-V0
    M=(V1+V0)/2
    A=torch.sqrt((N**2).sum(dim=1).clamp_(min=1e-6))
    return M,N/A[:,None],A    
    
#Classifier
class h_classifier(nn.Module):
    def __init__(self,c):
        super(h_classifier, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(6, 16),
            nn.Sigmoid(),
            nn.Linear(16, 64),
            nn.Sigmoid(),
            nn.Linear(64, c)
        )
        
    def forward(self, x):
        return self.model(x)

class Classifier(nn.Module):
    def __init__(self,c, datatype = "mesh"):
        super(Classifier, self).__init__()
        self.Omega = h_classifier(c)
        if datatype == "mesh":
            self.varifold = mesh_varifold
        elif datatype == "graph":
            self.varifold = graph_varifold

    def forward(self, mesh_ls):
        ls_o = [
            (self.Omega(torch.cat([C, N], dim=1)) * L[:,None]).sum(dim=0)
            for mesh in mesh_ls
            for C, N, L in [self.varifold(mesh)]  # Compute once per mesh
        ]
        return torch.stack(ls_o)

#Regression
class h_regression(nn.Module):
    def __init__(self):
        super(h_regression, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(6, 16),
            nn.Sigmoid(),
            nn.Linear(16, 64),
            nn.Sigmoid(),
            nn.Linear(64, 128),
            nn.Sigmoid(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.model(x)

class Regression(nn.Module):
    def __init__(self,datatype = "mesh"):
        super(Regression, self).__init__()
        self.Omega = h_regression()
        self.lin = nn.Linear(1, 1)
        if datatype == "mesh":
            self.varifold = mesh_varifold
        elif datatype == "graph":
            self.varifold = graph_varifold

    def forward(self, mesh_ls):
        ls_o = [
            (self.Omega(torch.cat([C, N], dim=1)) * L).sum(dim=0)
            for mesh in mesh_ls
            for C, N, L in [self.varifold(mesh)]  # Compute once per mesh
        ]
        return self.lin(torch.stack(ls_o))
    