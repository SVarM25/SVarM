import torch
import numpy as np
import os
from numpy import genfromtxt
import open3d as o3d

def loadData(file_name):  
    mesh = o3d.io.read_triangle_mesh(file_name)
    mesh = getDataFromMesh(mesh)
    return mesh

def getDataFromMesh(mesh):    
    V = torch.from_numpy(np.asarray(mesh.vertices, dtype=np.float32))
    F = torch.from_numpy(np.asarray(mesh.triangles, np.int64))
    return [V, F]