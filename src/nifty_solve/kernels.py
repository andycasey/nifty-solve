import numpy as np
from math import gamma as Γ

class BaseKernel:
    pass

class SquaredExponential(BaseKernel):

    def __init__(self, length_scale):
        self.length_scale = length_scale
        self.dimension = 1 # TODO

    def spectral_density(self, frequency):
        l = self.length_scale / (2 * np.pi)
        return (2 * np.pi * l**2)**(self.dimension/2) * np.exp(-2 * np.pi**2 * l**2 * frequency**2)

    def _modes_required(self, epsilon):
        L = 2 * np.pi # TODO
        h_spacing = 1/(L + self.length_scale * np.sqrt(2 * np.log(4 * self.dimension * 3**self.dimension/epsilon)))
        hm = int(np.ceil(np.sqrt(np.log(self.dimension*(4**(self.dimension+1))/epsilon)/2)/np.pi/self.length_scale/h_spacing))
        return 2 * hm + 1


class MaternKernel(BaseKernel):
    def spectral_density(self, frequency):
        from math import gamma
        den = 0.5 * np.sqrt(np.pi)
        D32 = (self.dimension + 3)/2
        num = (
            np.power(2, self.dimension)
        *   np.power(np.pi, self.dimension / 2)
        *   gamma(D32)
        *   np.power(3, 3/2)
        )
        p = np.power(3.0 + frequency**2, -D32)
        return (num / den) * p
        
    def _modes_required(self, epsilon):
        L = 2 * np.pi # TODO
        h_spacing = 1 / (L + 0.85*self.length_scale/np.sqrt(self.ν)*np.log(1/epsilon))  
        hm = int(
            np.ceil(
                (np.pi**(self.ν+self.dimension/2) * self.length_scale**(2*self.ν) * epsilon/0.15)**(-1/(2*self.ν+self.dimension/2)) 
            /   h_spacing
            )
        )
        return 2 * hm + 1
        

class Matern32(MaternKernel):

    ν = 3/2
    
    def __init__(self, length_scale):
        self.length_scale = length_scale
        self.dimension = 1 # TODO