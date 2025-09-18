from typing import List, Literal
from time import perf_counter
from opt_einsum import contract
import random

from ..constant import Nc, Ns, Nd
from ..backend import set_backend, get_backend, check_QUDA
from ..preset import GaugeField, Eigenvector, Perambulator



class SmearedPropagatorGenerator: 
    #We implement the calculation of hte smeared propagators onto a sublattice of the original
    #Method described in https://arxiv.org/abs/2412.09246

    def __init__(
            self,
            latt_size: List[int],
            eigenvector: Eigenvector,
            perambulator: Perambulator,
            Npoints: int,
            seed: int,          
            contract_prec: str = "<c16",
        )->None:
        
        backend = get_backend()
        Lx,Ly,Lz,Lt= latt_size
        self.lat_vol = Lx*Ly*Lz
        self.latt_size = latt_size
        self.eigenvector = eigenvector
        self.perambulator = perambulator
        self.seed = seed
        self.Ne = eigenvector.Ne
        self.contract_prec = contract_prec
        self._eigenvector_data = None
        self._perambulator_data = None
        self._sublattice = None
        self.Npoints = Npoints
        self._eigen_sublattice = None
        self._smeared_propagator = backend.zeros((Lt,self.lat_vol,self.lat_vol,Nd,Nd,Nc,Nc),self.contract_prec)
        

        random.seed(self.seed)


    def load(self,key : str):
        self._eigenvector_data = self.eigenvector.load(key)
        self._perambulator_data = self.perambulator.load(key)
        self._eigen_sublattice= self.get_random_sample(self.Npoints)

        

    def get_random_sample(self, Npoints):
        Lx,Ly,Lz,Lt = self.latt_size
        _eigenvector_data = self._eigenvector_data[:]
        _eigenvector_data =_eigenvector_data.reshape(Lt,self.Ne,self.lat_vol,Nc) #Ordered like x + y*lx +z*ly*lx

        points = random.sample(range(self.lat_vol),Npoints)
        sorted_points = sorted(points)
        #print(sorted_points)

        _sublattice = _eigenvector_data[:,:,sorted_points,:]

        return _sublattice

    def calc(self,t_source):
        #This should happen at initialization if we want all sources to have the same random selection. 
        #Also, add selection to use different random samplings at source and sink.
        #print(f"In timeslice {t_source} we get: {Npoints} points")
        _eigen_sublattice= self._eigen_sublattice
        _perambulator = self._perambulator_data[t_source]
        _smeared_propagator = self._smeared_propagator

        #print shapes of perambulator and eigen_sublattice
        #print(f"Perambulator shape: {_perambulator.shape}, eigen_sublattice shape: {self._eigen_sublattice.shape}")
        #Now we do the contraction
        #Indices should be: t source, t sink, x source, y sink, A,B Dirac, a,b color 
        _smeared_propagator = contract('tixa,tABij,jzb->txzABab',self._eigen_sublattice,_perambulator,self._eigen_sublattice[t_source,...].conj())
        return _smeared_propagator
