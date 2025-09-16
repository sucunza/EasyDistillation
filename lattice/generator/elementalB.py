from typing import List, Tuple
#This class is created by me. JS
from opt_einsum import contract

from ..constant import Nc, Nd
from ..backend import get_backend
from ..preset import Eigenvector
from ..insertion.phase import MomentumPhase

class BaryonElementalGenerator:
    def __init__(
            self,
            latt_size: List[int],
            #gauge_field: GaugeField,
            eigenvector: Eigenvector,
            momentum_list: List[Tuple[int]] = [(0,0,0)],
    )-> None:
        
        backend = get_backend()
        #Lx, Ly, Lz, Lt = latt_size
        
        #Here would be the kernel for the smearing
        #So far no derivative operator implemented. 

        self.latt_size = latt_size
        #self.gauge_field = gauge_field
        self.eigenvector = eigenvector
        self.num_mom = len(momentum_list)
        self.momentum_list = momentum_list
        Ne = eigenvector.Ne
        self.Ne = eigenvector.Ne
        self._mommenum_phase = MomentumPhase(latt_size)
        self._eigenvector_data = None

        self._VVV = backend.zeros((self.num_mom,Ne, Ne, Ne), dtype=complex)

    def load(self, configuration: str):
        #TODO: Implement correctly
        #self.gauge_field.load(configuration)
        print("Loading eigenvectors..., ne:", self.Ne)
        self._eigenvector_data= self.eigenvector.load(configuration)
        print(f"Loaded eigenvectors from {configuration}, shape: {self._eigenvector_data.shape}, dtype: {self._eigenvector_data.dtype}")

    def momentum_phase(self):
        momentum_phase = self._mommenum_phase
        for momentum_idx, momentum in enumerate(self.momentum_list):
            _phase = momentum_phase.get(momentum)
            return _phase # If there is a list of momentum, it will return only the first one. Thats okay because this is for testing purposes.

    def calc(self, t:int):
        eigenvector = self._eigenvector_data[t]
        momentum_phase = self._mommenum_phase
        VVV = self._VVV

        for momentum_idx, momentum in enumerate(self.momentum_list):
            _phase = momentum_phase.get(momentum)
            _contractZero = contract("xyz,axyz->axyz",_phase, eigenvector[...,0])
            _contractOne = eigenvector[...,1]
            _contractTwo = eigenvector[...,2]

            VVV[momentum_idx] = 0
            VVV[momentum_idx] += contract("axyz,bxyz,cxyz->abc",_contractZero,_contractOne,_contractTwo)
            VVV[momentum_idx] += contract("axyz,bxyz,cxyz->abc",_contractOne,_contractTwo,_contractZero)
            VVV[momentum_idx] += contract("axyz,bxyz,cxyz->abc",_contractTwo,_contractZero,_contractOne)
            VVV[momentum_idx] += -contract("axyz,bxyz,cxyz->abc",_contractOne,_contractZero,_contractTwo)
            VVV[momentum_idx] += -contract("axyz,bxyz,cxyz->abc",_contractZero,_contractTwo,_contractOne)
            VVV[momentum_idx] += -contract("axyz,bxyz,cxyz->abc",_contractTwo,_contractOne,_contractZero)

        return VVV