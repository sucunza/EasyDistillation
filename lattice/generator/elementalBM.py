from typing import List, Tuple
#This class is created by me. JS
from opt_einsum import contract

from ..constant import Nc, Nd
from ..backend import get_backend
from ..preset import Eigenvector
from ..insertion.phase import MomentumPhase
from time import perf_counter
import gc

class BaryonMesonElementalGenerator:
    def __init__(
            self,
            latt_size: List[int],
            #gauge_field: GaugeField,
            eigenvector: Eigenvector,
            momentum_list: List[Tuple[int]] = [(0,0,0)],
    )-> None:
        #This is the generator for a baryun-meson elemental.
        
        backend = get_backend()
        self.Lx, self.Ly, self.Lz, self.Lt = latt_size
        
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
        print(f"EASYDISTILLATION: BaryonMesonElementalGenerator initialized with Ne={Ne}, num_mom={self.num_mom}, momentum_list={self.momentum_list}")

        self._VVVVdV = backend.zeros((self.num_mom,Ne, Ne, Ne, Ne, Ne), dtype=complex)

    def load(self, configuration: str, t:int = 0) -> None:
        #TODO: Implement correctly
        #self.gauge_field.load(configuration)
        start_load = perf_counter()
        self._eigenvector_data= self.eigenvector.load(configuration)[t] 
        self._eigenvector_data= self._eigenvector_data[:self.Ne,...]
        #print(f"Loaded eigenvectors from {configuration}, shape: {self._eigenvector_data.shape}, size: {self._eigenvector_data.nbytes / 1024**3} Gb in {perf_counter() - start_load:.5f} sec")

    def momentum_phase(self):
        momentum_phase = self._mommenum_phase
        for momentum_idx, momentum in enumerate(self.momentum_list):
            _phase = momentum_phase.get(momentum)
            return _phase # If there is a list of momentum, it will return only the first one. Thats okay because this is for testing purposes.

    def calc(self, t:int):
        eigenvector = self._eigenvector_data#[t]
        momentum_phase = self._mommenum_phase
        VVVVdV = self._VVVVdV

        for momentum_idx, momentum in enumerate(self.momentum_list):
            if momentum ==(0,0,0):
                _contractZero = eigenvector[...,0].reshape(self.Ne, self.Ly*self.Lz*self.Lx)
            else: 
                _phase = momentum_phase.get(momentum)
                _contractZero = contract("xyz,axyz->axyz",_phase, eigenvector[...,0]).reshape(self.Ne, self.Ly*self.Lz*self.Lx)
            
            _contractOne = eigenvector[...,1].reshape(self.Ne, self.Ly*self.Lz*self.Lx)
            _contractTwo = eigenvector[...,2].reshape(self.Ne, self.Ly*self.Lz*self.Lx)
            _contractMeson = contract("axyzA,bxyzA->abxyz", eigenvector.conj(), eigenvector).reshape(self.Ne,self.Ne, self.Ly*self.Lz*self.Lx)#This needs a dagger in one. 

            _contractBaryonMeson = 0 
            _Lx2 = self.Lz*self.Ly
            for xi in range(0,self.Lx): #Contraction of full matrix too big to fit in a single GPU
                _start = xi*_Lx2
                _end = (xi+1)*_Lx2
                _contractBaryonMeson +=  contract("ax,bx,cx,dex->abcde",_contractZero[:,_start:_end],_contractOne[:,_start:_end],_contractTwo[:,_start:_end],_contractMeson[:,:,_start:_end],optimize = 'optimal') 
                _contractBaryonMeson +=  contract("ax,bx,cx,dex->abcde",_contractOne[:,_start:_end],_contractTwo[:,_start:_end],_contractZero[:,_start:_end],_contractMeson[:,:,_start:_end],optimize = 'optimal')
                _contractBaryonMeson +=  contract("ax,bx,cx,dex->abcde",_contractTwo[:,_start:_end],_contractZero[:,_start:_end],_contractOne[:,_start:_end],_contractMeson[:,:,_start:_end],optimize = 'optimal')
                _contractBaryonMeson += -contract("ax,bx,cx,dex->abcde",_contractOne[:,_start:_end],_contractZero[:,_start:_end],_contractTwo[:,_start:_end],_contractMeson[:,:,_start:_end],optimize = 'optimal')
                _contractBaryonMeson += -contract("ax,bx,cx,dex->abcde",_contractZero[:,_start:_end],_contractTwo[:,_start:_end],_contractOne[:,_start:_end],_contractMeson[:,:,_start:_end],optimize = 'optimal')
                _contractBaryonMeson += -contract("ax,bx,cx,dex->abcde",_contractTwo[:,_start:_end],_contractOne[:,_start:_end],_contractZero[:,_start:_end],_contractMeson[:,:,_start:_end],optimize = 'optimal')



            VVVVdV[momentum_idx] = _contractBaryonMeson

            del _contractZero, _contractOne, _contractTwo, _contractMeson, _contractBaryonMeson, self._eigenvector_data
            
        gc.collect() 
        return VVVVdV[0,...]