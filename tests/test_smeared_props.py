import os
import sys
from time import perf_counter
from opt_einsum import contract

test_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(test_dir, ".."))
from lattice import set_backend, get_backend
set_backend("cupy")
backend = get_backend()
from lattice import GaugeFieldIldg,  Nc, Nd
from lattice import EigenvectorNpy,PerambulatorNpy,SmearedPropagatorGenerator, ElementalNpy

test_dir = "/home/javier/EasyDistillation/tests"

lat_size = [4,4,4,8] # Try to see with a smaller for test

Lx, Ly, Lz, Lt = lat_size
Vol = Lx * Ly * Lz * Lt
Ne = 20
Npoints = 64

gauge_field = GaugeFieldIldg("/home/javier/EasyDistillation/tests/",".lime",[Lt,Lz,Ly,Lx,Nd,Nc,Nc])

eigenvector =EigenvectorNpy(f"{test_dir}/", R".eigenvector.input.npy", [Lt, Ne, Lz, Ly, Lx, Nc], Ne)
perambulator = PerambulatorNpy(f"{test_dir}/",".perambulator.npy",[Lt, Lt, Nd, Nd, Ne, Ne], Ne)
elemental = ElementalNpy(f"{test_dir}/",".elemental.npy",[13, 6, Lt, Ne, Ne], Ne)  

smear_prop_object = SmearedPropagatorGenerator(latt_size=lat_size,eigenvector=eigenvector,perambulator=perambulator,Npoints=Npoints,seed=151222)
smear_prop_object.load('weak_field')

start = perf_counter()
smeared_prop = backend.zeros((Lt,Lt,Npoints,Npoints,Nd,Nd,Nc,Nc),backend.complex128)
for t_source in range(Lt):
    start_source = perf_counter()
    smeared_prop[t_source] = smear_prop_object.calc(t_source)
    print(f"Calculated t_source = {t_source} in {perf_counter()-start_source} s")


#gamma5
g5=backend.zeros((4,4),dtype=backend.complex128)
g5[0,0]=1.0+0.0*1j
g5[1,1]=1.0+0.0*1j
g5[2,2]=-1.0+0.0*1j
g5[3,3]=-1.0+0.0*1j

G5perambG5 = contract('AB,CD,twxyCBab->twxyDAab',g5,g5,smeared_prop.conj())
correlator = backend.zeros((Lt),backend.complex128)
for t_source in range(Lt):
    corr_temp = 1 * contract('AB,wyxCBab,CD,wyxDAab->w',g5,smeared_prop[t_source],g5,G5perambG5[t_source])
    #print(f'Corr at t_source {t_source} is {corr_temp.shape}')
    correlator += backend.roll(corr_temp,-t_source)/Lt

correlator = correlator.get()
print(f"Calculated the smeared prop with shape {smeared_prop.shape} and size {smeared_prop.nbytes/1024**3} GB in {perf_counter()-start} s")


start = perf_counter()
elementalM = backend.load(test_dir+"/weak_field.elemental.npy")
print(f'elmentalM shape {elementalM.shape}')
elementalM= elementalM[0,0,:,:,:] #This chooses the zero momentum and zero derivative
peram = backend.load(test_dir+"/weak_field.perambulator.npy")
print(f'Elemental shape {elementalM.shape} perambulator shape {peram.shape}')
correlator_dist = backend.zeros((Lt),backend.complex128)
for t_source in range(Lt):
    peramb = peram[t_source,:,...].transpose(0,3,4,1,2) # (48, 100, 100, 4, 4)
    G5peramG5 = contract("AB,tijBC,CD->tijAD", g5,backend.conj(peramb),g5)
    corr_temp = 1*contract('AB,CD,yij,kl,ykjCB,yliDA->y',g5,g5,elementalM,elementalM[t_source],G5peramG5,peramb)
    #print(f'Corr at t_source {t_source} is {corr_temp.shape}')
    correlator_dist += backend.roll(corr_temp,-t_source)/Lt

print(f"Calculated the dist prop with shape {smeared_prop.shape} and size {smeared_prop.nbytes/1024**3} GB in {perf_counter()-start} s")
correlator_dist = correlator_dist.get()




###################################################################
########Plot the corr and eff mass to check########################
###################################################################

import matplotlib.pyplot as plt
plt.plot(range(Lt),correlator.real,'o',label='Random sampling')
plt.plot(range(Lt),correlator_dist.real,'r^',label='Distillation')
plt.yscale('log')
plt.xlabel('t')
plt.ylabel('C(t)')
plt.savefig('correlator.pdf') 


eff_mass_random = backend.zeros((Lt-1),backend.float64)
eff_mass_dist = backend.zeros((Lt-1),backend.float64)
for t in range(1,Lt):
    eff_mass_random[t-1] = backend.log(backend.abs(correlator[t-1]/correlator[t]))
    eff_mass_dist[t-1] = backend.log(backend.abs(correlator_dist[t-1]/correlator_dist[t]))

eff_mass_random = eff_mass_random.get()
eff_mass_dist = eff_mass_dist.get()

plt.figure()
plt.plot(range(1,Lt),eff_mass_random,'o',label='Random sampling')
plt.plot(range(1,Lt),eff_mass_dist,'r^',label='Distillation')
plt.xlabel('t')
plt.ylabel('Effective mass')
plt.legend()
plt.savefig('eff_mass.pdf')




def check(smeared1,smeared2):
    are_equal = backend.allclose(smeared1, smeared2, rtol=1e-12, atol=1e-12)

    if are_equal:
        print("The two smeared propagators are equal within the tolerance.")
    else:
        print("The two smeared propagators are NOT equal within the tolerance.")
    
    difference = backend.abs(smeared1 - smeared2)
    max_diff = backend.max(difference)
    print(f"Maximum absolute difference: {max_diff}")
    print(f'Mean absolute difference: {backend.mean(difference)}')
    print(f'Standard deviation of difference: {backend.std(difference)}')


check(backend.asarray(smeared_prop),backend.asarray(smeared_prop))