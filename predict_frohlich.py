'''
@author Lauren Smith

Combine LO frequencies and dielectric constants to calculate Frohlich Coefficient

NeuralNetworkRegressorResults_infinity.csv -- contains predicted epsilon_inf
NeuralNetworkRegressorResults_static.csv -- contains predicted epsilon_0
predictedLO.csv -- predicted LO frequencies in cm^-1

MSE 395 -- Group 26a
'''

import numpy as np
import matplotlib.pyplot as plt
from pymatgen.ext.matproj import MPRester, MPRestError

einf_str = np.genfromtxt('NeuralNetworkRegressorResults_infinity.csv',delimiter=',',skip_header=1,dtype=np.str)
e0_str = np.genfromtxt('NeuralNetworkRegressorResults_static.csv',delimiter=',',skip_header=1,dtype=np.str)
lo = np.genfromtxt('predictedLO.csv',delimiter=',')

#only keep permittivity values that also exist in the LO list
einf = np.zeros((len(einf_str),2))
e0 = np.zeros((len(e0_str),2))
for i,e in enumerate(einf_str):
    if 'mp' in e[0]:
        einf[i,0] = e[0][e[0].index('-')+1:]
        einf[i,1] = e[1]
for i,e in enumerate(e0_str):
    if 'mp' in e[0]:
        e0[i,0] = e[0][e[0].index('-')+1:]
        e0[i,1] = e[1]

einf = einf[einf[:,0] != 0]
e0 = e0[e0[:,0] != 0]

e0 = e0[np.argsort(e0[:,0])]
einf = einf[np.argsort(einf[:,0])]

lo[:,1] = lo[:,-1] * 100 #convert from cm^-1 to m^-1

indices = np.array([i for i in lo[:,0] if i in einf[:,0] and i in e0[:,0]],dtype=np.int) #common MPIDs between all 3 sets

#constants
e = 1.602e-19
hbar = 1.0545718e-34
me = 9.10938356e-31
c = 299792458
evac = 8.854e-12

frohlich = np.zeros((len(indices),2))
for j,i in enumerate(indices):
    lo_i = lo[lo[:,0] == i][0]
    einf_i = einf[einf[:,0]==i][0]
    e0_i = e0[e0[:,0]==i][0]
    omega = 2*np.pi*c*lo_i[1]
    frohlich[j,1] = e**2/hbar*np.sqrt(me/(2*hbar*omega))*(1/(evac*einf_i[1]) - 1/(evac*e0_i[1]))
    frohlich[j,0] = i
    if j % 100 == 0:
        print(frohlich[j,1])

frohlich = frohlich[np.argsort(frohlich[:,1])] #sort by frohlich coef

#match MPID with material name
API_Key='OWo1m43CVHt07bR5'
name = []
with MPRester(API_Key) as mp:
    for i in range(0,len(frohlich)):
        MPName='mp-'+str(int(frohlich[i,0]))
        StructDat = mp.query(criteria={"task_id": MPName}, properties=['pretty_formula'])
        try:
            name.append(StructDat[0]['pretty_formula'])
        except:
            name.append('not found') #this case never happens don't worry
            print(MPName)

#format for saving to file
names = np.array(name)
names = names.reshape((len(names),1))
indices_col = frohlich[:,0].reshape((len(frohlich[:,0]),1))
names2 = np.append(indices_col,names,axis=1)

#np.savetxt('frohlich.csv',frohlich,delimiter=',') #MPID, frohlich coef
#np.savetxt('names.csv',np.array(names2),delimiter=',',fmt='%s') #MPID, name

#pretty graph
plt.figure()
plt.hist(frohlich[:,1],bins=100)
plt.xlabel('Frohlich Coefficient')
plt.ylabel('Number')
plt.title('Histogram of Frohlich Coefficient')
plt.xlim([-40,40])
plt.show()
