#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  1 12:23:13 2019

@author: jiedeng
"""
import numpy as np
import matplotlib.pyplot as plt
test = np.matrix([[1,2,3],[4,5,6]])
# physical properties
c=3*1e8;                           # speed of light in vaccum
h = 6.625*1e-34;                    # Planck constant 
k = 1.38*1e-23;                     # Boltzmann constant

sg = 0.8; pvs = 0.84; sm = 0.56; mm = 11.12; ms = 0.68; spv = 0.52; gs = 1.12;
D = [sg,pvs,sm,mm,ms,spv,gs]
Ti = 4000
T0 = 300
#l7_grad(D,Ti = 4000)

#def l7_grad(D, Ti,T0=300):
#    """
#    derive from bd2_7layers_kir4_grad, 7 layer
#    %bd2_7layers_kir4_grad(D_coe, coe_iso, iso_core,core,core_iso, iso_coe, coe_D,...
#    %                      ab_D_coe, ab_coe_iso,ab_iso_core, ab_core, ab_core_iso,...
#    %                      ab_iso_coe,ab_coe_D, T0, Ti, lamda)
#    %
#    % Input list:
#    % layer thickness (um) - D_coe, coe_iso, iso_core,core,core_iso, iso_coe,coe_D
#    % absorption coefficient (m^-1) - ab_D_coe, ab_coe_iso,ab_iso_core,
#    % ab_core, ab_core_iso, ab_iso_coe,ab_coe_D
#    % T0 (K) - diamond/sample intereface temperature
#    % Ti (K) - core temperature
#    % lamda (m) - wavelength of light
#    % Output list:
#    % I (W/m^3) - intensity of light from D_coe side
#    % Note
#    % absorption coefficient is 0 when absorption is not considered
#    % Temperature profile can be generated by uncomment figure(11), it is
#    % always good T0 re-check it T0 see if T profile makes sense
#    %
#    % Ref. Deng et al., 2017
#    % By Jie Deng, 03/30/2017
#    """ 
#    ######layers######
#    ## D1 D2 D3 D4 D5 D6 D7 
#    ######layers######
D = np.array(D)*1e-6
D_coe    = D1 = D[0];
coe_iso  = D2 = D[1];
iso_core = D3 = D[1];
core     = D4 = D[3];
core_iso = D5 = D[4];
iso_coe  = D6 = D[5];
coe_D    = D7 = D[6];

step  = 0.01e-6
# Absolute distance from diamond, spacing is 0.01 um
z_D_coe    = np.arange(0,D1,step);  n1 = len(z_D_coe);   
z_coe_iso  = np.arange(0,D2,step);  n2 = len(z_coe_iso);   
z_iso_core = np.arange(0,D3,step);  n3 = len(z_iso_core); 
z_core     = np.arange(0,D4,step);
z_core_iso = np.arange(0,D5,step);  n4 = len(z_core_iso);
z_iso_coe  = np.arange(0,D6,step);  n5 = len(z_iso_coe);  
z_coe_D    = np.arange(0,D7,step);  n6 = len(z_coe_D);  

z_D_core = np.concatenate((z_D_coe,(D_coe + z_coe_iso),(D_coe+ coe_iso + z_iso_core)), axis=None); # 0 is D
z_core_D = np.concatenate((z_core_iso,(z_iso_coe+core_iso), (z_coe_D + iso_coe+core_iso)), axis=None); # 0 is iso/coe boundary
D_core   = D_coe+ coe_iso+iso_core; 
core_D   = core_iso+iso_coe+coe_D;

T_D_core = T0+2*(Ti-T0)/(D_core**2)*((z_D_core)*D_core - (z_D_core)**2/2);
T_core_D = T0+2*(Ti-T0)/(core_D**2)*(core_D**2/2 - (z_core_D)**2/2);    

T_D_coe      = T_D_core[0:n1];
T_coe_iso    = T_D_core[(n1):(n1+n2)];
T_iso_core   = T_D_core[(n1+n2):(n1+n2+n3)]; 
T_core       = Ti*np.ones(len(z_core)); 
T_core_iso   = T_core_D[0:n4]; 
T_iso_coe    = T_core_D[(n4):(n4+n5)];
T_coe_D      = T_core_D[(n4+n5):(n4+n5+n6)];

z1 = z_D_coe
z2 = D_coe + z_coe_iso
z3 = D_coe + coe_iso + z_iso_core
z4 = D_coe + coe_iso + iso_core + z_core
z5 = D_coe + coe_iso + iso_core + core + z_core_iso
z6 = D_coe + coe_iso + iso_core + core + core_iso + z_iso_coe 
z7 = D_coe + coe_iso + iso_core + core + core_iso + iso_coe + z_coe_D

plt.figure()
plt.plot(z1, T_D_coe);
plt.plot(z2, T_coe_iso);
plt.plot(z3, T_iso_core);
plt.plot(z4, T_core);
plt.plot(z5, T_core_iso);
plt.plot(z6, T_iso_coe);
plt.plot(z7, T_coe_D);

z = np.concatenate((z1,z2,z3,z4,z5,z6,z7), axis=None)
T = np.concatenate((T_D_coe,T_coe_iso,T_iso_core,T_core,T_core_iso,T_iso_coe,T_coe_D), axis=None)
plt.figure()
plt.plot(z*1e6,T)
plt.xlabel('Distance from upper diamond/sample interface (um)')
plt.ylabel('Temperature (K)')


### to calculate the radiation rom point 0, we need integrate from the other end to 0
## for each wavelength, there is an optical depth profile
lamda = np.linspace(518, 683, 100)*1e-9
N    = len(lamda);
test =  np.linspace(0.01, 0.4, 5)*1e2; # m^-1
ab = np.linspace(2970, 1000, 100)*1e2;
#ab = np.concatenate((test,test,test,test,test,test,test),axis=0)

A1 = ab
A2 = ab
A3 = ab
A4 = np.ones(len(lamda))*1e8
A5 = ab
A6 = ab
A7 = ab
# zA has number of wavelegnth column and number of points row
# ztau = A*distance
zA   = np.ones((len(z),len(lamda)))
ztau = np.ones((len(z),len(lamda)))
I    = np.ones(len(lamda))
for i in range(len(lamda)):
    print(i)
    zab1 = np.ones(len(z1))*A1[i]
    zab2 = np.ones(len(z2))*A2[i]
    zab3 = np.ones(len(z3))*A3[i]
    zab4 = np.ones(len(z4))*A4[i]
    zab5 = np.ones(len(z5))*A5[i]
    zab6 = np.ones(len(z6))*A6[i]
    zab7 = np.ones(len(z7))*A7[i]
    zA[:,i]   = np.concatenate((zab1,zab2,zab3,zab4,zab5,zab6,zab7))
    
for i in range(len(lamda)):
    for j in range(len(z)):
        ztau[j,i] = np.trapz(zA[:(j+1),i],x = z[:(j+1)])
        
for i in range(len(lamda)):  
    temp = 2*np.pi*h*(c**2)/(lamda[i]**5)*1/(np.exp(h*c/(lamda[i]*k*T))-1)
    zI = zA[:,i]*temp*np.exp(-ztau[:,i]);
    I[i] = np.trapz(zI,z)
    
print("absorption")
plt.figure()
plt.subplot(121)
plt.plot(z,zA[:,0])   
plt.subplot(122) 
plt.plot(z,ztau[:,0])
plt.figure()
plt.plot(lamda, I)