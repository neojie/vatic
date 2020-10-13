
# Copyright (c) 2018 Jie Deng
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE

import numpy as np
import pandas as pd
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from collections import OrderedDict


xdatcar = open('XDATCAR', 'r')
xyz = open('XDATCAR_JD.xyz', 'w')
xyz_fract = open('XDATCAR_fract_JD.xyz', 'w')

xdt = xdatcar.readlines()

# invoke readline once, read and then proceed one more step
# 0th -> system name
system = xdt[1]
# 1st  -> scale

scale = float(xdt[1].rstrip('\n'))
print(scale)

#get lattice vectors 2-a1; 3-a2; 4-a3
a1_map = list(map(float,xdt[2].rstrip('\n').split()))
a2_map = list(map(float,xdt[3].rstrip('\n').split()))
a3_map = list(map(float,xdt[4].rstrip('\n').split()))

a1 = scale*np.array(a1_map)
a2 = scale*np.array(a2_map)
a3 = scale*np.array(a3_map)

print(a1)
print(a2)
print(a3)

#Save scaled lattice vectors
lat_rec = open('lattice.vectors', 'w')
lat_rec.write(str(a1[0])+' '+str(a1[1])+' '+str(a1[2])+'\n')
lat_rec.write(str(a2[0])+' '+str(a2[1])+' '+str(a2[2])+'\n')
lat_rec.write(str(a3[0])+' '+str(a3[1])+' '+str(a3[2]))
lat_rec.close()


#Read xdatcar
# 5- elements
element_names = xdatcar.readline().rstrip('\n').split()
ele_name = xdt[5].rstrip('\n').split()
element_dict = {}
# 6- element number
element_numbers = xdatcar.readline().rstrip('\n').split()
ele_str = xdt[6].rstrip('\n').split()
ele_num =list(map(int,ele_str))
ele_sum = np.sum(ele_num)

#xdt 168 = 7 + (ele_sum+1)*iter

# This is the last iter starting row Dire
iter_max = int(xdt[-ele_sum-1].rstrip().split('=')[-1])

row_tot = np.shape(xdt)[0]

if int((row_tot - 7)/(ele_sum+1)) == iter_max:
    print("Max iteration step is", iter_max)
else:
    print("XDATCAR format is wrong")

ele_index      = [ele_name[0]]*ele_num[0]+[ele_name[1]]*ele_num[1]+[ele_name[2]]*ele_num[2]
ele_num_index  = list(np.arange(ele_sum)+1)

for i in np.arange(iter_max):
    iter_num  = i +1;
    iter_num_index = [iter_num]*ele_sum
    arrays = [iter_num_index,ele_index,ele_num_index]
    tuples = list(zip(*arrays))
    index = pd.MultiIndex.from_tuples(tuples, names=['iter','ele', 'ele_num'])
    
    row_start = 9+i*(ele_sum+1)
    row_end   = 9+ele_sum-1+i*(ele_sum+1)
    xyz_fra   = np.loadtxt(xdt[(row_start-1):row_end])   
    xyz_car   = xyz_fra*[a1[0],a2[1],a3[2]]
    #ele1=xyz_fra[:ele_num[0]]
    df_xyz_car = pd.DataFrame(xyz_car,index =index,columns=['x','y','z'])
    df_xyz_fra = pd.DataFrame(xyz_fra,index =index,columns=['x','y','z'])
    if i == 0:
        car = df_xyz_car
        fra = df_xyz_fra
    else:
        car = car.append(df_xyz_car)
        fra = fra.append(df_xyz_fra)
#
# car.loc[1]      - > first iter
# car.loc[(1,'Fe',1),]-> FIRST ITER
#car1 = car
#car1.index=car.index.droplevel()
#slice does not work
#fra.loc[1,'Fe']
#fra.loc[(slice(1,8), ['O', 'Fe'], slice(None)), :]
#fra.loc[(slice(None), ['O'], slice(1)), :]
#####
#### loc axis way > https://pandas.pydata.org/pandas-docs/stable/advanced.html
#fra.loc(axis=0)[:, :, [1, 2]]
#fra.loc(axis=0)[:, :, [1, 2]]['x']
colorlist = ['r','b','k']
mpl.rcParams['legend.fontsize'] = 10
fig = plt.figure()
ax = fig.gca(projection='3d')
for i in np.arange(np.size(ele_num)):
    if i == 0:
        atom_list = np.arange(ele_num[i])+1
    else:     
        atom_list = np.arange(np.sum(ele_num[:i]),np.sum(ele_num[:i+1]),1)+1
    #atom_label = ele_name[i]   
    #atoms = car.loc(axis=0)[:, :, list(atom_list)]
    
    for j in atom_list:
        atom = car.loc(axis=0)[:, :, j]
        
        ### PBC if the migrates exceeds half of the mininal dimension -> cross PB
        pos_diff  = np.diff(atom,axis=0)
        statement = np.abs(pos_diff)>=min([a1[0]*scale, a2[1]*scale, a3[2]*scale])/2
        true_row,true_col = np.where(statement)
        ### divide atom into several groups based on true_row, true col
        seg_num     = np.size(true_row)+1
        iloc_list   = list(np.arange(0,iter_max))    
        # No segments
        if seg_num ==1:
            ax.plot(atom['x'], atom['y'], atom['z'],color=colorlist[i],label=ele_name[i])
        # Segments =1, only one breakpoint
        elif seg_num ==2:
            iloc_list_all = [None]*2
            iloc_list_all[0] = iloc_list[:(true_row[0])+1]
            iloc_list_all[1] = iloc_list[(true_row[0])+1:]
         # Segments =1, only one breakpoint
        else:
            iloc_list_all = [None]*seg_num
            iloc_list_all[0] = iloc_list[:(true_row[0]+1)]
            iloc_list_all[-1] = iloc_list[true_row[-1]+1:]
            for seg in np.arange(seg_num-1):
                if seg+1 < np.size(true_row) and true_row[seg+1] <iter_max:
                    iloc_list_all[seg+1] = iloc_list[(true_row[seg]+1):(true_row[seg+1]+1)]
                    
        for c, value in enumerate(iloc_list_all):
            atom_iloc = atom.iloc[value]            
            ax.plot(atom_iloc['x'], atom_iloc['y'], atom_iloc['z'],color=colorlist[i],label=ele_name[i])
            handles, labels = plt.gca().get_legend_handles_labels()
            by_label = OrderedDict(zip(labels, handles))
            plt.legend(by_label.values(), by_label.keys())


#handles, labels = plt.gca().get_legend_handles_labels()
#by_label = OrderedDict(zip(labels, handles))
#plt.legend(by_label.values(), by_label.keys())
#ax.legend()
plt.show()
xdatcar.close()

### TODO
### make it a class
### PBC
### same atom same clor