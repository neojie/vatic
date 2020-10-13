#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 23 10:02:41 2019

@author: jiedeng
"""

#!/usr/local/bin/python

#Author: Jeff Houze
#Converts vasp's POSCAR file into a pdb file.

import re			#regular expression package

datain = open('POTCAR', 'r')
potcar = datain.read()
datain.close()

p = re.compile(r"""VRHFIN\s=\S*:""") 	#raw triple quote string, for regexp.
symbols = p.findall(potcar)		#list of strings containing atoms,
					  #and junk spliced out later.

whatfile = input("File to convert: ")  #or "echo file | pos2pdb.py"
datain = open(whatfile, 'r')

datain.readline()                 #POSCAR comment line, useless to me.

scale=float(datain.readline())   	   #assuming ONE value
(a1, a2, a3) = datain.readline().split()   #read in unit cell vectors,
(b1, b2, b3) = datain.readline().split()   #but there are strings
(c1, c2, c3) = datain.readline().split()

a1=float(a1); a2=float(a2); a3=float(a3);  #Now I have numbers.
b1=float(b1); b2=float(b2); b3=float(b3);
c1=float(c1); c2=float(c2); c3=float(c3);

atoms = datain.readline().split()	   #List containing how many of each.

tmp = datain.readline().split()		#tmp need to figure out if this is...
					#Selective dynamics, or not.

if (tmp[0][0] == 'S'):
    SD = True
else:
    SD = False

if (SD):
    CorD = datain.readline().split()
else:
    CorD = tmp

#filename=""				#Building up descriptive filename.
#for i in range(len(atoms)):
#    filename += symbols[i][8:-1]
#filename += ".pdb"



dataout = open('temp.pdb', 'w')
count = 0
for i in range(len(atoms)):		#loop over different atoms
    symbol = symbols[i][8:-1]
    for j in range(int(atoms[i])):	#loop over different posistions
        nt=count+1
        tmp = datain.readline().split()
        a=float(tmp[0]); b=float(tmp[1]); c=float(tmp[2]);

        if (CorD[0][0] == 'D'):		#if DLV calculate cart. pos.
            x = scale * (a1*a + b1*b + c1*c)
            y = scale * (a2*a + b2*b + c2*c)
            z = scale * (a3*a + b3*b + c3*c)
        else:				#are ready in cart.
            x = a
            y = b
            z = c
        
        S="ATOM  %5d %2s   MOL     1  %8.3f%8.3f%8.3f  1.00  0.00\n" % (count, symbol, float(x), float(y), float(z))
        dataout.write(S)

datain.close()
dataout.close()