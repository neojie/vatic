#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 09:10:18 2020

write_vasp_xdatcar in ase.io.vasp. does not work
# from ase [atoms] to XDATCAR
# I also put it in ase.io.vasp, but import does not work
@author: jiedeng
"""
import numpy as np
from ase import Atoms
def write_vasp_xdatcar(fd, images, label=None):
    """Write VASP MD trajectory (XDATCAR) file

    Only Vasp 5 format is supported (for consistency with read_vasp_xdatcar)

    Args:
        fd (str, fp): Output file
        images (iterable of Atoms): Atoms images to write. These must have
            consistent atom order and lattice vectors - this will not be
            checked.
        label (str): Text for first line of file. If empty, default to list of
            elements.

    """

    images = iter(images)
    image = next(images)

    if not isinstance(image, Atoms):
        raise TypeError("images should be a sequence of Atoms objects.")

    symbol_count = _symbol_count_from_symbols(image.get_chemical_symbols())

    if label is None:
        label = ' '.join([s for s, _ in symbol_count])
    fd.write(label + '\n')

    # Not using lattice constants, set it to 1
    fd.write('           1\n')

    # Lattice vectors; use first image
    float_string = '{:11.6f}'
    for row_i in range(3):
        fd.write('  ')
        fd.write(' '.join(float_string.format(x) for x in image.cell[row_i]))
        fd.write('\n')

    _write_symbol_count(fd, symbol_count)
    _write_xdatcar_config(fd, image, index=1)
    for i, image in enumerate(images):
        # Index is off by 2: 1-indexed file vs 0-indexed Python;
        # and we already wrote the first block.
        _write_xdatcar_config(fd, image, i + 2)


def _write_xdatcar_config(fd, atoms, index):
    """Write a block of positions for XDATCAR file

    Args:
        fd (fd): writeable Python file descriptor
        atoms (ase.Atoms): Atoms to write
        index (int): configuration number written to block header

    """
    fd.write("Direct configuration={:6d}\n".format(index))
    float_string = '{:11.8f}'
    scaled_positions = atoms.get_scaled_positions()
    for row in scaled_positions:
        fd.write(' ')
        fd.write(' '.join([float_string.format(x) for x in row]))
        fd.write('\n')


def _symbol_count_from_symbols(symbols):
    """Reduce list of chemical symbols into compact VASP notation

    args:
        symbols (iterable of str)

    returns:
        list of pairs [(el1, c1), (el2, c2), ...]
    """
    sc = []
    psym = symbols[0]
    count = 0
    for sym in symbols:
        if sym != psym:
            sc.append((psym, count))
            psym = sym
            count = 1
        else:
            count += 1
    sc.append((psym, count))
    return sc


def _write_symbol_count(fd, sc, vasp5=True):
    """Write the symbols and numbers block for POSCAR or XDATCAR

    Args:
        f (fd): Descriptor for writable file
        sc (list of 2-tuple): list of paired elements and counts
        vasp5 (bool): if False, omit symbols and only write counts

    e.g. if sc is [(Sn, 4), (S, 6)] then write::

      Sn   S
       4   6

    """
    if vasp5:
        for sym, _ in sc:
            fd.write(' {:3s}'.format(sym))
        fd.write('\n')

    for _, count in sc:
        fd.write(' {:3d}'.format(count))
    fd.write('\n')


def write_vasp(filename, atoms, label=None, direct=False, sort=None,
               symbol_count=None, long_format=True, vasp5=False,
               ignore_constraints=False):
    """Method to write VASP position (POSCAR/CONTCAR) files.

    Writes label, scalefactor, unitcell, # of various kinds of atoms,
    positions in cartesian or scaled coordinates (Direct), and constraints
    to file. Cartesian coordinates is default and default label is the
    atomic species, e.g. 'C N H Cu'.
    """

    from ase.constraints import FixAtoms, FixScaled, FixedPlane, FixedLine

    fd = filename  # @writer decorator ensures this arg is a file descriptor

    if isinstance(atoms, (list, tuple)):
        if len(atoms) > 1:
            raise RuntimeError('Don\'t know how to save more than ' +
                               'one image to VASP input')
        else:
            atoms = atoms[0]

    # Check lattice vectors are finite
    if np.any(atoms.get_cell_lengths_and_angles() == 0.):
        raise RuntimeError(
            'Lattice vectors must be finite and not coincident. '
            'At least one lattice length or angle is zero.')

    # Write atom positions in scaled or cartesian coordinates
    if direct:
        coord = atoms.get_scaled_positions()
    else:
        coord = atoms.get_positions()

    constraints = atoms.constraints and not ignore_constraints

    if constraints:
        sflags = np.zeros((len(atoms), 3), dtype=bool)
        for constr in atoms.constraints:
            if isinstance(constr, FixScaled):
                sflags[constr.a] = constr.mask
            elif isinstance(constr, FixAtoms):
                sflags[constr.index] = [True, True, True]
            elif isinstance(constr, FixedPlane):
                mask = np.all(np.abs(np.cross(constr.dir, atoms.cell)) < 1e-5,
                              axis=1)
                if sum(mask) != 1:
                    raise RuntimeError(
                        'VASP requires that the direction of FixedPlane '
                        'constraints is parallel with one of the cell axis')
                sflags[constr.a] = mask
            elif isinstance(constr, FixedLine):
                mask = np.all(np.abs(np.cross(constr.dir, atoms.cell)) < 1e-5,
                              axis=1)
                if sum(mask) != 1:
                    raise RuntimeError(
                        'VASP requires that the direction of FixedLine '
                        'constraints is parallel with one of the cell axis')
                sflags[constr.a] = ~mask

    if sort:
        ind = np.argsort(atoms.get_chemical_symbols())
        symbols = np.array(atoms.get_chemical_symbols())[ind]
        coord = coord[ind]
        if constraints:
            sflags = sflags[ind]
    else:
        symbols = atoms.get_chemical_symbols()

    # Create a list sc of (symbol, count) pairs
    if symbol_count:
        sc = symbol_count
    else:
        sc = _symbol_count_from_symbols(symbols)

    # Create the label
    if label is None:
        label = ''
        for sym, c in sc:
            label += '%2s ' % sym
    fd.write(label + '\n')

    # Write unitcell in real coordinates and adapt to VASP convention
    # for unit cell
    # ase Atoms doesn't store the lattice constant separately, so always
    # write 1.0.
    fd.write('%19.16f\n' % 1.0)
    if long_format:
        latt_form = ' %21.16f'
    else:
        latt_form = ' %11.6f'
    for vec in atoms.get_cell():
        fd.write(' ')
        for el in vec:
            fd.write(latt_form % el)
        fd.write('\n')

    # Write out symbols (if VASP 5.x) and counts of atoms
    _write_symbol_count(fd, sc, vasp5=vasp5)

    if constraints:
        fd.write('Selective dynamics\n')

    if direct:
        fd.write('Direct\n')
    else:
        fd.write('Cartesian\n')

    if long_format:
        cform = ' %19.16f'
    else:
        cform = ' %9.6f'
    for iatom, atom in enumerate(coord):
        for dcoord in atom:
            fd.write(cform % dcoord)
        if constraints:
            for flag in sflags[iatom]:
                if flag:
                    s = 'F'
                else:
                    s = 'T'
                fd.write('%4s' % s)
        fd.write('\n')

