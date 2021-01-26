import math

import ase as ase
from ase import Atoms, io
import numpy as np
from numpy.linalg import norm
from scipy.spatial import distance
from scipy.spatial.transform import Rotation as R

def atoms_of_type(types, element):
    """ returns all atom indices in types that match the symbol element """
    return [i for i, t in enumerate(types) if t == element]

def uc_neighbor_offsets(uc_vectors):
    multipliers = np.array(np.meshgrid([-1, 0, 1],[-1, 0, 1],[-1, 0, 1])).T.reshape(-1, 3)
    return {tuple((uc_vectors * m).sum(axis=1)) for m in multipliers}

def remove_duplicates(match_indices):
    found_tuples = set()
    new_match_indices = []
    for m in match_indices:
        mkey = tuple(sorted(m))
        if mkey not in found_tuples:
            new_match_indices.append(m)
            found_tuples.add(mkey)
    return new_match_indices

def get_types_ss_map_limited_near_uc(structure, length, cell):
    """
    structure:
    length: the length of the longest dimension of the search pattern
    cell:

    creates master lists of indices, types and positions, for all atoms in the structure and all
    atoms across the PBCs. Limits atoms across PBCs to those that are within a distance of the
    boundary that is less than the length of the search pattern (i.e. atoms further away from the
    boundary than this will never match the search pattern).
    """
    if not (cell.angles() == [90., 90., 90.]).all():
        raise Exception("Currently optimizations do not support unit cell angles != 90")

    uc_offsets = list(uc_neighbor_offsets(structure.cell))
    uc_offsets[uc_offsets.index((0.0, 0.0, 0.0))] = uc_offsets[0]
    uc_offsets[0] = (0.0, 0.0, 0.0)

    s_positions = [structure.positions + uc_offset for uc_offset in uc_offsets]
    s_positions = [x for y in s_positions for x in y]

    s_types = list(structure.symbols) * len(uc_offsets)
    cell = list(structure.cell.lengths())
    index_mapper = []
    s_pos_view = []
    s_types_view = []

    for i, pos in enumerate(s_positions):
        # only currently works for orthorhombic crystals
        if (pos[0] >= -length and pos[0] < length + cell[0] and
                pos[1] >= -length and pos[1] < length + cell[1] and
                pos[2] >= -length and pos[2] < length + cell[2]):
            index_mapper.append(i)
            s_pos_view.append(pos)
            s_types_view.append(s_types[i])

    s_ss = distance.cdist(s_pos_view, s_pos_view, "sqeuclidean")
    return s_types_view, s_ss, index_mapper

def atoms_by_type_dict(atom_types):
    atoms_by_type = {k:[] for k in set(atom_types)}
    for i, k in enumerate(atom_types):
        atoms_by_type[k].append(i)
    return atoms_by_type

def find_pattern_in_structure(structure, pattern):
    """find pattern in structure, where both are ASE atoms objects

    Returns:
        a list of indice lists for each set of matched atoms found
    """
    p_ss = distance.cdist(pattern.positions, pattern.positions, "sqeuclidean")
    s_types_view, s_ss, index_mapper = get_types_ss_map_limited_near_uc(structure, p_ss.max(), structure.cell)
    atoms_by_type = atoms_by_type_dict(s_types_view)

    for i, pattern_atom_1 in enumerate(pattern):
        # Search instances of first atom in a search pattern
        if i == 0:
            # 0,0,0 uc atoms are always indexed first from 0 to # atoms in structure.
            match_index_tuples = [[idx] for idx in atoms_of_type(s_types_view[0: len(structure)], pattern.symbols[0])]
            print("round %d (%d): " % (i, len(match_index_tuples)), match_index_tuples)
            continue

        last_match_index_tuples = match_index_tuples
        match_index_tuples = []
        for match in last_match_index_tuples:
            for atom_idx in atoms_by_type[pattern_atom_1.symbol]:
                found_match = True
                for j in range(i):
                    if not math.isclose(p_ss[i,j], s_ss[match[j], atom_idx], rel_tol=5e-2):
                        found_match = False
                        break

                # anything that matches the distance to all prior pattern atoms is a good match so far
                if found_match:
                    match_index_tuples.append(match + [atom_idx])


        print("round %d: (%d) " % (i, len(match_index_tuples)), match_index_tuples)

    match_index_tuples = remove_duplicates(match_index_tuples)
    return [tuple([index_mapper[m] % len(structure) for m in match]) for match in match_index_tuples]


def position_index_farthest_from_axis(axis, atoms):
    q = quaternion_from_two_axes(axis, [1., 0., 0.])
    ratoms = q.apply(atoms.positions)
    ss = (ratoms[:,1:3] ** 2).sum(axis=1)
    return np.nonzero(ss==ss.max())[0][0]

def quaternion_from_two_axes(p1, p2, axis=None, posneg=1):
    """ returns the quaternion necessary to rotate ax1 to ax2"""
    v1 = np.array(p1) / np.linalg.norm(p1)
    v2 = np.array(p2) / np.linalg.norm(p2)
    angle = posneg * np.arccos(max(-1.0,min(np.dot(v1, v2),1)))
    if axis is None:
        axis = np.cross(v1, v2)
        if np.isclose(axis, [0., 0., 0.], 1e-3).all() and angle != 0.0:
            # the antiparallel case requires we arbitrarily find a orthogonal rotation axis, since the
            # cross product of a two parallel / antiparallel vectors is 0.
            axis = np.cross(v1, np.random.random(3))

    if np.linalg.norm(axis) > 1e-15:
        axis /= np.linalg.norm(axis)
    return R.from_quat([*(axis*np.sin(angle / 2)), np.cos(angle/2)])

def replace_pattern_in_structure(structure, search_pattern, replace_pattern):
    search_pattern = search_pattern.copy()
    replace_pattern = replace_pattern.copy()

    match_indices = find_pattern_in_structure(structure, search_pattern)
    print(match_indices)

    # translate both search and replace patterns so that first atom of search pattern is at the origin
    replace_pattern.translate(-search_pattern.positions[0])
    search_pattern.translate(-search_pattern.positions[0])
    search_axis = search_pattern.positions[-1]
    print("search_axis: ", search_axis)

    if len(search_pattern) > 2:
        orientation_point_index = position_index_farthest_from_axis(search_axis, search_pattern)
        orientation_point = search_pattern.positions[orientation_point_index]
        orientation_axis = orientation_point - (np.dot(orientation_point, search_axis) / np.dot(search_axis, search_axis)) * search_axis
        print("orientation_axis: ", orientation_axis)

    new_structure = structure.copy()
    if len(replace_pattern) > 0:

        for match in match_indices:
            atoms = structure[match]
            print("--------------")
            print("original atoms:\n", atoms.positions)
            new_atoms = replace_pattern.copy()
            print("new atoms:\n", new_atoms.positions)
            if len(atoms) > 1:
                found_axis = atoms.positions[-1] - atoms.positions[0]
                print("found axis: ", found_axis)
                q1 = quaternion_from_two_axes(search_axis, found_axis)
                if q1 is not None:
                    new_atoms.positions = q1.apply(new_atoms.positions)
                    print("q1: ", q1.as_quat())
                    print("new atoms after q1:\n", new_atoms.positions)
                    print("new atoms after q1 (translated):\n", new_atoms.positions + atoms.positions[0])

                if len(atoms) > 2:
                    found_orientation_point = atoms.positions[orientation_point_index] - atoms.positions[0]
                    found_orientation_axis = found_orientation_point - (np.dot(found_orientation_point, found_axis) / np.dot(found_axis, found_axis)) * found_axis
                    print("found orientation_axis: ", found_orientation_axis)
                    q1_o_axis = orientation_axis
                    if q1:
                        q1_o_axis = q1.apply(q1_o_axis)

                    print("(transformed) orientation_axis: ", q1_o_axis)
                    q2 = quaternion_from_two_axes(found_orientation_axis, q1_o_axis, axis=found_axis, posneg=-1)
                    print("orienting: ", found_orientation_point, q1_o_axis, found_orientation_axis, q2)
                    if q2 is not None:
                        print("q2: ", q2.as_quat())
                        new_atoms.positions = q2.apply(new_atoms.positions)
                        print("new atoms after q2:\n", new_atoms.positions)

            # move replacement atoms into correct position
            new_atoms.translate(atoms.positions[0])
            print("new atoms after translate:\n", new_atoms.positions)
            new_structure.extend(new_atoms)

    indices_to_delete = [idx for match in match_indices for idx in match]
    del(new_structure[indices_to_delete])

    return new_structure