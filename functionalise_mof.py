import math

import ase as ase
from ase import Atoms, io
import numpy as np
from numpy.linalg import norm

def atoms_of_type(types, element):
    """ returns all atom indices in types that match the symbol element """
    return [i for i, t in enumerate(types) if t == element]

def uc_neighbor_offsets(uc_vectors):
    multipliers = np.array(np.meshgrid([-1, 0, 1],[-1, 0, 1],[-1, 0, 1])).T.reshape(-1, 3)
    # offsets = np.array([[0, 0, 0]])
    # return np.array([(structure.cell * m).sum(axis=1) for m in multipliers]).flatten().reshape(27, 3)
    return [tuple((uc_vectors * m).sum(axis=1)) for m in multipliers]

def remove_duplicates(match_indices):
    match1 = set([tuple(sorted(matches)) for matches in match_indices])
    return [list(m) for m in match1]

def calc_sum_of_squares(positions):
    ss = np.zeros([len(positions), len(positions)])
    for j in range(len(positions)):
        for i in range(j, len(positions)):
            pdiff = positions[j] - positions[i]
            ss[i,j] = np.inner(pdiff, pdiff)
    return ss


def find_pattern_in_structure(structure, pattern):
    """find pattern in structure, where both are ASE atoms objects

    Returns:
        a list of indice lists for each set of matched atoms found
    """

    uc_offsets = uc_neighbor_offsets(structure.cell)

    s_positions = structure.positions
    s_types = list(structure.symbols)

    p_positions = pattern.positions
    p_types = list(pattern.symbols)

    p_ss = calc_sum_of_squares(p_positions)

    for i, pattern_atom_1 in enumerate(pattern):
        # Search instances of first atom in a search pattern
        if i == 0:
            match_indices = [[(idx,  (0., 0., 0.))] for idx in atoms_of_type(s_types, p_types[0])]
            print("round %d: " % i, match_indices)
            continue

        last_match_indices = match_indices
        match_indices = []
        for match in last_match_indices:
            for atom_idx in atoms_of_type(s_types, pattern_atom_1.symbol):
                for uc_offset in uc_offsets:
                    found_match = True
                    for j in range(i):
                        match_idx = match[j][0]
                        match_offset = match[j][1]

                        # we don't need an actual distance here, using the sum of squares instead
                        # saves us the square root calculation and allows us to use an inner product
                        # which is very fast in numpy
                        s_diff = s_positions[match_idx] + match_offset - s_positions[atom_idx] - uc_offset
                        s_ss = np.inner(s_diff, s_diff)

                        if not math.isclose(p_ss[i,j], s_ss, rel_tol=5e-2):
                            found_match = False
                            break

                    # anything that matches the distance to all prior pattern atoms is a good match so far
                    if found_match:
                        match_indices.append(match + [(atom_idx, uc_offset)])

        match_indices = remove_duplicates(match_indices)
        print("round %d: (%d) " % (i, len(match_indices)), match_indices)

    # get ASE atoms objects for each set of indices
    match_atoms = [structure.__getitem__([m[0] for m in match]) for match in match_indices]

    return match_indices, match_atoms


def replace_pattern_in_structure(structure, search_pattern, replace_pattern):
    pass

def rotate_replace_pattern(pattern, pivot_atom_index, axis, angle):

    numatoms = pattern.get_global_number_of_atoms()
    c = math.cos(angle)
    s = math.sin(angle)

    x0 = pattern[pivot_atom_index].position[0]
    y0 = pattern[pivot_atom_index].position[1]
    z0 = pattern[pivot_atom_index].position[2]

#     position_rotated = np.empty([numatoms, 3])

    for i in range(numatoms):

        dx = pattern[i].position[0] - x0
        dy = pattern[i].position[1] - y0
        dz = pattern[i].position[2] - z0

        nX = axis[0]
        nY = axis[1]
        nZ = axis[2]

        # dxr, dyr, and dzr are the new, rotated coordinates (assuming the pivot atom is the origin)

        # We use a rotation matrix from axis and angle formula
        dxr = (nX*nX + (1 - nX*nX)*c)*dx +  (nX*nY*(1 - c) - nZ*s)*dy +  (nX*nZ*(1 - c) + nY*s)*dz
        dyr = (nX*nY*(1 - c) + nZ*s)*dx + (nY*nY + (1 - nY*nY)*c)*dy +  (nY*nZ*(1 - c) - nX*s)*dz
        dzr = (nX*nZ*(1 - c) - nY*s)*dx +  (nY*nZ*(1 - c) + nX*s)*dy + (nZ*nZ + (1 - nZ*nZ)*c)*dz

#         position_rotated[i][0] = pattern[pivot_atom_index].position[0] + dxr
#         position_rotated[i][1] = pattern[pivot_atom_index].position[1] + dyr
#         position_rotated[i][2] = pattern[pivot_atom_index].position[2] + dzr

        pattern[i].position[0] = x0 + dxr
        pattern[i].position[1] = y0 + dyr
        pattern[i].position[2] = z0 + dzr

    return pattern

def translate_molecule_origin(pattern):

    # Translate the molecule so that the first defined atom's position is at 0,0,0

    numatoms = pattern.get_global_number_of_atoms()
    first_atom_position = pattern[0].position
    origin = np.array([0,0,0])

    dx = origin[0] - first_atom_position[0]
    dy = origin[1] - first_atom_position[1]
    dz = origin[2] - first_atom_position[2]

    for i in range(numatoms):
        pattern[i].position[0] += dx
        pattern[i].position[1] += dy
        pattern[i].position[2] += dz

    return pattern

def translate_replace_pattern(replace_pattern, search_instance):

    # Translate the molecule so that the 0,0,0 coordinate in the replace pattern
    # is at the position of the first atom found in the search pattern

    numatoms = replace_pattern.get_global_number_of_atoms()
    first_atom_position = replace_pattern[0].position
    position = search_instance[0].position


    dx = position[0] - first_atom_position[0]
    dy = position[1] - first_atom_position[1]
    dz = position[2] - first_atom_position[2]

    for i in range(numatoms):
        replace_pattern[i].position[0] += dx
        replace_pattern[i].position[1] += dy
        replace_pattern[i].position[2] += dz

    return replace_pattern

def replace_pattern_orient(search_instance, replace_pattern):

    rt = 0.01; # rotation tolerance error, in radians
    PI = 3.14159265359

    r_pivot_atom_index = replace_pattern[0].index

    s_numatoms = search_instance.get_global_number_of_atoms()
    r_numatoms = replace_pattern.get_global_number_of_atoms()

    s_first_atom_pos = search_instance[0].position
    s_last_atom_pos = search_instance[s_numatoms-1].position
    s_second_atom_pos = search_instance[1].position

    r_first_atom_pos = replace_pattern[0].position
    r_last_atom_pos = replace_pattern[r_numatoms-1].position
    r_second_atom_pos = replace_pattern[1].position

    # Define the vectors

    # first - f, second - c, last - l, s - search, r - replace
    # SF_SL - vector between the first atom and the last atom
    # in the instance of the search pattern in the structure

    SF_SL = np.empty([3])
    SF_SS = np.empty([3])
    RF_RL = np.empty([3])
    RF_RS = np.empty([3])

    # Vectors on the search structure
    SF_SL[0] = s_last_atom_pos[0] - s_first_atom_pos[0]
    SF_SL[1] = s_last_atom_pos[1] - s_first_atom_pos[1]
    SF_SL[2] = s_last_atom_pos[2] - s_first_atom_pos[2]

    SF_SS[0] = s_second_atom_pos[0] - s_first_atom_pos[0]
    SF_SS[1] = s_second_atom_pos[1] - s_first_atom_pos[1]
    SF_SS[2] = s_second_atom_pos[2] - s_first_atom_pos[2]

    # Vectors on the replace
    RF_RL[0] = r_last_atom_pos[0] - r_first_atom_pos[0]
    RF_RL[1] = r_last_atom_pos[1] - r_first_atom_pos[1]
    RF_RL[2] = r_last_atom_pos[2] - r_first_atom_pos[2]

    RF_RS[0] = r_second_atom_pos[0] - r_first_atom_pos[0]
    RF_RS[1] = r_second_atom_pos[1] - r_first_atom_pos[1]
    RF_RS[2] = r_second_atom_pos[2] - r_first_atom_pos[2]

    # Use the dot-product formula to find the angle between the vectors: SF-SL & RF-RL

    arg = np.dot(SF_SL, RF_RL) / (np.linalg.norm(SF_SL) * np.linalg.norm(RF_RL))

    if arg > 1:
        arg = 1
    if arg < -1:
        arg = -1

    theta = math.acos(arg) # Angle beteen two vectors: SF-SL & RF-RL in Radians

    if theta > rt and theta < PI - rt: # Vectors are not parallel or anti-parallel

        # Find the axis of rotation by taking the cross-product of: SF-SL & RF-RL

        crs = np.cross(SF_SL, RF_RL)
        mag = np.linalg.norm(crs)
        mag *= -1
        crs[0] *= (1.0 / mag)
        crs[1] *= (1.0 / mag)
        crs[2] *= (1.0 / mag)
        rotate_replace_pattern(replace_pattern, r_pivot_atom_index, crs, theta)

    elif theta < rt:
        # Vectors are parallel, do nothing
        pass

    else: # Vectors are anti-parallel - rotate by 180 degrees
        # Now we can rotate by an arbitary normal vector. We can generate an arbitrary normal vector
        # by taking the cross-product of RF_RS with RF_RL
        crs = np.cross(RF_RS, RF_RL)
        mag = np.linalg.norm(crs)
        crs[0] *= (-1.0 / mag)
        crs[1] *= (-1.0 / mag)
        crs[2] *= (-1.0 / mag)
        # Now rotate by 180 degrees
        rotate_replace_pattern(replace_pattern, r_pivot_atom_index, crs, theta)

    # Update fgroup vectors after rotation
    r_first_atom_pos = replace_pattern[0].position
    r_last_atom_pos = replace_pattern[r_numatoms-1].position
    r_second_atom_pos = replace_pattern[1].position

    RF_RL[0] = r_last_atom_pos[0] - r_first_atom_pos[0]
    RF_RL[1] = r_last_atom_pos[1] - r_first_atom_pos[1]
    RF_RL[2] = r_last_atom_pos[2] - r_first_atom_pos[2]

    RF_RS[0] = r_second_atom_pos[0] - r_first_atom_pos[0]
    RF_RS[1] = r_second_atom_pos[1] - r_first_atom_pos[1]
    RF_RS[2] = r_second_atom_pos[2] - r_first_atom_pos[2]

    # Next - twist rotation

    normS = np.cross(SF_SL, SF_SS)
    normR = np.cross(RF_RL, RF_RS)

    arg = np.dot(normS, normR) / (np.linalg.norm(normS) * np.linalg.norm(normR))

    if arg > 1:
        arg = 1
    if arg < -1:
        arg = -1

    theta = math.acos(arg)

    if theta > rt and theta < PI - rt: # Vectors are not parallel or anti-parallel

        # Find the axis of rotation by taking the cross-product of: SF-SL & RF-RL

        crs = np.cross(normS, normR)
        mag = np.linalg.norm(crs)
        mag *= -1
        crs[0] *= (1.0 / mag)
        crs[1] *= (1.0 / mag)
        crs[2] *= (1.0 / mag)
        rotate_replace_pattern(replace_pattern, r_pivot_atom_index, crs, theta)

    elif theta < rt:
        # Vectors are parallel, do nothing
        pass

    else: # Rotate around SF_SL vector

        crs = SF_SL
        mag = np.linalg.norm(crs)
        crs[0] *= (1.0 / mag)
        crs[1] *= (1.0 / mag)
        crs[2] *= (1.0 / mag)
        # Now rotate by 180 degrees
        rotate_replace_pattern(replace_pattern, r_pivot_atom_index, crs, theta)
    
