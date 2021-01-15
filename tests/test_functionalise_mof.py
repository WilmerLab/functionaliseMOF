import importlib
from importlib import resources

import ase
from ase import Atoms
import numpy as np
import pytest
from pytest import approx

from functionalise_mof import find_pattern_in_structure, replace_pattern_in_structure, translate_molecule_origin, translate_replace_pattern, rotate_replace_pattern, replace_pattern_orient
import tests

from scipy.spatial.transform import Rotation as R

def test_find_pattern_in_structure__octane_has_8_carbons():
    # CH3 CH2 CH2 CH2 CH2 CH2 CH2 CH3 #
    with importlib.resources.path(tests, "octane.xyz") as octane_path:
        structure = ase.io.read(octane_path)
    pattern = Atoms('C', positions=[(0, 0, 0)])
    match_indices, match_atoms = find_pattern_in_structure(structure, pattern)
    assert len(match_atoms) == 8
    for pattern_found in match_atoms:
        assert pattern_found.get_chemical_symbols() == ["C"]

def test_find_pattern_in_structure__octane_has_2_CH3():
    # CH3 CH2 CH2 CH2 CH2 CH2 CH2 CH3 #
    with importlib.resources.path(tests, "octane.xyz") as octane_path:
        structure = ase.io.read(octane_path)
    pattern = Atoms('CHHH', positions=[(0, 0, 0), (-0.538, -0.635,  0.672), (-0.397,  0.993,  0.052), (-0.099, -0.371, -0.998)])
    match_indices, match_atoms = find_pattern_in_structure(structure, pattern)
    assert len(match_atoms) == 2
    for pattern_found in match_atoms:
        assert pattern_found.get_chemical_symbols() == ["C", "H", "H", "H"]
        cpos = pattern_found[0].position
        assert ((pattern_found[1].position - cpos) ** 2).sum() == approx(1.18704299, 5e-2)
        assert ((pattern_found[2].position - cpos) ** 2).sum() == approx(1.18704299, 5e-2)
        assert ((pattern_found[3].position - cpos) ** 2).sum() == approx(1.18704299, 5e-2)

def test_find_pattern_in_structure__octane_has_12_CH2():
    # there are technically 12 matches, since each CH3 makes 3 variations of CH2
    # CH3 CH2 CH2 CH2 CH2 CH2 CH2 CH3 #
    with importlib.resources.path(tests, "octane.xyz") as octane_path:
        structure = ase.io.read(octane_path)
    pattern = Atoms('CHH', positions=[(0, 0, 0),(-0.1  , -0.379, -1.017), (-0.547, -0.647,  0.685)])
    match_indices, match_atoms = find_pattern_in_structure(structure, pattern)

    assert len(match_atoms) == 12
    for pattern_found in match_atoms:
        assert pattern_found.get_chemical_symbols() == ["C", "H", "H"]
        cpos = pattern_found[0].position
        assert ((pattern_found[1].position - cpos) ** 2).sum() == approx(1.18704299, 5e-2)
        assert ((pattern_found[2].position - cpos) ** 2).sum() == approx(1.18704299, 5e-2)

def test_find_pattern_in_structure__octane_over_pbc_has_2_CH3():
    # CH3 CH2 CH2 CH2 CH2 CH2 CH2 CH3 #
    with importlib.resources.path(tests, "octane.xyz") as octane_path:
        structure = ase.io.read(octane_path)#[0:4]
        positions = structure.get_positions()

        # move positions to get part of CH3 across two boundary conditions
        positions += -1.8

        # move coordinates into main 15 Å unit cell
        positions %= 15
        structure.set_positions(positions)
        structure.set_cell(15 * np.identity(3))

    pattern = Atoms('CHHH', positions=[(0, 0, 0), (-0.538, -0.635,  0.672), (-0.397,  0.993,  0.052), (-0.099, -0.371, -0.998)])
    match_indices, match_atoms = find_pattern_in_structure(structure, pattern)
    assert len(match_atoms) == 2
    for pattern_found in match_atoms:
        assert pattern_found.get_chemical_symbols() == ["C", "H", "H", "H"]

def test_find_pattern_in_structure__hkust1_unit_cell_has_32_benzene_rings():
    with importlib.resources.path(tests, "HKUST-1_withbonds.cif") as hkust1_path:
        structure = ase.io.read(hkust1_path)
    with importlib.resources.path(tests, "HKUST-1_benzene.xyz") as linker_path:
        pattern = ase.io.read(linker_path)
    match_indices, match_atoms = find_pattern_in_structure(structure, pattern)

    assert len(match_atoms) == 32
    for pattern_found in match_atoms:
        assert pattern_found.get_chemical_symbols() == ['C','C','C','C','C','C','H','H','H']
        assert ((pattern_found[0].position - pattern_found[1].position) ** 2).sum() == approx(5.8620934418, 5e-2)
        assert ((pattern_found[0].position - pattern_found[3].position) ** 2).sum() == approx(1.9523164046, 5e-2)
        assert ((pattern_found[0].position - pattern_found[4].position) ** 2).sum() == approx(7.8072193204, 5e-2)
        assert ((pattern_found[5].position - pattern_found[8].position) ** 2).sum() == approx(0.8683351588, 5e-2)

def test_find_pattern_in_structure__hkust1_unit_cell_has_48_Cu_metal_nodes():
    with importlib.resources.path(tests, "HKUST-1_withbonds.cif") as hkust1_path:
        structure = ase.io.read(hkust1_path)
    pattern = Atoms('Cu', positions=[(0, 0, 0)])
    match_indices, match_atoms = find_pattern_in_structure(structure, pattern)

    assert len(match_atoms) == 48
    for pattern_found in match_atoms:
        assert pattern_found.get_chemical_symbols() == ['Cu']

def test_find_pattern_in_structure__hkust1_3x3x3_supercell_has_864_benzene_rings():
    with importlib.resources.path(tests, "HKUST-1_3x3x3.cif") as hkust1_path:
        structure = ase.io.read(hkust1_path)
    with importlib.resources.path(tests, "HKUST-1_benzene.xyz") as linker_path:
        pattern = ase.io.read(linker_path)
    match_indices, match_atoms = find_pattern_in_structure(structure, pattern)

    assert len(match_atoms) == 864
    for pattern_found in match_atoms:
        assert pattern_found.get_chemical_symbols() == ['C','C','C','C','C','C','H','H','H']
        assert ((pattern_found[0].position - pattern_found[1].position) ** 2).sum() == approx(5.8620934418, 5e-2)
        assert ((pattern_found[0].position - pattern_found[3].position) ** 2).sum() == approx(1.9523164046, 5e-2)
        assert ((pattern_found[0].position - pattern_found[4].position) ** 2).sum() == approx(7.8072193204, 5e-2)
        assert ((pattern_found[5].position - pattern_found[8].position) ** 2).sum() == approx(0.8683351588, 5e-2)

def test_find_pattern_in_structure__hkust1_3x3x3_supercell_has_1296_Cu_metal_nodes():
    with importlib.resources.path(tests, "HKUST-1_3x3x3.cif") as hkust1_path:
        structure = ase.io.read(hkust1_path)
    pattern = Atoms('Cu', positions=[(0, 0, 0)])
    match_indices, match_atoms = find_pattern_in_structure(structure, pattern)

    assert len(match_atoms) == 1296
    for pattern_found in match_atoms:
        assert pattern_found.get_chemical_symbols() == ['Cu']

def test_replace_pattern_in_structure__replace_hydrogens_in_octane_with_nothing():
    # CH3 CH2 CH2 CH2 CH2 CH2 CH2 CH3 #
    with importlib.resources.path(tests, "octane.xyz") as octane_path:
        structure = ase.io.read(octane_path)
    search_pattern = Atoms('H', positions=[(0, 0, 0)])
    replace_pattern = search_pattern

    replaced_structure = replace_pattern_in_structure(structure, search_pattern, replace_pattern)
    assert len(replaced_structure) == 8
    assert replaced_structure.get_chemical_symbols() == ["C"] * 8

def test_replace_pattern_in_structure__replace_hydrogens_in_octane_with_hydrogens():
    # CH3 CH2 CH2 CH2 CH2 CH2 CH2 CH3 #
    with importlib.resources.path(tests, "octane.xyz") as octane_path:
        structure = ase.io.read(octane_path)
    search_pattern = Atoms('H', positions=[(0, 0, 0)])
    replace_pattern = search_pattern

    replaced_structure = replace_pattern_in_structure(structure, search_pattern, replace_pattern)
    assert len(replaced_structure) == 26
    assert replaced_structure.get_chemical_symbols() == ["C", "H", "H", "H", "C", "H", "H",
        "C", "H", "H", "C", "H", "H", "C", "H", "H", "C", "H", "H", "C", "H", "H", "C", "H", "H", "H"]
# TODO: assert positions are the same as when we started

def test_translate_molecule__inital_and_final_linker_atoms_positions_difference_is_consistent():
    with importlib.resources.path(tests, "HKUST-1_benzene.xyz") as linker_path:
        linker = ase.io.read(linker_path)
    linker_positions = np.empty([len(linker), 3])
    for i in range(len(linker)):
        linker_positions[i][0] = linker[i].position[0]
        linker_positions[i][1] = linker[i].position[1]
        linker_positions[i][2] = linker[i].position[2]

    translated_linker = translate_molecule_origin(linker)
    assert len(translated_linker) == 9
    assert translated_linker.get_chemical_symbols() == ["C", "C", "C", "C", "C", "C", "H", "H", "H"]

    for j in range(len(translated_linker)):
        assert translated_linker[j].position[0] - linker_positions[j][0] == approx(-4.68905, 5e-2)
        assert translated_linker[j].position[1] - linker_positions[j][1] == approx(-23.36598, 5e-2)
        assert translated_linker[j].position[2] - linker_positions[j][2] == approx(-8.48192, 5e-2)

    assert ((translated_linker[0].position - translated_linker[1].position) ** 2).sum() == approx(5.8620934418, 5e-2)
    assert ((translated_linker[0].position - translated_linker[3].position) ** 2).sum() == approx(1.9523164046, 5e-2)
    assert ((translated_linker[0].position - translated_linker[4].position) ** 2).sum() == approx(7.8072193204, 5e-2)
    assert ((translated_linker[5].position - translated_linker[8].position) ** 2).sum() == approx(0.8683351588, 5e-2)

def test_translate_replace_pattern_inital_and_final_linker_atoms_positions_difference_is_consistent():
    with importlib.resources.path(tests, "HKUST-1_benzene.xyz") as search_path:
        search_pattern = ase.io.read(search_path)
    with importlib.resources.path(tests, "HKUST-1_benzene_replace.xyz") as replace_path:
        replace_pattern = ase.io.read(replace_path)

    linker_positions = np.empty([len(replace_pattern), 3])

    for i in range(len(replace_pattern)):
        linker_positions[i][0] = replace_pattern[i].position[0]
        linker_positions[i][1] = replace_pattern[i].position[1]
        linker_positions[i][2] = replace_pattern[i].position[2]

    translated_linker = translate_replace_pattern(replace_pattern, search_pattern)
    assert len(translated_linker) == 9
    assert translated_linker.get_chemical_symbols() == ["C", "C", "C", "C", "C", "C", "H", "H", "H"]

    for j in range(len(translated_linker)):
        assert translated_linker[j].position[0] - linker_positions[j][0] == approx(1.71202757, 5e-2)
        assert translated_linker[j].position[1] - linker_positions[j][1] == approx(1.712034, 5e-2)
        assert translated_linker[j].position[2] - linker_positions[j][2] == approx(-9.37916086, 5e-2)

    assert ((translated_linker[0].position - translated_linker[1].position) ** 2).sum() == approx(5.8620934418, 5e-2)
    assert ((translated_linker[0].position - translated_linker[3].position) ** 2).sum() == approx(1.9523164046, 5e-2)
    assert ((translated_linker[0].position - translated_linker[4].position) ** 2).sum() == approx(7.8072193204, 5e-2)
    assert ((translated_linker[5].position - translated_linker[8].position) ** 2).sum() == approx(0.8683351588, 5e-2)

def test_rotate_replace_pattern_rotation_90_degrees_about_z_axis_at_origin():
    with importlib.resources.path(tests, "HKUST-1_benzene.xyz") as linker_path:
        linker = ase.io.read(linker_path)

    translate_molecule_origin(linker)

    linker_positions = np.empty([len(linker), 3])
    for i in range(len(linker)):
        linker_positions[i][0] = linker[i].position[0]
        linker_positions[i][1] = linker[i].position[1]
        linker_positions[i][2] = linker[i].position[2]

    # Rotate the linker by 90 degrees about z axis at origin (0, 0, 0)
    r =  R.from_euler('z', 90, degrees=True)
    rot_linker_euler = r.apply(linker_positions)

    # Rotate the linker by 90 degrees about z axis at origin (0, 0, 0)
    rotated_linker = rotate_replace_pattern(linker, 0, [0, 0, 1], 1.5707963267948966)

    assert len(rotated_linker) == 9
    assert rotated_linker.get_chemical_symbols() == ["C", "C", "C", "C", "C", "C", "H", "H", "H"]

    for j in range(len(rotated_linker)):
        assert (rotated_linker[j].position[0] - rot_linker_euler[j][0]) ** 2 == approx(0, 5e-2)
        assert (rotated_linker[j].position[1] - rot_linker_euler[j][1]) ** 2 == approx(0, 5e-2)
        assert (rotated_linker[j].position[2] - rot_linker_euler[j][2]) ** 2 == approx(0, 5e-2)

    assert ((rotated_linker[0].position - rotated_linker[1].position) ** 2).sum() == approx(5.8620934418, 5e-2)
    assert ((rotated_linker[0].position - rotated_linker[3].position) ** 2).sum() == approx(1.9523164046, 5e-2)
    assert ((rotated_linker[0].position - rotated_linker[4].position) ** 2).sum() == approx(7.8072193204, 5e-2)
    assert ((rotated_linker[5].position - rotated_linker[8].position) ** 2).sum() == approx(0.8683351588, 5e-2)

def test_rotate_replace_pattern_rotation_180_degrees_about_x_axis_at_origin():
    with importlib.resources.path(tests, "HKUST-1_benzene.xyz") as linker_path:
        linker = ase.io.read(linker_path)

    translate_molecule_origin(linker)

    linker_positions = np.empty([len(linker), 3])
    for i in range(len(linker)):
        linker_positions[i][0] = linker[i].position[0]
        linker_positions[i][1] = linker[i].position[1]
        linker_positions[i][2] = linker[i].position[2]

    # Rotate the linker by 90 degrees about z axis at origin (0, 0, 0)
    r =  R.from_euler('x', 180, degrees=True)
    rot_linker_euler = r.apply(linker_positions)

    # Rotate the linker by 90 degrees about z axis at origin (0, 0, 0)
    rotated_linker = rotate_replace_pattern(linker, 0, [1, 0, 0], 3.1415926535897)

    assert len(rotated_linker) == 9
    assert rotated_linker.get_chemical_symbols() == ["C", "C", "C", "C", "C", "C", "H", "H", "H"]

    for j in range(len(rotated_linker)):
        assert (rotated_linker[j].position[0] - rot_linker_euler[j][0]) ** 2 == approx(0, 5e-2)
        assert (rotated_linker[j].position[1] - rot_linker_euler[j][1]) ** 2 == approx(0, 5e-2)
        assert (rotated_linker[j].position[2] - rot_linker_euler[j][2]) ** 2 == approx(0, 5e-2)

    assert ((rotated_linker[0].position - rotated_linker[1].position) ** 2).sum() == approx(5.8620934418, 5e-2)
    assert ((rotated_linker[0].position - rotated_linker[3].position) ** 2).sum() == approx(1.9523164046, 5e-2)
    assert ((rotated_linker[0].position - rotated_linker[4].position) ** 2).sum() == approx(7.8072193204, 5e-2)
    assert ((rotated_linker[5].position - rotated_linker[8].position) ** 2).sum() == approx(0.8683351588, 5e-2)

def test_rotate_replace_pattern_rotation_360_degrees_about_y_axis_at_origin():
    with importlib.resources.path(tests, "HKUST-1_benzene.xyz") as linker_path:
        linker = ase.io.read(linker_path)

    translate_molecule_origin(linker)

    linker_positions = np.empty([len(linker), 3])
    for i in range(len(linker)):
        linker_positions[i][0] = linker[i].position[0]
        linker_positions[i][1] = linker[i].position[1]
        linker_positions[i][2] = linker[i].position[2]

    # Rotate the linker by 90 degrees about z axis at origin (0, 0, 0)
    rotated_linker = rotate_replace_pattern(linker, 0, [1, 1, 1], 6.2831853071794)

    assert len(rotated_linker) == 9
    assert rotated_linker.get_chemical_symbols() == ["C", "C", "C", "C", "C", "C", "H", "H", "H"]

    for j in range(len(rotated_linker)):
        assert (rotated_linker[j].position[0] - linker_positions[j][0]) ** 2 == approx(0, 5e-2)
        assert (rotated_linker[j].position[1] - linker_positions[j][1]) ** 2 == approx(0, 5e-2)
        assert (rotated_linker[j].position[2] - linker_positions[j][2]) ** 2 == approx(0, 5e-2)

    assert ((rotated_linker[0].position - rotated_linker[1].position) ** 2).sum() == approx(5.8620934418, 5e-2)
    assert ((rotated_linker[0].position - rotated_linker[3].position) ** 2).sum() == approx(1.9523164046, 5e-2)
    assert ((rotated_linker[0].position - rotated_linker[4].position) ** 2).sum() == approx(7.8072193204, 5e-2)
    assert ((rotated_linker[5].position - rotated_linker[8].position) ** 2).sum() == approx(0.8683351588, 5e-2)

def test_replace_pattern_orient_replacing_with_the_same_search_pattern_positions_will_match():
    with importlib.resources.path(tests, "HKUST-1_withbonds.cif") as hkust1_path:
        structure = ase.io.read(hkust1_path)
    with importlib.resources.path(tests, "HKUST-1_benzene.xyz") as linker_path:
        search_pattern = ase.io.read(linker_path)

    match_indices, match_atoms = find_pattern_in_structure(structure, search_pattern)


    replace_pattern = []
    for i in range(len(match_atoms)):
        replace_pattern.append(ase.io.read(linker_path))

    translate_molecule_origin(search_pattern)
    for i in range(len(replace_pattern)):
        translate_molecule_origin(replace_pattern[i])

    for i in range(len(replace_pattern)):
        replace_pattern_orient(search_pattern, replace_pattern[i])

    for i in range(len(replace_pattern)):
        replace_pattern_orient(match_atoms[i], replace_pattern[i])

    for i in range(len(replace_pattern)):
        translate_replace_pattern(replace_pattern[i], match_atoms[i])

    for i in range(len(replace_pattern)):
        diff = match_atoms[i].positions - replace_pattern[i].positions
        for j in range(len(diff)):
            for k in range(3):
                (diff[j][k])**2 == approx(0, abs=1e-3)

        assert len(replace_pattern[i]) == 9
        assert replace_pattern[i].get_chemical_symbols() == ["C", "C", "C", "C", "C", "C", "H", "H", "H"]

        assert ((replace_pattern[i][0].position - replace_pattern[i][1].position) ** 2).sum() == approx(5.8620934418, 5e-2)
        assert ((replace_pattern[i][0].position - replace_pattern[i][3].position) ** 2).sum() == approx(1.9523164046, 5e-2)
        assert ((replace_pattern[i][0].position - replace_pattern[i][4].position) ** 2).sum() == approx(7.8072193204, 5e-2)
        assert ((replace_pattern[i][5].position - replace_pattern[i][8].position) ** 2).sum() == approx(0.8683351588, 5e-2)
