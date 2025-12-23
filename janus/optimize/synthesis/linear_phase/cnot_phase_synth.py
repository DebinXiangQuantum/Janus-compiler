"""
Implementation of the GraySynth algorithm for synthesizing CNOT-Phase
circuits with efficient CNOT cost, and the Patel-Hayes-Markov algorithm
for optimal synthesis of linear (CNOT-only) reversible circuits.
"""
from __future__ import annotations

import copy
import numpy as np
from janus.circuit import Circuit as QuantumCircuit
from janus.compat.exceptions import QiskitError
from janus.compat.synthesis.linear import synth_cnot_count_full_pmh as synthesize_cnot_count_pmh


def synthesize_cnot_phase_aam(
    cnots: list[list[int]], angles: list[str], section_size: int = 2
) -> QuantumCircuit:
    r"""This function is an implementation of the `GraySynth` algorithm of
    Amy, Azimadeh and Mosca.

    GraySynth is a heuristic algorithm from [1] for synthesizing small parity networks.
    It is inspired by Gray codes. Given a set of binary strings :math:`S`
    (called ``cnots`` bellow), the algorithm synthesizes a parity network for :math:`S` by
    repeatedly choosing an index :math:`i` to expand and then effectively recursing on
    the co-factors :math:`S_0` and :math:`S_1`, consisting of the strings :math:`y \in S`,
    with :math:`y_i = 0` or :math:`1` respectively. As a subset :math:`S` is recursively expanded,
    ``cx`` gates are applied so that a designated target bit contains the
    (partial) parity :math:`\chi_y(x)` where :math:`y_i = 1` if and only if :math:`y'_i = 1` for all
    :math:`y' \in S`. If :math:`S` contains a single element :math:`\{y'\}`, then :math:`y = y'`,
    and the target bit contains the value :math:`\chi_{y'}(x)` as desired.

    Notably, rather than uncomputing this sequence of ``cx`` (CNOT) gates when a subset :math:`S`
    is finished being synthesized, the algorithm maintains the invariant
    that the remaining parities to be computed are expressed over the current state
    of bits. This allows the algorithm to avoid the 'backtracking' inherent in
    uncomputing-based methods.

    The algorithm is described in detail in section 4 of [1].

    Args:
        cnots: A matrix whose columns are the parities to be synthesized
            e.g.::

                [[0, 1, 1, 1, 1, 1],
                 [1, 0, 0, 1, 1, 1],
                 [1, 0, 0, 1, 0, 0],
                 [0, 0, 1, 0, 1, 0]]

            corresponds to::

                 x1^x2 + x0 + x0^x3 + x0^x1^x2 + x0^x1^x3 + x0^x1

        angles: A list containing all the phase-shift gates which are
            to be applied, in the same order as in ``cnots``. A number is
            interpreted as the angle of :math:`p(angle)`, otherwise the elements
            have to be ``'t'``, ``'tdg'``, ``'s'``, ``'sdg'`` or ``'z'``.

        section_size: The size of every section in the Patel–Markov–Hayes algorithm.
            ``section_size`` must be a factor of the number of qubits.

    Returns:
        The decomposed quantum circuit.

    Raises:
        QiskitError: when dimensions of ``cnots`` and ``angles`` don't align.

    References:
        1. Matthew Amy, Parsiad Azimzadeh, and Michele Mosca.
           *On the controlled-NOT complexity of controlled-NOT–phase circuits.*,
           Quantum Science and Technology 4.1 (2018): 015002.
           `arXiv:1712.01859 <https://arxiv.org/abs/1712.01859>`_
    """
    num_qubits = len(cnots)

    # Create a quantum circuit on num_qubits
    qcir = QuantumCircuit(num_qubits)

    if len(cnots[0]) != len(angles):
        raise QiskitError('Size of "cnots" and "angles" do not match.')

    range_list = list(range(num_qubits))
    epsilon = num_qubits
    sta = []
    cnots_copy = np.transpose(np.array(copy.deepcopy(cnots)))
    # This matrix keeps track of the state in the algorithm
    state = np.eye(num_qubits).astype("int")

    # Check if some phase-shift gates can be applied, before adding any C-NOT gates
    for qubit in range(num_qubits):
        index = 0
        for icnots in cnots_copy:
            if np.array_equal(icnots, state[qubit]):
                if angles[index] == "t":
                    qcir.t(qubit)
                elif angles[index] == "tdg":
                    qcir.tdg(qubit)
                elif angles[index] == "s":
                    qcir.s(qubit)
                elif angles[index] == "sdg":
                    qcir.sdg(qubit)
                elif angles[index] == "z":
                    qcir.z(qubit)
                else:
                    qcir.p(angles[index] % np.pi, qubit)
                del angles[index]
                cnots_copy = np.delete(cnots_copy, index, axis=0)
                if index == len(cnots_copy):
                    break
                index -= 1
            index += 1

    # Implementation of the pseudo-code (Algorithm 1) in the aforementioned paper
    sta.append([cnots, range_list, epsilon])
    while sta:
        [cnots, ilist, qubit] = sta.pop()
        if len(cnots) == 0 or (isinstance(cnots, list) and cnots == []):
            continue
        if 0 <= qubit < num_qubits:
            condition = True
            while condition:
                condition = False
                for j in range(num_qubits):
                    if (j != qubit) and (sum(cnots[j]) == len(cnots[j])):
                        condition = True
                        qcir.cx(j, qubit)
                        state[qubit] ^= state[j]
                        index = 0
                        for icnots in cnots_copy:
                            if np.array_equal(icnots, state[qubit]):
                                if angles[index] == "t":
                                    qcir.t(qubit)
                                elif angles[index] == "tdg":
                                    qcir.tdg(qubit)
                                elif angles[index] == "s":
                                    qcir.s(qubit)
                                elif angles[index] == "sdg":
                                    qcir.sdg(qubit)
                                elif angles[index] == "z":
                                    qcir.z(qubit)
                                else:
                                    qcir.p(angles[index] % np.pi, qubit)
                                del angles[index]
                                cnots_copy = np.delete(cnots_copy, index, axis=0)
                                if index == len(cnots_copy):
                                    break
                                index -= 1
                            index += 1
                        for x in _remove_duplicates(sta + [[cnots, ilist, qubit]]):
                            [cnotsp, _, _] = x
                            if len(cnotsp) == 0 or (isinstance(cnotsp, list) and cnotsp == []):
                                continue
                            for ttt in range(len(cnotsp[j])):
                                cnotsp[j][ttt] ^= cnotsp[qubit][ttt]
        if len(ilist) == 0 or (isinstance(ilist, list) and ilist == []):
            continue
        # See line 18 in pseudo-code of Algorithm 1 in the aforementioned paper
        # this choice of j maximizes the size of the largest subset (S_0 or S_1)
        # and the larger a subset, the closer it gets to the ideal in the
        # Gray code of one CNOT per string.
        j = ilist[np.argmax([[max(list(row).count(0), list(row).count(1)) for row in cnots][k] for k in ilist])]
        cnots0 = []
        cnots1 = []
        for y in list(map(list, zip(*cnots))):
            if y[j] == 0:
                cnots0.append(y)
            elif y[j] == 1:
                cnots1.append(y)
        cnots0 = list(map(list, zip(*cnots0)))
        cnots1 = list(map(list, zip(*cnots1)))
        if qubit == epsilon:
            sta.append([cnots1, list(set(ilist).difference([j])), j])
        else:
            sta.append([cnots1, list(set(ilist).difference([j])), qubit])
        sta.append([cnots0, list(set(ilist).difference([j])), qubit])
    qcir &= synthesize_cnot_count_pmh(state, section_size).inverse()
    return qcir


def _remove_duplicates(lists):
    """
    Remove duplicates in list

    Args:
        lists (list): a list which may contain duplicate elements.

    Returns:
        list: a list which contains only unique elements.
    """

    unique_list = []
    for element in lists:
        if element not in unique_list:
            unique_list.append(element)
    return unique_list


# Backward compatibility alias
synth_cnot_phase_aam = synthesize_cnot_phase_aam
