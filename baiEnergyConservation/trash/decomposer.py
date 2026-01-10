"""Three-qubit energy-conserving gate decomposition."""

import numpy as np
from typing import List, Dict, Tuple
from qiskit.circuit import QuantumCircuit
from qiskit.synthesis import OneQubitEulerDecomposer
from qiskit.quantum_info import Operator

from utils import (is_energy_conserving, get_hamming_weight_blocks, extract_two_level_blocks, get_nontrivial_rows_cols, su2_AB_from_U)
from custom_gates import SqrtISWAPGate, SqrtISWAPdgGate, SGate, SdgGate
from utils import TwoLevelUnitary, ControlledTwoLevel, ComplexMatrix, RealVector
import scipy.linalg


class ThreeQubitDecomposer:
    
    """Decomposes 3-qubit energy-conserving unitaries."""
    
    @staticmethod
    def apply_2level_3qubit_unitary(qc: QuantumCircuit, U: np.ndarray, qargs: List[int]):
        """Apply a 2-level unitary from Reck decomposition to a 3-qubit circuit.
        
        Note: Correctly implements up to global phase of the hamming-weight subspace.
        
        Args:
            qc: QuantumCircuit
            U: Unitary acting on 2 levels
            qargs: [q0, q1, q2] physical qubit indices (q0 is MSB)
        """
        q0, q1, q2 = qargs
        
        # Check that U is a two level unitary
        local_i, local_j = get_nontrivial_rows_cols(U)
        
        # Get submatrix
        submatrix = np.zeros((2, 2), dtype=complex)
        submatrix[0, 0] = U[local_i, local_i]
        submatrix[0, 1] = U[local_i, local_j]
        submatrix[1, 0] = U[local_j, local_i]
        submatrix[1, 1] = U[local_j, local_j]
        
        # XYX decomposition
        theta, phi, lam, global_phase = OneQubitEulerDecomposer('XYX').angles_and_phase(
            Operator(submatrix)
        )
        
        # Check the determinant of the submatrix if it is 1 and nonzero phase then get phaseless U and get its decomposition
        det_submatrix = np.linalg.det(submatrix)
        if np.isclose(det_submatrix, 1.0) and not np.isclose(global_phase, 0.0):
            # Recompute decomposition without global phase
            phaseless_U = submatrix * np.exp(-1j * global_phase)
            theta, phi, lam, _ = OneQubitEulerDecomposer('XYX').angles_and_phase(
                Operator(phaseless_U)
            )
        
        # Figure out control and target qubits from U
        bits_i = bin(local_i)[2:].zfill(3)
        bits_j = bin(local_j)[2:].zfill(3)
        print(f"  Applying TLU: |{bits_i}⟩ ↔ |{bits_j}⟩ on local indices {local_i}, {local_j}")
        
        # Find the common bit (control qubit) and its value
        control_bit = None
        control_val = None
        target_bits = []
        
        for bit in range(3):
            bit_i = (local_i >> bit) & 1
            bit_j = (local_j >> bit) & 1
            if bit_i == bit_j:
                control_bit = bit
                control_val = bit_i
            else:
                target_bits.append(bit)
        
        print(f"    Control: q{control_bit}={control_val}, Targets: q{target_bits}")
        print(f"    XYX angles: θ={theta:.3f}, φ={phi:.3f}, λ={lam:.3f}")
        
        # Map bit positions to physical qubits
        control = qargs[control_bit]
        target1 = qargs[target_bits[0]]
        target2 = qargs[target_bits[1]]

        # Apply XYX decomposition using controlled rotations
        GateSynthesizer.controlled_exp_i_alpha_R(qc, -phi/2, control_val, target1, target2, control)
        GateSynthesizer.controlled_exp_i_theta_L(qc, -theta/2, control_val, target1, target2, control)
        GateSynthesizer.controlled_exp_i_alpha_R(qc, -lam/2, control_val, target1, target2, control)
    
    @staticmethod
    def decompose_3qubit_unitary(qc: QuantumCircuit, U: np.ndarray, 
                                qargs: List[int]) -> List[float]:
        """Decompose and apply a 3-qubit energy-conserving unitary.
        
        Args:
            qc: QuantumCircuit to append gates to
            U: 8x8 unitary matrix
            qargs: [q0, q1, q2] physical qubit indices
            
        Returns:
            List of phases for each Hamming weight block
        """
        q0, q1, q2 = qargs
        n = 8
        
        if not is_energy_conserving(U):
            raise ValueError("Unitary is not energy-conserving")
        
        # Group basis states by Hamming weight
        hw_to_indices = {}
        for i in range(n):
            hw = bin(i).count('1')
            if hw not in hw_to_indices:
                hw_to_indices[hw] = []
            hw_to_indices[hw].append(i)
        
        print(f"\nDecomposing 3-qubit EC unitary:")
        
        ret_phases = []
        for hw in sorted(hw_to_indices.keys()):
            indices = hw_to_indices[hw]
            block_size = len(indices)
            
            # Extract block
            block = np.zeros((block_size, block_size), dtype=complex)
            for i, idx_i in enumerate(indices):
                for j, idx_j in enumerate(indices):
                    block[i, j] = U[idx_i, idx_j]
            
            print(f"\nHW={hw} block ({block_size}x{block_size}):")
            
            if block_size == 1:
                phase = np.angle(block[0, 0])
                print(f"  Phase: {phase:.4f} rad (handled separately)")
                ret_phases.append(phase)
            else:
                # Import here to avoid circular dependency
                from reck_decompose import reck_decomposition
                
                # Reck decomposition
                tlus, D = reck_decomposition(block)
                print(f"  {len(tlus)} 2-level unitaries")
                
                # Apply each 2-level unitary (in reverse order)
                for tlu in reversed(tlus):
                    # Expand to full 8x8 space
                    full_U = np.eye(8, dtype=complex)
                    for ii, idx_i in enumerate(indices):
                        for jj, idx_j in enumerate(indices):
                            full_U[idx_i, idx_j] = tlu.dagger().matrix[ii, jj]
                    
                    ThreeQubitDecomposer.apply_2level_3qubit_unitary(qc, full_U, qargs)
                
                # Diagonal phases
                diag_phases = [np.angle(D[i, i]) for i in range(block_size)]
                print(f"  Diagonal phases: {[f'{p:.3f}' for p in diag_phases]}")
                phase = diag_phases[0]
                ret_phases.append(phase)
        
        return ret_phases




class TwoQubitDecomposer:
    """Decomposes 2-qubit energy-conserving unitaries."""
    
    @staticmethod
    def decompose_2qubit_ec_gate(qc: QuantumCircuit, instr, qargs: List[int]) -> List[float]:
        """Decompose a 2-qubit energy-conserving gate.
        
        Args:
            qc: Quantum circuit to append to
            instr: Gate instruction to decompose
            qargs: Qubit indices [q0, q1]
            
        Returns:
            List of phases [theta_0, theta_1, theta_2] for each Hamming weight block
        """
        q0, q1 = qargs[0], qargs[1]
        U = Operator(instr).data
        
        if not is_energy_conserving(U):
            print("Non-energy-conserving 2-qubit gate encountered; applying directly.")
            qc.append(instr, [q0, q1])
            return [0.0, 0.0, 0.0]
        
        blocks = get_hamming_weight_blocks(U)
        
        # HW=0 block (1x1)
        block_0 = blocks.get(0, np.eye(1))
        theta_0 = np.angle(block_0[0, 0])
        
        # HW=1 block (2x2) - main decomposition
        block_1 = blocks.get(1, np.eye(2))
        print(f"2-qubit EC block (HW=1): {block_1}")
        
        theta, phi, lam, global_phase = OneQubitEulerDecomposer('ZYZ').angles_and_phase(
            Operator(block_1)
        )
        
        print(f"2-qubit EC block angles: θ={theta:.3f}, φ={phi:.3f}, "
              f"λ={lam:.3f}, global_phase={global_phase:.3f}")
        
        theta_1 = global_phase
        
        # Apply the decomposition
        qc.rz(-phi/2, q1)
        qc.rz(phi/2, q0)
        qc.append(SqrtISWAPGate(), [q0, q1])
        qc.rz(-theta/2, q1)
        qc.rz(theta/2, q0)
        qc.append(SqrtISWAPdgGate(), [q0, q1])
        qc.rz(-lam/2, q1)
        qc.rz(lam/2, q0)
        
        # HW=2 block (1x1)
        block_2 = blocks.get(2, np.eye(1))
        theta_2 = np.angle(block_2[0, 0])
        
        return [theta_0, theta_1, theta_2]


             


class GateSynthesizer:
    """Synthesizes complex gates from basic energy-conserving primitives."""
    
    @staticmethod
    def apply_cz_decomposition(qc: QuantumCircuit, q0: int, q1: int, ancilla: int):
        """Apply CZ gate using √iSWAP decomposition with ancilla.
        
        The ancilla is assumed to be |0⟩ and should return to |0⟩.
        
        Args:
            qc: Quantum circuit to append to
            q0, q1: Logical qubit indices
            ancilla: Ancilla qubit index
        """
        qc.append(SqrtISWAPGate(), [q0, q1])
        qc.append(SqrtISWAPGate(), [q0, q1])
        qc.append(SqrtISWAPGate(), [q1, ancilla])
        qc.append(SqrtISWAPGate(), [q1, ancilla])
        
        qc.append(SqrtISWAPdgGate(), [q0, q1])
        qc.append(SqrtISWAPdgGate(), [q0, q1])
        
        qc.append(SqrtISWAPGate(), [q0, ancilla])
        qc.append(SqrtISWAPGate(), [q0, ancilla])
        
        qc.append(SGate(), [q0])
        qc.append(SdgGate(), [ancilla])
    
    @staticmethod
    def apply_swap_decomposition(qc: QuantumCircuit, q0: int, q1: int, ancilla: int):
        """Apply SWAP gate using √iSWAP decomposition with ancilla.
        
        The ancilla is assumed to be |0⟩ and should return to |0⟩.
        
        Args:
            qc: Quantum circuit to append to
            q0, q1: Logical qubit indices
            ancilla: Ancilla qubit index
        """
        GateSynthesizer.apply_cz_decomposition(qc, q0, q1, ancilla)
        qc.append(SdgGate(), [q0])
        qc.append(SdgGate(), [q1])
        qc.append(SqrtISWAPGate(), [q0, q1])
        qc.append(SqrtISWAPGate(), [q0, q1])
    
    @staticmethod
    def exp_i_alpha_R(qc: QuantumCircuit, theta: float, q0: int, q1: int):
        """Apply exp(i*alpha*R) rotation.
        
        Args:
            qc: Quantum circuit
            theta: Rotation angle
            q0, q1: Qubit indices
        """
        qc.append(SGate(), [q1])
        qc.append(SqrtISWAPGate(), [q0, q1])
        qc.rz(-theta, q0)
        qc.rz(theta, q1)
        qc.append(SqrtISWAPdgGate(), [q0, q1])
        qc.append(SdgGate(), [q1])
    
    @staticmethod
    def controlled_exp_i_alpha_R(qc: QuantumCircuit, theta: float, b: int, q0: int, q1: int, control: int):
        """Apply controlled exp(i*alpha*R) rotation.
        
        Args:
            qc: Quantum circuit
            theta: Rotation angle
            b: Control bit value (0 or 1)
            q0, q1: Target qubit indices
            control: Control qubit index
        """
        q2 = control
        qc.append(SqrtISWAPGate(), [q1, q2])
        qc.append(SqrtISWAPGate(), [q1, q2])
        
        qc.append(SqrtISWAPdgGate(), [q0, q1])
        qc.append(SqrtISWAPdgGate(), [q0, q1])
    
        qc.append(SqrtISWAPGate(), [q0, q2])
        qc.append(SqrtISWAPGate(), [q0, q2])
        
        qc.append(SGate(), [q0])
        GateSynthesizer.exp_i_alpha_R(qc, (-1)**b * (theta/2), q0, q1)
        qc.append(SdgGate(), [q0])
        
        qc.append(SqrtISWAPdgGate(), [q0, q2])
        qc.append(SqrtISWAPdgGate(), [q0, q2])
        
        qc.append(SqrtISWAPGate(), [q0, q1])
        qc.append(SqrtISWAPGate(), [q0, q1])
        
        qc.append(SqrtISWAPdgGate(), [q1, q2])
        qc.append(SqrtISWAPdgGate(), [q1, q2])
        
        GateSynthesizer.exp_i_alpha_R(qc, theta/2, q0, q1)
    
    @staticmethod
    def controlled_exp_i_theta_L(qc: QuantumCircuit, theta: float, b: int, q0: int, q1: int, control: int):
        """Apply controlled exp(i*theta*L) rotation.
        
        Args:
            qc: Quantum circuit
            theta: Rotation angle
            b: Control bit value (0 or 1)
            q0, q1: Target qubit indices
            control: Control qubit index
        """
        q2 = control
        qc.append(SGate(), [q0])
        GateSynthesizer.controlled_exp_i_alpha_R(qc, theta, b, q0, q1, q2)
        qc.append(SdgGate(), [q0])
        
    @staticmethod
    def apply_controlled_iswap(qc: QuantumCircuit, control: int, target1: int, target2: int, 
                            control_val: int = 1):
        """Apply a controlled-iSWAP gate.
        
        Args:
            qc: QuantumCircuit
            control: Control qubit index
            target1: First target qubit
            target2: Second target qubit
            control_val: Control value (0 or 1)
        """
        # # If control_val is 0, flip the control qubit
        # if control_val == 0:
        #     qc.x(control)
        
        # # Implement controlled-iSWAP using standard gates
        # # iSWAP = exp(i*pi/4) * exp(-i*pi/4 * (XX + YY))
        # # This can be done with: CNOT, Ry, CNOT, Rz, CNOT sequence
        # # Or use your existing gate synthesis methods
        
        # # Simplified version using built-in gates if available:
        # # qc.ciswap(control, target1, target2)  # If available
        
        # # Otherwise decompose manually - this is a placeholder
        # # You may want to use your GateSynthesizer class here
        # pass  # TODO: Implement using your gate synthesis methods
        
        GateSynthesizer.controlled_exp_i_alpha_R(qc, np.pi/2, control_val, target1, target2, control)
        
        # if control_val == 0:
        #     qc.x(control)

    @staticmethod
    def construct_conjugation_gates(b: int, b_prime: int, n_qubits: int) -> Tuple[List[Tuple[int, int, int]], int]:
        """Construct the sequence of controlled-iSWAP gates K that conjugates
        a 2-level unitary from (b, b') to (b'', b') where d(b'', b') = 2.
        
        Args:
            b: First basis state index
            b_prime: Second basis state index  
            n_qubits: Number of qubits
            
        Returns:
            Tuple of (gates_list, b_double_prime) where:
            - gates_list: List of (control_bit, target_bit1, target_bit2) for controlled-iSWAPs
            - b_double_prime: The intermediate state index with d(b_double_prime, b_prime) = 2
        """
        bits_b = bin(b)[2:].zfill(n_qubits)
        bits_b_prime = bin(b_prime)[2:].zfill(n_qubits)
        
        # Find positions where b and b' differ
        diff_positions = [i for i in range(n_qubits) if bits_b[i] != bits_b_prime[i]]
        hamming_dist = len(diff_positions)
        
        if hamming_dist % 2 != 0:
            raise ValueError("Hamming distance must be even for equal Hamming weight states")
        
        t = hamming_dist // 2  # Number of swaps needed
        
        if t <= 1:
            # Already at distance 2 or 0
            return [], b
        
        # Partition differing positions into l_j (1 in b, 0 in b') and r_j (0 in b, 1 in b')
        l_positions = [i for i in diff_positions if bits_b[i] == '1' and bits_b_prime[i] == '0']
        r_positions = [i for i in diff_positions if bits_b[i] == '0' and bits_b_prime[i] == '1']
        
        # Build sequence of controlled-iSWAP gates
        gates = []
        current_state = b
        
        for j in range(t - 1):  # We need t-1 controlled-iSWAP gates
            lj = l_positions[j]
            rj = r_positions[j]
            lj_plus_1 = l_positions[j + 1] if j + 1 < len(l_positions) else l_positions[0]
            
            # Control bit that differentiates current state from b'
            # We need a bit where b' has 0 and current_state has 1
            control_bit = lj_plus_1
            
            gates.append((control_bit, lj, rj))
            
            # Update current state by swapping bits at lj and rj
            bits_current = list(bin(current_state)[2:].zfill(n_qubits))
            bits_current[lj], bits_current[rj] = bits_current[rj], bits_current[lj]
            current_state = int(''.join(bits_current), 2)
        
        return gates, current_state
