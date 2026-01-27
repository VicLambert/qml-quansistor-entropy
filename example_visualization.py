"""Example script demonstrating the visualization module.

This script shows how to use the CliffordCircuitVisualizer and QuimbCircuitVisualizer
to create, simulate, and visualize Clifford quantum circuits.
"""

from src.experiments import (
    CliffordCircuitVisualizer,
    QuimbCircuitVisualizer,
)


def example_pennylane():
    """Example 1: Visualize a Clifford circuit with PennyLane backend."""
    print("Example 1: PennyLane Clifford Circuit Visualization")
    print("-" * 50)

    # Create visualizer with 4 qubits, 3 layers, 10% T-gate doping
    visualizer = CliffordCircuitVisualizer(
        n_qubits=4,
        n_layers=3,
        global_seed=42,
        tdoping=0.1,
    )

    # Create the circuit specification
    visualizer.create_circuit()

    # Simulate using PennyLane backend
    visualizer.simulate()

    # Plot circuit structure (gate layers and connectivity)
    print("\nGenerating circuit structure plot...")
    visualizer.plot_circuit_structure(save_path="pennylane_circuit.png")

    # Plot state probabilities (top 10 basis states)
    print("Generating state probability plot...")
    visualizer.plot_state_probabilities(top_k=10, save_path="pennylane_state.png")

    print("\nDone! Plots saved to pennylane_circuit.png and pennylane_state.png")


def example_quimb_dense():
    """Example 2: Visualize a Clifford circuit with Quimb backend (dense)."""
    print("\n\nExample 2: Quimb Clifford Circuit Visualization (Dense)")
    print("-" * 50)

    # Create visualizer with dense state representation
    visualizer = QuimbCircuitVisualizer(
        n_qubits=5,
        n_layers=3,
        global_seed=123,
        tdoping=0.15,
        state_type="dense",
    )

    # Create and simulate
    visualizer.create_circuit()
    visualizer.simulate()

    # Generate plots
    print("\nGenerating circuit structure plot...")
    visualizer.plot_circuit_structure(save_path="quimb_dense_circuit.png")

    print("Generating state probability plot...")
    visualizer.plot_state_probabilities(top_k=10, save_path="quimb_dense_state.png")

    print("\nDone! Plots saved to quimb_dense_circuit.png and quimb_dense_state.png")


def example_quimb_mps():
    """Example 3: Visualize a Clifford circuit with Quimb backend (MPS)."""
    print("\n\nExample 3: Quimb Clifford Circuit Visualization (MPS)")
    print("-" * 50)

    # Create visualizer with MPS (matrix product state) representation
    visualizer = QuimbCircuitVisualizer(
        n_qubits=8,  # Larger system for MPS benefit
        n_layers=4,
        global_seed=456,
        tdoping=0.2,
        state_type="mps",
        max_bond=64,  # Limit bond dimension for efficiency
    )

    # Create and simulate
    visualizer.create_circuit()
    visualizer.simulate()

    # Generate plots
    print("\nGenerating circuit structure plot...")
    visualizer.plot_circuit_structure(save_path="quimb_mps_circuit.png")

    print("Generating MPS state information plot...")
    visualizer.plot_state_probabilities(save_path="quimb_mps_state.png")

    print("\nDone! Plots saved to quimb_mps_circuit.png and quimb_mps_state.png")


def example_quick_check():
    """Quick example: Use visualize_all() for a complete workflow."""
    print("\n\nExample 4: Quick Check with visualize_all()")
    print("-" * 50)

    visualizer = CliffordCircuitVisualizer(n_qubits=4, n_layers=2, global_seed=789)

    # This method creates circuit, simulates, and generates all plots at once
    visualizer.visualize_all(output_dir="quick_check_output")

    print("\nDone! All plots saved to quick_check_output/ directory")


if __name__ == "__main__":
    print("=" * 60)
    print("Quantum Circuit Visualization Examples")
    print("=" * 60)

    # Uncomment the example(s) you want to run:

    example_pennylane()
    # example_quimb_dense()
    # example_quimb_mps()
    # example_quick_check()

    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)
