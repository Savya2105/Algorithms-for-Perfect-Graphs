#!/usr/bin/env python3
"""
QUICK START GUIDE
=================

Complete Paper Comparison System - Ready to Run

This script demonstrates how to use all components together.
Run this to see everything in action.

Usage:
    python quickstart.py
"""

from graph_core import Graph
from paper1_lovasz import LovaszTheta
from paper2_coloring import CombinatorialColouringSolver, GraphHelpers
from benchmark_graphs import BenchmarkGraphs
import time


def demo_basic_graphs():
    """Demonstrate basic graph operations."""
    print("\n" + "="*80)
    print("DEMO 1: Basic Graph Operations")
    print("="*80 + "\n")
    
    # Pentagon (C5) - the classic example from Paper 1
    print("Creating Pentagon (C5)...")
    pentagon = BenchmarkGraphs.create_pentagon()
    print(f"  Vertices: {pentagon.all_vertices()}")
    print(f"  Size: n={len(pentagon)}, m={pentagon.number_of_edges()}")
    print(f"  ω(G) = {pentagon.get_omega()}")
    print(f"  α(G) = {pentagon.get_independence_number()}")
    
    # Petersen graph - from Paper 1
    print("\nCreating Petersen Graph...")
    petersen = BenchmarkGraphs.create_petersen()
    print(f"  Size: n={len(petersen)}, m={petersen.number_of_edges()}")
    print(f"  ω(G) = {petersen.get_omega()}")
    print(f"  Density: {petersen.density():.4f}")


def demo_paper1_theta():
    """Demonstrate Lovász Theta computation (Paper 1)."""
    print("\n" + "="*80)
    print("DEMO 2: Lovász Theta Function (Paper 1)")
    print("="*80 + "\n")
    
    graphs_to_test = [
        ("Pentagon C5", BenchmarkGraphs.create_pentagon()),
        ("Petersen", BenchmarkGraphs.create_petersen()),
        ("Complete K4", BenchmarkGraphs.create_complete(4)),
        ("K(3,3)", BenchmarkGraphs.create_complete_bipartite(3, 3)),
    ]
    
    for name, graph in graphs_to_test:
        print(f"\n{name}:")
        print(f"  Graph size: n={len(graph)}, m={graph.number_of_edges()}")
        
        # Compute theta using all methods
        results = LovaszTheta.compute_theta_all_methods(graph)
        
        print(f"  θ(G) simple method: {results['simple']:.6f}")
        print(f"  θ(G) eigenvalue method: {results['eigenvalue']:.6f}")
        print(f"  θ(G) exact small graphs: {results['exact_small']:.6f}")
        print(f"  θ(G) BEST: {results['best_estimate']:.6f}")
        
        # Special result for pentagon
        if name == "Pentagon C5":
            import math
            print(f"  → Theoretical: θ(C5) = √5 = {math.sqrt(5):.6f} ✓")


def demo_paper2_coloring():
    """Demonstrate Perfect Graph Coloring (Paper 2)."""
    print("\n" + "="*80)
    print("DEMO 3: Perfect Graph Coloring Algorithm (Paper 2)")
    print("="*80 + "\n")
    
    graphs_to_test = [
        ("Triangle K3", BenchmarkGraphs.create_complete(3)),
        ("Interval Graph", BenchmarkGraphs.create_interval_graph(6)),
        ("Chordal Graph", BenchmarkGraphs.create_chordal_graph(6)),
        ("Bipartite K(4,4)", BenchmarkGraphs.create_complete_bipartite(4, 4)),
    ]
    
    for name, graph in graphs_to_test:
        print(f"\n{name}:")
        print(f"  Graph size: n={len(graph)}, m={graph.number_of_edges()}")
        print(f"  ω(G) = {graph.get_omega()}")
        
        # Color the graph
        solver = CombinatorialColouringSolver(graph)
        start = time.time()
        coloring = solver.solve()
        elapsed = time.time() - start
        
        if coloring:
            num_colors = max(coloring.values())
            is_valid = solver.verify_coloring(coloring)
            print(f"  χ(G) = {num_colors} colors (valid: {is_valid}) - Time: {elapsed*1000:.2f}ms")
            print(f"  Perfect graph check: χ(G)={num_colors}, ω(G)={graph.get_omega()} → {num_colors == graph.get_omega()}")
        else:
            print(f"  Coloring FAILED - Time: {elapsed*1000:.2f}ms")


def demo_bounds_verification():
    """Verify theoretical bounds."""
    print("\n" + "="*80)
    print("DEMO 4: Theoretical Bounds Verification")
    print("="*80 + "\n")
    
    print("For a perfect graph G:")
    print("  Relationship 1: α(G) ≤ θ(G) ≤ χ(G-bar)")
    print("  Relationship 2: χ(G) = ω(G) (defining property)")
    print("  Relationship 3: θ(G) * θ(G-bar) ≥ n")
    
    graph = BenchmarkGraphs.create_pentagon()
    print(f"\nTesting Pentagon C5:")
    
    n = len(graph)
    omega = graph.get_omega()
    alpha = graph.get_independence_number()
    theta = LovaszTheta.compute_theta_simple(graph)
    
    print(f"  n = {n}")
    print(f"  ω(G) = {omega}")
    print(f"  α(G) = {alpha}")
    print(f"  θ(G) = {theta:.4f}")
    
    print(f"\nBound 1: α(G) ≤ θ(G) → {alpha} ≤ {theta:.4f} = {alpha <= theta} ✓")
    
    complement = graph.get_complement()
    theta_complement = LovaszTheta.compute_theta_simple(complement)
    print(f"\nBound 3: θ(G) * θ(Ḡ) ≥ n → {theta:.4f} * {theta_complement:.4f} ≥ {n}")
    print(f"         {theta * theta_complement:.4f} ≥ {n} = {theta * theta_complement >= n - 0.01} ✓")


def demo_benchmark_summary():
    """Run mini benchmark on subset of graphs."""
    print("\n" + "="*80)
    print("DEMO 5: Mini Benchmark (5 representative graphs)")
    print("="*80 + "\n")
    
    mini_benchmarks = {
        "Pentagon": BenchmarkGraphs.create_pentagon,
        "K(3,3)": lambda: BenchmarkGraphs.create_complete_bipartite(3, 3),
        "Interval": lambda: BenchmarkGraphs.create_interval_graph(8),
        "Grid": lambda: BenchmarkGraphs.create_grid_graph(3, 3),
        "K4": lambda: BenchmarkGraphs.create_complete(4),
    }
    
    print(f"{'Graph':<15} {'n':>4} {'ω':>3} {'θ':>8} {'χ':>3} {'θ Time':>10} {'χ Time':>10}")
    print("-" * 65)
    
    for name, factory in mini_benchmarks.items():
        graph = factory()
        n = len(graph)
        omega = graph.get_omega()
        
        # Theta
        start = time.time()
        theta = LovaszTheta.compute_theta_simple(graph)
        time_theta = time.time() - start
        
        # Coloring
        solver = CombinatorialColouringSolver(graph)
        start = time.time()
        coloring = solver.solve()
        time_coloring = time.time() - start
        
        chi = max(coloring.values()) if coloring else 0
        
        print(f"{name:<15} {n:>4} {omega:>3} {theta:>8.4f} {chi:>3} {time_theta*1000:>10.2f}ms {time_coloring*1000:>10.2f}ms")


def main():
    """Run all demos."""
    print("\n" + "="*80)
    print("COMPREHENSIVE PAPER COMPARISON - QUICK START")
    print("="*80)
    print("Paper 1: Lovász Theta Function (1979)")
    print("Paper 2: Perfect Graph Coloring (2017)")
    print("="*80)
    
    # Run all demos
    demo_basic_graphs()
    demo_paper1_theta()
    demo_paper2_coloring()
    demo_bounds_verification()
    demo_benchmark_summary()
    
    # Final instructions
    print("\n" + "="*80)
    print("NEXT STEPS")
    print("="*80)
    print("\n1. Run full benchmark suite:")
    print("   $ python benchmark_runner.py")
    print("\n2. View individual graph tests:")
    print("   - Edit benchmark_runner.py to focus on specific graphs")
    print("   - Or create custom test scripts importing the modules")
    print("\n3. Examine algorithm details:")
    print("   - paper1_lovasz.py - Lovász theta computation")
    print("   - paper2_coloring.py - Perfect graph coloring algorithm")
    print("   - graph_core.py - Core graph operations")
    print("\n4. Generate results:")
    print("   - Results will be saved as benchmark_results.csv")
    print("   - Summary table printed to console")
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()
