"""
Comprehensive Benchmarking Script
==================================
Compares Paper 1 (Lovász Theta) and Paper 2 (Perfect Graph Coloring) algorithms
across a diverse suite of benchmark graphs.

Usage:
    python benchmark_runner.py

Output:
    - Timing analysis for both algorithms
    - Algorithm correctness verification
    - Comparative performance metrics
    - Results table in CSV format
"""

import time
import sys
from typing import Dict, List, Tuple
from io import StringIO

# Import all modules
from graph_core import Graph
from paper1_lovasz import LovaszTheta
from paper2_coloring import CombinatorialColouringSolver
from benchmark_graphs import BenchmarkGraphs


class BenchmarkRunner:
    """Main benchmarking orchestrator."""

    def __init__(self):
        """Initialize benchmark runner."""
        self.results = []
        self.errors = []

    def benchmark_graph(self, graph_name: str, graph_factory, description: str):
        """
        Run benchmarks on a single graph.
        
        Args:
            graph_name: Display name
            graph_factory: Function that creates the graph
            description: Graph description
        
        Returns:
            Dict with benchmark results
        """
        print(f"\n{'='*80}")
        print(f"Testing: {graph_name}")
        print(f"Description: {description}")
        print(f"{'='*80}")
        
        try:
            # Create graph
            start_create = time.time()
            graph = graph_factory()
            time_create = time.time() - start_create
            
            n = len(graph)
            m = graph.number_of_edges()
            density = graph.density()
            
            print(f"Graph Size: n={n}, m={m}, density={density:.4f}")
            
            # Compute graph properties
            omega = graph.get_omega()
            alpha = graph.get_independence_number()
            
            print(f"Graph Properties: ω(G)={omega}, α(G)={alpha}")
            
            # ============ PAPER 1: LOVÁSZ THETA ============
            print(f"\n--- Paper 1: Lovász Theta Function ---")
            
            try:
                start_theta_simple = time.time()
                theta_simple = LovaszTheta.compute_theta_simple(graph)
                time_theta_simple = time.time() - start_theta_simple
                print(f"  Simple method: θ={theta_simple:.6f}, time={time_theta_simple*1000:.3f}ms")
            except Exception as e:
                theta_simple = None
                time_theta_simple = None
                print(f"  Simple method: ERROR - {str(e)[:50]}")
            
            try:
                start_theta_eigen = time.time()
                theta_eigen = LovaszTheta.compute_theta_eigenvalue_approximation(graph)
                time_theta_eigen = time.time() - start_theta_eigen
                print(f"  Eigenvalue method: θ={theta_eigen:.6f}, time={time_theta_eigen*1000:.3f}ms")
            except Exception as e:
                theta_eigen = None
                time_theta_eigen = None
                print(f"  Eigenvalue method: ERROR - {str(e)[:50]}")
            
            try:
                start_theta_exact = time.time()
                theta_exact = LovaszTheta.compute_theta_exact_small_graphs(graph)
                time_theta_exact = time.time() - start_theta_exact
                print(f"  Exact small graphs: θ={theta_exact:.6f}, time={time_theta_exact*1000:.3f}ms")
            except Exception as e:
                theta_exact = None
                time_theta_exact = None
                print(f"  Exact small graphs: ERROR - {str(e)[:50]}")
            
            candidates = [t for t in [theta_simple, theta_eigen, theta_exact] if t is not None]
            theta_best = max(candidates) if candidates else None
            time_theta_total = (time_theta_simple or 0) + (time_theta_eigen or 0) + (time_theta_exact or 0)
            
            # ============ PAPER 2: PERFECT GRAPH COLORING ============
            print(f"\n--- Paper 2: Perfect Graph Coloring ---")
            
            try:
                solver = CombinatorialColouringSolver(graph)
                start_coloring = time.time()
                coloring = solver.solve()
                time_coloring = time.time() - start_coloring
                
                if coloring:
                    num_colors = max(coloring.values())
                    is_valid = solver.verify_coloring(coloring)
                    print(f"  Coloring: {num_colors} colors, time={time_coloring*1000:.3f}ms")
                    print(f"  Valid coloring: {is_valid}")
                    print(f"  Colors used: {sorted(set(coloring.values()))}")
                else:
                    num_colors = None
                    is_valid = False
                    print(f"  Coloring: FAILED (no coloring found), time={time_coloring*1000:.3f}ms")
            except Exception as e:
                num_colors = None
                time_coloring = None
                is_valid = False
                print(f"  Coloring: ERROR - {str(e)[:50]}")
            
            # ============ VERIFICATION & BOUNDS ============
            print(f"\n--- Theoretical Bounds Verification ---")
            
            if theta_best and omega:
                print(f"  ω(G)={omega} ≤ θ(G)={theta_best:.4f}: {omega <= theta_best}")
                print(f"  θ(G)={theta_best:.4f} ≥ χ(G)≥ω(G)={omega}: {theta_best >= omega}")
            
            if num_colors and omega:
                print(f"  χ(G)={num_colors} ≥ ω(G)={omega}: {num_colors >= omega}")
            
            # ============ COMPUTE RATIO & SPEEDUP ============
            if time_coloring and time_theta_total:
                speedup = time_coloring / time_theta_total
            else:
                speedup = None
            
            # Build result record
            result = {
                'name': graph_name,
                'description': description,
                'n': n,
                'm': m,
                'density': density,
                'omega': omega,
                'alpha': alpha,
                'theta_best': theta_best,
                'num_colors': num_colors,
                'coloring_valid': is_valid,
                'time_theta_simple': time_theta_simple,
                'time_theta_eigen': time_theta_eigen,
                'time_theta_exact': time_theta_exact,
                'time_theta_total': time_theta_total,
                'time_coloring': time_coloring,
                'speedup': speedup,
                'status': 'OK' if num_colors is not None else 'TIMEOUT/ERROR'
            }
            
            self.results.append(result)
            return result
        
        except Exception as e:
            print(f"CRITICAL ERROR: {str(e)}")
            self.errors.append((graph_name, str(e)))
            return None

    def run_full_benchmark_suite(self):
        """Run complete benchmark suite."""
        benchmarks = BenchmarkGraphs.get_all_benchmarks()
        
        print(f"\n{'='*80}")
        print("COMPREHENSIVE ALGORITHM BENCHMARKING SUITE")
        print("Comparing Paper 1 (Lovász Theta) and Paper 2 (Perfect Graph Coloring)")
        print(f"{'='*80}")
        print(f"Total graphs to test: {len(benchmarks)}\n")
        
        for idx, (graph_name, (factory, _, _, desc)) in enumerate(benchmarks.items(), 1):
            print(f"\n[{idx}/{len(benchmarks)}]", end=" ")
            self.benchmark_graph(graph_name, factory, desc)
        
        print(f"\n{'='*80}")
        print("BENCHMARK SUITE COMPLETED")
        print(f"{'='*80}\n")

    def print_summary_table(self):
        """Print summary table of results."""
        if not self.results:
            print("No results to display")
            return
        
        print(f"\n{'='*150}")
        print("SUMMARY TABLE")
        print(f"{'='*150}\n")
        
        # Header
        header = f"{'Graph':<25} {'n':>4} {'m':>5} {'ω':>3} {'α':>3} {'θ':>8} {'χ':>3} {'χ Valid':>8} {'Θ Time(ms)':>12} {'Color Time(ms)':>15} {'Speedup':>10} {'Status':>10}"
        print(header)
        print("-" * 150)
        
        # Rows
        for r in self.results:
            theta_str = f"{r['theta_best']:.2f}" if r['theta_best'] else "N/A"
            chi_str = f"{r['num_colors']}" if r['num_colors'] else "N/A"
            valid_str = "✓" if r['coloring_valid'] else "✗"
            theta_time = f"{r['time_theta_total']*1000:.2f}" if r['time_theta_total'] else "N/A"
            color_time = f"{r['time_coloring']*1000:.2f}" if r['time_coloring'] else "N/A"
            speedup_str = f"{r['speedup']:.2f}x" if r['speedup'] else "N/A"
            
            line = f"{r['name']:<25} {r['n']:>4} {r['m']:>5} {r['omega']:>3} {r['alpha']:>3} {theta_str:>8} {chi_str:>3} {valid_str:>8} {theta_time:>12} {color_time:>15} {speedup_str:>10} {r['status']:>10}"
            print(line)
        
        print("-" * 150)

    def print_csv_export(self, filename="benchmark_results.csv"):
        """Export results to CSV."""
        if not self.results:
            print("No results to export")
            return
        
        with open(filename, 'w') as f:
            # Header
            header = "Graph,Description,n,m,Density,omega,alpha,Theta,Num_Colors,Coloring_Valid,Theta_Time_ms,Coloring_Time_ms,Speedup,Status\n"
            f.write(header)
            
            # Rows
            for r in self.results:
                theta_str = f"{r['theta_best']:.6f}" if r['theta_best'] is not None else "N/A"
                theta_time_str = f"{r['time_theta_total']*1000:.3f}" if r['time_theta_total'] is not None else "N/A"
                color_time_str = f"{r['time_coloring']*1000:.3f}" if r['time_coloring'] is not None else "N/A"
                speedup_str = f"{r['speedup']:.2f}" if r['speedup'] is not None else "N/A"
            line = (
                f"{r['name']},"
                f"{r['description']},"
                f"{r['n']},"
                f"{r['m']},"
                f"{r['density']:.4f},"
                f"{r['omega']},"
                f"{r['alpha']},"
                f"{theta_str},"
                f"{r['num_colors'] if r['num_colors'] is not None else 'N/A'},"
                f"{r['coloring_valid']},"
                f"{theta_time_str},"
                f"{color_time_str},"
                f"{speedup_str},"
                f"{r['status']}\n"
            )
            f.write(line)
        print(f"Results exported to {filename}")

    def print_analysis_report(self):
        """Print detailed analysis report."""
        if not self.results:
            print("No results to analyze")
            return
        
        print(f"\n{'='*80}")
        print("DETAILED ANALYSIS REPORT")
        print(f"{'='*80}\n")
        
        # Category 1: Coloring Performance
        print("COLORING ALGORITHM PERFORMANCE:")
        print("-" * 80)
        
        valid_colorings = [r for r in self.results if r['coloring_valid']]
        print(f"  Valid colorings: {len(valid_colorings)}/{len(self.results)}")
        
        if valid_colorings:
            avg_color_time = sum(r['time_coloring'] for r in valid_colorings if r['time_coloring']) / len([r for r in valid_colorings if r['time_coloring']])
            max_color_time = max(r['time_coloring'] for r in valid_colorings if r['time_coloring'])
            min_color_time = min(r['time_coloring'] for r in valid_colorings if r['time_coloring'])
            print(f"  Average coloring time: {avg_color_time*1000:.3f}ms")
            print(f"  Max coloring time: {max_color_time*1000:.3f}ms")
            print(f"  Min coloring time: {min_color_time*1000:.3f}ms")
        
        # Category 2: Theta Bounds
        print(f"\nLOVÁSZ THETA PERFORMANCE:")
        print("-" * 80)
        
        avg_theta_time = sum(r['time_theta_total'] for r in self.results if r['time_theta_total']) / len([r for r in self.results if r['time_theta_total']])
        print(f"  Average theta computation time: {avg_theta_time*1000:.3f}ms")
        
        tight_bounds = [r for r in self.results if r['theta_best'] and r['omega'] and r['theta_best'] == r['omega']]
        print(f"  Tight bounds (θ=ω): {len(tight_bounds)}/{len(self.results)}")
        
        # Category 3: Perfect Graph Verification
        print(f"\nPERFECT GRAPH VERIFICATION:")
        print("-" * 80)
        
        perfect_verified = [r for r in self.results if r['num_colors'] and r['omega'] and r['num_colors'] == r['omega']]
        print(f"  Graphs where χ(G) = ω(G): {len(perfect_verified)}/{len(self.results)}")
        
        # Category 4: Efficiency Analysis
        print(f"\nEFFICIENCY ANALYSIS:")
        print("-" * 80)
        
        speedups = [r['speedup'] for r in self.results if r['speedup']]
        if speedups:
            avg_speedup = sum(speedups) / len(speedups)
            max_speedup = max(speedups)
            min_speedup = min(speedups)
            print(f"  Average speedup (Coloring vs Theta): {avg_speedup:.2f}x")
            print(f"  Max speedup: {max_speedup:.2f}x")
            print(f"  Min speedup: {min_speedup:.2f}x")
        
        # Category 5: Size Analysis
        print(f"\nGRAPH SIZE ANALYSIS:")
        print("-" * 80)
        
        avg_n = sum(r['n'] for r in self.results) / len(self.results)
        avg_m = sum(r['m'] for r in self.results) / len(self.results)
        max_n = max(r['n'] for r in self.results)
        max_m = max(r['m'] for r in self.results)
        
        print(f"  Average vertices: {avg_n:.1f}")
        print(f"  Average edges: {avg_m:.1f}")
        print(f"  Max vertices: {max_n}")
        print(f"  Max edges: {max_m}")
        
        # Category 6: Errors
        if self.errors:
            print(f"\nERRORS ENCOUNTERED ({len(self.errors)}):")
            print("-" * 80)
            for graph_name, error in self.errors:
                print(f"  {graph_name}: {error[:60]}")


def main():
    """Main entry point."""
    runner = BenchmarkRunner()
    
    # Run full benchmark suite
    runner.run_full_benchmark_suite()
    
    # Print summary
    runner.print_summary_table()
    
    # Print analysis
    runner.print_analysis_report()
    
    # Export to CSV
    runner.print_csv_export("benchmark_results.csv")
    
    print(f"\n{'='*80}")
    print("BENCHMARKING COMPLETE")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
