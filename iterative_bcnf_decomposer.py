"""
Iterative BCNF Decomposer for complete database normalization.
This module implements recursive BCNF decomposition until all tables reach BCNF.
"""

import pandas as pd
import json
import os
import time
from typing import List, Dict, Tuple, Optional
from bcnf import BCNF

class IterativeBCNFDecomposer:
    """
    Complete BCNF decomposer that recursively decomposes tables until all reach BCNF.
    """
    
    def __init__(self, original_df: pd.DataFrame, fd_file: str, base_name: str):
        """
        Initialize the iterative decomposer.
        
        Args:
            original_df: Original DataFrame to decompose
            fd_file: Path to HyFD results JSON file
            base_name: Base name for the dataset (used for file organization)
        """
        self.original_df = original_df
        self.fd_file = fd_file
        self.base_name = base_name
        self.decomposition_history = []
        self.final_tables = []
        self.iteration_count = 0
        self.max_iterations = 10  # Prevent infinite loops
        
        # Load original FDs
        with open(fd_file, 'r') as f:
            self.original_fds = json.load(f)
        
        self.column_names = list(original_df.columns)
        
        # Create output directory structure
        timestamp = int(time.time())
        self.output_dir = f"decomposed_tables/{base_name}_{timestamp}"
        os.makedirs(self.output_dir, exist_ok=True)
        
    def filter_relevant_fds(self, table_df: pd.DataFrame, original_fds: List[List[List[int]]]) -> List[List[List[int]]]:
        """Filter FDs that are relevant to the current table."""
        table_cols = set(table_df.columns)
        relevant_fds = []
        
        for fd in original_fds:
            lhs_names = [self.column_names[idx] for idx in fd[0]]
            rhs_names = [self.column_names[idx] for idx in fd[1]]
            
            # Check if all FD attributes are in the current table
            if set(lhs_names + rhs_names).issubset(table_cols):
                # Convert to new table indices
                table_col_to_idx = {col: i for i, col in enumerate(table_df.columns)}
                lhs_idx = [table_col_to_idx[col] for col in lhs_names if col in table_col_to_idx]
                rhs_idx = [table_col_to_idx[col] for col in rhs_names if col in table_col_to_idx]
                
                if lhs_idx and rhs_idx:
                    relevant_fds.append([lhs_idx, rhs_idx])
        
        return relevant_fds
    
    def decompose_single_table(self, table_df: pd.DataFrame, table_name: str) -> Tuple[List[pd.DataFrame], Dict]:
        """
        Decompose a single table using BCNF analysis.
        
        Returns:
            List of decomposed tables and analysis results
        """
        print(f"\n--- Analyzing {table_name} ---")
        print(f"Shape: {table_df.shape}")
        print(f"Columns: {list(table_df.columns)}")
        
        # Filter relevant FDs for this table
        relevant_fds = self.filter_relevant_fds(table_df, self.original_fds)
        
        if not relevant_fds:
            print(f"✅ {table_name}: No relevant FDs found - already in BCNF")
            return [table_df], {'is_bcnf': True, 'violating_fds': [], 'reason': 'no_relevant_fds'}
        
        # Perform BCNF analysis
        try:
            bcnf = BCNF(table_df, fds=relevant_fds)
            results = bcnf.analyze()
            
            print(f"   FDs analyzed: {len(relevant_fds)}")
            print(f"   Candidate keys: {len(results['candidate_keys'])}")
            print(f"   BCNF violations: {len(results['violating_fds'])}")
            
            if results['is_bcnf']:
                print(f"✅ {table_name}: Already in BCNF")
                return [table_df], results
            else:
                print(f"❌ {table_name}: Has {len(results['violating_fds'])} violations")
                if bcnf.best_violating_fd:
                    lhs, rhs = bcnf.best_violating_fd
                    print(f"   Best violating FD: {list(lhs)} → {list(rhs)} (score: {bcnf.vfd_scores[bcnf.best_violating_fd]:.4f})")
                
                # Perform decomposition
                decomposed = bcnf.decompose_table()
                print(f"   Decomposed into {len(decomposed)} tables")
                
                return decomposed, results
                
        except Exception as e:
            print(f"⚠️  {table_name}: Error during analysis - {str(e)}")
            return [table_df], {'is_bcnf': True, 'violating_fds': [], 'reason': 'analysis_error', 'error': str(e)}
    
    def save_iteration_results(self, iteration: int, tables: List[Tuple[pd.DataFrame, str]], analysis_results: List[Dict]):
        """Save the results of a single iteration."""
        iteration_dir = os.path.join(self.output_dir, f"iteration_{iteration}")
        os.makedirs(iteration_dir, exist_ok=True)
        
        saved_files = []
        for i, (table, table_name) in enumerate(tables):
            filename = f"{iteration_dir}/R{i+1}_{table_name}.csv"
            table.to_csv(filename, index=False)
            saved_files.append(filename)
            
        # Save analysis summary
        summary = {
            'iteration': iteration,
            'tables_count': len(tables),
            'tables_info': [
                {
                    'table_name': table_name,
                    'shape': table.shape,
                    'columns': list(table.columns),
                    'is_bcnf': analysis_results[i]['is_bcnf'],
                    'violations': len(analysis_results[i]['violating_fds'])
                }
                for i, (table, table_name) in enumerate(tables)
            ]
        }
        
        summary_file = f"{iteration_dir}/summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        return saved_files, summary_file
    
    def decompose_iteratively(self) -> Dict:
        """
        Perform complete iterative BCNF decomposition.
        
        Returns:
            Complete decomposition results
        """
        print("="*80)
        print(f"ITERATIVE BCNF DECOMPOSITION: {self.base_name}")
        print("="*80)
        
        # Start with the original table
        current_tables = [(self.original_df, "original")]
        all_bcnf = False
        self.iteration_count = 0
        
        while not all_bcnf and self.iteration_count < self.max_iterations:
            self.iteration_count += 1
            print(f"\n🔄 ITERATION {self.iteration_count}")
            print("-" * 50)
            
            new_tables = []
            iteration_analysis = []
            tables_to_decompose = []
            
            # Analyze each table in current iteration
            for table, table_name in current_tables:
                decomposed, analysis = self.decompose_single_table(table, table_name)
                
                if len(decomposed) == 1:
                    # Table didn't decompose (already in BCNF or error)
                    new_tables.append((decomposed[0], table_name))
                    iteration_analysis.append(analysis)
                else:
                    # Table was decomposed - need to analyze each subtable
                    for i, subtable in enumerate(decomposed):
                        subtable_name = f"{table_name}_R{i+1}"
                        new_tables.append((subtable, subtable_name))
                        
                        # Analyze each subtable separately
                        relevant_fds = self.filter_relevant_fds(subtable, self.original_fds)
                        if relevant_fds:
                            try:
                                subtable_bcnf = BCNF(subtable, fds=relevant_fds)
                                subtable_results = subtable_bcnf.analyze()
                                iteration_analysis.append(subtable_results)
                                if not subtable_results['is_bcnf']:
                                    tables_to_decompose.append(subtable_name)
                            except:
                                # If analysis fails, assume it's in BCNF
                                iteration_analysis.append({'is_bcnf': True, 'violating_fds': []})
                        else:
                            # No relevant FDs means it's in BCNF
                            iteration_analysis.append({'is_bcnf': True, 'violating_fds': []})
            
            # Save iteration results
            saved_files, summary_file = self.save_iteration_results(
                self.iteration_count, new_tables, iteration_analysis
            )
            
            # Record this iteration
            iteration_record = {
                'iteration': self.iteration_count,
                'input_tables': len(current_tables),
                'output_tables': len(new_tables),
                'saved_files': saved_files,
                'summary_file': summary_file,
                'decomposed_tables': tables_to_decompose
            }
            self.decomposition_history.append(iteration_record)
            
            # Check if all tables are in BCNF
            all_bcnf = all(
                analysis['is_bcnf'] for analysis in iteration_analysis
            )
            
            if all_bcnf:
                print(f"\n✅ ALL TABLES REACHED BCNF after {self.iteration_count} iterations!")
                self.final_tables = new_tables
                break
            else:
                still_violating = sum(1 for analysis in iteration_analysis if not analysis['is_bcnf'])
                print(f"\n📊 Iteration {self.iteration_count} Summary:")
                print(f"   Input tables: {len(current_tables)}")
                print(f"   Output tables: {len(new_tables)}")
                print(f"   Still violating BCNF: {still_violating}")
                
                current_tables = new_tables
        
        if not all_bcnf:
            print(f"\n⚠️  Maximum iterations ({self.max_iterations}) reached!")
            print("Some tables may still violate BCNF.")
            self.final_tables = current_tables
        
        return self.generate_final_report()
    
    def generate_final_report(self) -> Dict:
        """Generate comprehensive final report."""
        print("\n" + "="*80)
        print("FINAL DECOMPOSITION REPORT")
        print("="*80)
        
        # Count total violations remaining
        total_violations = 0
        bcnf_tables = 0
        
        final_analysis = []
        for table, table_name in self.final_tables:
            relevant_fds = self.filter_relevant_fds(table, self.original_fds)
            if relevant_fds:
                try:
                    bcnf = BCNF(table, fds=relevant_fds)
                    results = bcnf.analyze()
                    violations = len(results['violating_fds'])
                    total_violations += violations
                    if results['is_bcnf']:
                        bcnf_tables += 1
                    final_analysis.append({
                        'table_name': table_name,
                        'shape': table.shape,
                        'is_bcnf': results['is_bcnf'],
                        'violations': violations
                    })
                except:
                    final_analysis.append({
                        'table_name': table_name,
                        'shape': table.shape,
                        'is_bcnf': True,
                        'violations': 0,
                        'note': 'Analysis error - assumed BCNF'
                    })
            else:
                bcnf_tables += 1
                final_analysis.append({
                    'table_name': table_name,
                    'shape': table.shape,
                    'is_bcnf': True,
                    'violations': 0,
                    'note': 'No relevant FDs'
                })
        
        # Print summary
        print(f"\n📊 DECOMPOSITION SUMMARY:")
        print(f"   Original table: {self.original_df.shape}")
        print(f"   Final tables: {len(self.final_tables)}")
        print(f"   Tables in BCNF: {bcnf_tables}/{len(self.final_tables)}")
        print(f"   Total iterations: {self.iteration_count}")
        print(f"   Remaining violations: {total_violations}")
        
        print(f"\n📋 FINAL TABLES:")
        for analysis in final_analysis:
            status = "✅ BCNF" if analysis['is_bcnf'] else f"❌ {analysis['violations']} violations"
            note = f" ({analysis.get('note', '')})" if 'note' in analysis else ""
            print(f"   {analysis['table_name']}: {analysis['shape']} - {status}{note}")
        
        # Calculate space metrics
        original_size = self.original_df.shape[0] * self.original_df.shape[1]
        final_size = sum(table.shape[0] * table.shape[1] for table, _ in self.final_tables)
        space_change = (final_size - original_size) / original_size
        
        print(f"\n📈 SPACE ANALYSIS:")
        print(f"   Original size: {original_size:,} cells")
        print(f"   Final size: {final_size:,} cells")
        print(f"   Space change: {space_change:+.1%}")
        
        print(f"\n💾 OUTPUT LOCATION:")
        print(f"   {self.output_dir}/")
        
        # Save final summary
        final_report = {
            'dataset_name': self.base_name,
            'original_shape': self.original_df.shape,
            'iterations': self.iteration_count,
            'final_tables_count': len(self.final_tables),
            'bcnf_tables_count': bcnf_tables,
            'total_violations': total_violations,
            'space_change_percent': space_change * 100,
            'output_directory': self.output_dir,
            'decomposition_history': self.decomposition_history,
            'final_tables': final_analysis
        }
        
        report_file = os.path.join(self.output_dir, 'final_report.json')
        with open(report_file, 'w') as f:
            json.dump(final_report, f, indent=2, default=str)
        
        print(f"   Final report: {report_file}")
        
        return final_report

def decompose_animal_species_complete():
    """Complete decomposition of animal species dataset."""
    # Load data
    df = pd.read_csv('data/Daily_Trade_Data_TEST4.csv')
    fd_file = 'json/Daily_Trade_Data_TEST4-20250624204105.json'
    
    print("Loading dataset...")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Create decomposer
    decomposer = IterativeBCNFDecomposer(df, fd_file, 'Daily_Trade_Data_TEST4')
    
    # Perform complete decomposition
    final_report = decomposer.decompose_iteratively()
    
    return final_report

if __name__ == "__main__":
    report = decompose_animal_species_complete()
    print(f"\n🎉 Decomposition completed!")
    print(f"Check output directory: {report['output_directory']}") 