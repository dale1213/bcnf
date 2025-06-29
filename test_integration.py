"""
Integration test for HyFD + BCNF analysis workflow.
This script demonstrates the complete pipeline from FD discovery to BCNF normalization.
"""

import pandas as pd
import numpy as np
import json
import os
from bcnf import BCNF

def create_test_dataset(filename='test_dataset.csv'):
    """Create a test dataset with known BCNF violations."""
    np.random.seed(42)
    
    # Create a dataset representing a university course system
    # Known FDs:
    # - CourseID → CourseName, Department, Credits
    # - InstructorID → InstructorName, Department  
    # - StudentID, CourseID → Grade, Semester
    
    n_records = 100
    
    # Base data
    course_ids = ['CS101', 'CS102', 'CS201', 'MATH101', 'MATH201'] * 20
    course_names = ['Intro to CS', 'Data Structures', 'Algorithms', 'Calculus I', 'Linear Algebra'] * 20
    departments = ['Computer Science', 'Computer Science', 'Computer Science', 'Mathematics', 'Mathematics'] * 20
    credits = [3, 4, 4, 3, 3] * 20
    
    instructor_ids = ['I001', 'I002', 'I003', 'I004', 'I005'] * 20
    instructor_names = ['Dr. Smith', 'Dr. Jones', 'Dr. Brown', 'Dr. Davis', 'Dr. Wilson'] * 20
    
    student_ids = [f'S{i:03d}' for i in range(1, n_records + 1)]
    grades = np.random.choice(['A', 'B', 'C', 'D', 'F'], n_records, p=[0.3, 0.3, 0.25, 0.1, 0.05])
    semesters = np.random.choice(['Fall2023', 'Spring2024'], n_records)
    
    # Create DataFrame
    data = {
        'StudentID': student_ids,
        'CourseID': course_ids,
        'CourseName': course_names,
        'Department': departments,
        'Credits': credits,
        'InstructorID': instructor_ids,
        'InstructorName': instructor_names,
        'Grade': grades,
        'Semester': semesters
    }
    
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    print(f"Created test dataset: {filename}")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    return df

def simulate_hyfd_results():
    """Simulate HyFD results for the test dataset."""
    # These are the expected FDs for our test dataset:
    # CourseID → CourseName, Department, Credits
    # InstructorID → InstructorName, Department
    # StudentID, CourseID → Grade, Semester
    
    simulated_fds = [
        # CourseID → CourseName
        [[1], [2]],
        # CourseID → Department  
        [[1], [3]],
        # CourseID → Credits
        [[1], [4]],
        # InstructorID → InstructorName
        [[5], [6]],
        # InstructorID → Department (partial dependency)
        [[5], [3]],
        # StudentID, CourseID → Grade
        [[0, 1], [7]],
        # StudentID, CourseID → Semester
        [[0, 1], [8]],
        # Additional composite dependencies
        [[1, 5], [2, 6, 3]],  # CourseID, InstructorID → CourseName, InstructorName, Department
    ]
    
    # Save as JSON file
    with open('simulated_hyfd_results.json', 'w') as f:
        json.dump(simulated_fds, f)
    
    print(f"Created simulated HyFD results with {len(simulated_fds)} functional dependencies")
    return simulated_fds

def run_integration_test():
    """Run the complete integration test."""
    print("=" * 60)
    print("HyFD + BCNF Integration Test")
    print("=" * 60)
    
    # Step 1: Create test dataset
    print("\n1. Creating test dataset...")
    df = create_test_dataset()
    print(df.head())
    
    # Step 2: Simulate HyFD results
    print("\n2. Simulating HyFD functional dependency discovery...")
    fds = simulate_hyfd_results()
    
    # Step 3: BCNF Analysis
    print("\n3. Performing BCNF analysis...")
    bcnf = BCNF(df, fds=fds)
    results = bcnf.analyze()
    
    # Step 4: Display results
    print("\n4. Analysis Results:")
    bcnf.print_analysis()
    
    # Step 5: Demonstrate scoring details
    print("\n5. Detailed Scoring Analysis:")
    
    if bcnf.best_primary_key:
        key = list(bcnf.best_primary_key)
        score = bcnf.pk_scores[bcnf.best_primary_key]
        print(f"\nBest Primary Key Analysis:")
        print(f"Key: {key}")
        print(f"Score: {score:.4f}")
        
        # Show score components
        len_score = 1.0 / len(key)
        max_len = max(bcnf.meta.value_lens[a] for a in key)
        val_score = 1.0 / max(1, max_len - 7)
        positions = [bcnf.meta.attr_order.index(a) for a in key]
        left_nonkeys = min(positions)
        between_nonkeys = (max(positions) - min(positions) + 1) - len(key)
        pos_score = 0.5 * (1/(left_nonkeys+1) + 1/(between_nonkeys+1))
        
        print(f"  - Length Component: 1/{len(key)} = {len_score:.4f}")
        print(f"  - Value Length Component: 1/max(1,{max_len}-7) = {val_score:.4f}")
        print(f"  - Position Component: {pos_score:.4f}")
    
    if bcnf.best_violating_fd:
        lhs, rhs = bcnf.best_violating_fd
        score = bcnf.vfd_scores[bcnf.best_violating_fd]
        print(f"\nBest Violating FD Analysis:")
        print(f"FD: {list(lhs)} → {list(rhs)}")
        print(f"Score: {score:.4f}")
        
        # Show why it's good for decomposition
        lhs_distinct = df[list(lhs)].drop_duplicates().shape[0]
        rhs_distinct = df[list(rhs)].drop_duplicates().shape[0]
        total_rows = len(df)
        
        print(f"  - LHS cardinality: {lhs_distinct}/{total_rows} ({lhs_distinct/total_rows:.2%})")
        print(f"  - RHS cardinality: {rhs_distinct}/{total_rows} ({rhs_distinct/total_rows:.2%})")
        print(f"  - Potential space savings: {(1 - lhs_distinct/total_rows):.2%}")
    
    # Step 6: Table Decomposition
    if not results['is_bcnf']:
        print("\n6. Table Decomposition:")
        decomposed_tables = bcnf.decompose_table()
        
        for i, table in enumerate(decomposed_tables):
            print(f"\nTable R{i+1}:")
            print(f"  Columns: {list(table.columns)}")
            print(f"  Shape: {table.shape}")
            print(f"  Sample data:")
            print(table.head(3).to_string(index=False))
            
            # Check if this table is now in BCNF
            try:
                # Filter FDs that are relevant to this table
                table_cols = set(table.columns)
                relevant_fds = []
                for fd in fds:
                    lhs_names = [bcnf.column_names[idx] for idx in fd[0]]
                    rhs_names = [bcnf.column_names[idx] for idx in fd[1]]
                    if set(lhs_names + rhs_names).issubset(table_cols):
                        # Convert back to indices for this table
                        table_col_to_idx = {col: i for i, col in enumerate(table.columns)}
                        lhs_idx = [table_col_to_idx[col] for col in lhs_names if col in table_col_to_idx]
                        rhs_idx = [table_col_to_idx[col] for col in rhs_names if col in table_col_to_idx]
                        if lhs_idx and rhs_idx:
                            relevant_fds.append([lhs_idx, rhs_idx])
                
                if relevant_fds:
                    table_bcnf = BCNF(table, fds=relevant_fds)
                    table_results = table_bcnf.analyze()
                    print(f"  BCNF Status: {'✅ Normalized' if table_results['is_bcnf'] else '❌ Still needs work'}")
                    print(f"  Violations: {len(table_results['violating_fds'])}")
                else:
                    print(f"  BCNF Status: ✅ No relevant FDs (trivially normalized)")
                    
            except Exception as e:
                print(f"  BCNF Status: ⚠️  Could not analyze (error: {str(e)})")
    
    # Step 7: Summary
    print("\n7. Summary:")
    print(f"  Original table: {df.shape}")
    print(f"  Functional dependencies found: {len(fds)}")
    print(f"  Candidate keys: {len(results['candidate_keys'])}")
    print(f"  BCNF violations: {len(results['violating_fds'])}")
    print(f"  Normalization needed: {'Yes' if not results['is_bcnf'] else 'No'}")
    
    if not results['is_bcnf']:
        decomposed = bcnf.decompose_table()
        total_attrs = sum(len(t.columns) for t in decomposed)
        original_attrs = len(df.columns)
        print(f"  After decomposition: {len(decomposed)} tables")
        print(f"  Total attributes after: {total_attrs} (original: {original_attrs})")
    
    # Cleanup
    print("\n8. Cleaning up test files...")
    for file in ['test_dataset.csv', 'simulated_hyfd_results.json']:
        if os.path.exists(file):
            os.remove(file)
            print(f"  Removed {file}")
    
    print("\n" + "=" * 60)
    print("Integration test completed successfully!")
    print("=" * 60)

if __name__ == "__main__":
    run_integration_test() 