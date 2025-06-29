"""
BCNF (Boyce-Codd Normal Form) analysis using HyFD results.
This module implements primary key and violating FD scoring algorithms 
based on the Normalize paper approach.
"""

import json
import pandas as pd
from typing import List, Tuple, Dict, Sequence, Union, Set
from collections import namedtuple
import itertools

# Meta information for scoring calculations
Meta = namedtuple(
    "Meta",
    [
        "attr_order",     # List[str] – df.columns order
        "n_rows",         # int         – number of rows
        "value_lens",     # Dict[str, int] – max string length per column
        "n_unique",       # Dict[str, int] – unique count per column
    ]
)

class BCNF:
    """
    BCNF analyzer that processes HyFD results to identify primary keys 
    and violating functional dependencies for database normalization.
    """
    
    def __init__(self, df: pd.DataFrame, fds: List[List[List[int]]] = None, 
                 fd_file: str = None, column_names: List[str] = None):
        """
        Initialize BCNF analyzer.
        
        Args:
            df: pandas DataFrame containing the data
            fds: List of functional dependencies in format [[[lhs], [rhs]], ...]
            fd_file: Path to JSON file containing FD results from HyFD
            column_names: Optional list of column names to override df.columns
        """
        self.df = df
        self.column_names = column_names if column_names else list(df.columns)
        self.meta = self._build_meta(df)
        
        # Load FDs from file or use provided FDs
        if fd_file:
            self.fds = self._load_fds_from_file(fd_file)
        elif fds:
            self.fds = fds
        else:
            raise ValueError("Either fds list or fd_file must be provided")
        
        # Convert index-based FDs to column name-based FDs
        self.named_fds = self._convert_fds_to_names()
        
        # Analysis results
        self.candidate_keys = []
        self.violating_fds = []
        self.pk_scores = {}
        self.vfd_scores = {}
        self.best_primary_key = None
        self.best_violating_fd = None
        
    def _build_meta(self, df: pd.DataFrame) -> Meta:
        """Build metadata for scoring calculations."""
        attr_order = list(df.columns)
        n_rows = len(df)
        value_lens = {c: int(df[c].astype(str).map(len).max()) for c in df}
        n_unique = {c: df[c].nunique(dropna=False) for c in df}
        return Meta(attr_order, n_rows, value_lens, n_unique)
    
    def _load_fds_from_file(self, fd_file: str) -> List[List[List[int]]]:
        """Load functional dependencies from JSON file."""
        with open(fd_file, 'r') as f:
            return json.load(f)
    
    def _convert_fds_to_names(self) -> List[Tuple[List[str], List[str]]]:
        """Convert index-based FDs to column name-based FDs."""
        named_fds = []
        for fd in self.fds:
            lhs_indices, rhs_indices = fd[0], fd[1]
            lhs_names = [self.column_names[i] for i in lhs_indices]
            rhs_names = [self.column_names[i] for i in rhs_indices]
            named_fds.append((lhs_names, rhs_names))
        return named_fds
    
    def _is_superkey(self, attrs: Set[str]) -> bool:
        """Check if a set of attributes is a superkey."""
        # An attribute set is a superkey if it functionally determines all other attributes
        determined_attrs = set(attrs)
        
        # Keep applying FDs until no new attributes are determined
        changed = True
        while changed:
            changed = False
            for lhs, rhs in self.named_fds:
                if set(lhs).issubset(determined_attrs):
                    old_size = len(determined_attrs)
                    determined_attrs.update(rhs)
                    if len(determined_attrs) > old_size:
                        changed = True
        
        # Check if all attributes are determined
        return len(determined_attrs) == len(self.column_names)
    
    def _find_candidate_keys(self) -> List[List[str]]:
        """Find all candidate keys using FD analysis."""
        candidate_keys = []
        all_attrs = set(self.column_names)
        
        # Start with single attributes and work up
        for size in range(1, len(self.column_names) + 1):
            for attr_combo in itertools.combinations(self.column_names, size):
                attr_set = set(attr_combo)
                if self._is_superkey(attr_set):
                    # Check if it's minimal (no proper subset is also a superkey)
                    is_minimal = True
                    for existing_key in candidate_keys:
                        if set(existing_key).issubset(attr_set):
                            is_minimal = False
                            break
                    
                    if is_minimal:
                        # Remove any existing keys that are supersets
                        candidate_keys = [k for k in candidate_keys if not attr_set.issubset(set(k))]
                        candidate_keys.append(list(attr_combo))
        
        return candidate_keys
    
    def _find_violating_fds(self) -> List[Tuple[List[str], List[str]]]:
        """Find FDs that violate BCNF (LHS is not a superkey)."""
        violating_fds = []
        
        for lhs, rhs in self.named_fds:
            if not self._is_superkey(set(lhs)):
                # This FD violates BCNF
                violating_fds.append((lhs, rhs))
        
        return violating_fds
    
    def score_primary_key(self, key_attrs: Sequence[str]) -> float:
        """
        Score a primary key candidate.
        Returns 0-1 score, higher is better.
        """
        len_score = 1.0 / len(key_attrs)
        
        max_len = max(self.meta.value_lens[a] for a in key_attrs)
        val_score = 1.0 / max(1, max_len - 7)
        
        positions = [self.meta.attr_order.index(a) for a in key_attrs]
        left_nonkeys = min(positions)  # number of non-key columns to the left
        between_nonkeys = (max(positions) - min(positions) + 1) - len(key_attrs)
        pos_score = 0.5 * (1/(left_nonkeys+1) + 1/(between_nonkeys+1))
        
        return (len_score + val_score + pos_score) / 3
    
    def score_violating_fd(self, lhs: Sequence[str], rhs: Sequence[str]) -> float:
        """
        Score a violating FD for decomposition priority.
        Returns 0-1 score, higher is better for decomposition.
        """
        lhs_len = len(lhs)
        rhs_len = len(rhs)
        rel_width = self.df.shape[1]
        len_score = 0.5 * (1/lhs_len + rhs_len/(rel_width-2))
        
        max_lhs_val = max(self.meta.value_lens[a] for a in lhs)
        val_score = 1.0 / max(1, max_lhs_val - 7)
        
        lhs_pos = sorted(self.meta.attr_order.index(a) for a in lhs)
        rhs_pos = sorted(self.meta.attr_order.index(a) for a in rhs)
        between_lhs = (lhs_pos[-1] - lhs_pos[0] + 1) - lhs_len
        between_rhs = (rhs_pos[-1] - rhs_pos[0] + 1) - rhs_len
        pos_score = 0.5 * (1/(between_lhs+1) + 1/(between_rhs+1))
        
        def dup_ratio(attrs):
            # 1 - distinct/rows  ⇒  higher means more duplication
            distinct = self.df[list(attrs)].drop_duplicates().shape[0]
            return 1 - (distinct / self.meta.n_rows)
        
        dup_score = 0.5 * (dup_ratio(lhs) + dup_ratio(rhs))
        
        # ---------- Combined score ----------
        return (len_score + val_score + pos_score + dup_score) / 4
    
    def analyze(self) -> Dict:
        """
        Perform complete BCNF analysis.
        Returns dictionary with analysis results.
        """
        # Find candidate keys and violating FDs
        self.candidate_keys = self._find_candidate_keys()
        self.violating_fds = self._find_violating_fds()
        
        # Score primary key candidates
        self.pk_scores = {
            tuple(key): self.score_primary_key(key)
            for key in self.candidate_keys
        }
        
        # Score violating FDs
        self.vfd_scores = {
            (tuple(lhs), tuple(rhs)): self.score_violating_fd(lhs, rhs)
            for lhs, rhs in self.violating_fds
        }
        
        # Find best candidates
        if self.pk_scores:
            self.best_primary_key = max(self.pk_scores, key=self.pk_scores.get)
        
        if self.vfd_scores:
            self.best_violating_fd = max(self.vfd_scores, key=self.vfd_scores.get)
        
        return {
            'candidate_keys': self.candidate_keys,
            'violating_fds': self.violating_fds,
            'pk_scores': self.pk_scores,
            'vfd_scores': self.vfd_scores,
            'best_primary_key': self.best_primary_key,
            'best_violating_fd': self.best_violating_fd,
            'is_bcnf': len(self.violating_fds) == 0
        }
    
    def decompose_table(self) -> List[pd.DataFrame]:
        """
        Decompose table based on the best violating FD.
        Returns list of decomposed tables.
        """
        if not self.best_violating_fd:
            return [self.df]  # Already in BCNF
        
        lhs, rhs = self.best_violating_fd
        lhs_list, rhs_list = list(lhs), list(rhs)
        
        # Create first relation: R1(LHS ∪ RHS)
        r1_cols = lhs_list + rhs_list
        r1 = self.df[r1_cols].drop_duplicates()
        
        # Create second relation: R2(LHS ∪ (R - RHS))
        remaining_cols = [col for col in self.column_names if col not in rhs_list]
        r2 = self.df[remaining_cols].drop_duplicates()
        
        return [r1, r2]
    
    def print_analysis(self):
        """Print formatted analysis results."""
        print("=== BCNF Analysis Results ===\n")
        
        print(f"Dataset: {self.df.shape[0]} rows, {self.df.shape[1]} columns")
        print(f"Columns: {', '.join(self.column_names)}\n")
        
        print(f"Found {len(self.fds)} functional dependencies")
        print(f"Found {len(self.candidate_keys)} candidate keys")
        print(f"Found {len(self.violating_fds)} BCNF violations\n")
        
        if self.candidate_keys:
            print("Candidate Keys (with scores):")
            for key in self.candidate_keys:
                score = self.pk_scores[tuple(key)]
                print(f"  {key} → Score: {score:.4f}")
            print()
        
        if self.violating_fds:
            print("BCNF Violating FDs (with scores):")
            for lhs, rhs in self.violating_fds:
                score = self.vfd_scores[(tuple(lhs), tuple(rhs))]
                print(f"  {lhs} → {rhs} → Score: {score:.4f}")
            print()
        
        if self.best_primary_key:
            print(f"Best Primary Key: {list(self.best_primary_key)}")
            print(f"Primary Key Score: {self.pk_scores[self.best_primary_key]:.4f}\n")
        
        if self.best_violating_fd:
            lhs, rhs = self.best_violating_fd
            print(f"Best Violating FD for decomposition: {list(lhs)} → {list(rhs)}")
            print(f"Violating FD Score: {self.vfd_scores[self.best_violating_fd]:.4f}\n")
        
        if len(self.violating_fds) == 0:
            print("✅ Table is already in BCNF!")
        else:
            print("❌ Table violates BCNF and needs decomposition.")


# Example usage and testing
if __name__ == "__main__":
    # Example with sample data
    import numpy as np
    
    # Create sample DataFrame
    data = {
        'StudentID': [1, 2, 3, 4, 5],
        'CourseID': ['CS101', 'CS102', 'CS101', 'CS103', 'CS102'],
        'Instructor': ['Smith', 'Jones', 'Smith', 'Brown', 'Jones'],
        'Grade': ['A', 'B', 'A', 'C', 'A'],
        'Department': ['CS', 'CS', 'CS', 'CS', 'CS']
    }
    df = pd.DataFrame(data)
    
    # Example FDs: CourseID → Instructor, CourseID → Department
    example_fds = [
        [[1], [2]],  # CourseID → Instructor
        [[1], [4]],  # CourseID → Department
        [[0, 1], [3]]  # StudentID, CourseID → Grade
    ]
    
    # Analyze
    bcnf = BCNF(df, fds=example_fds)
    results = bcnf.analyze()
    bcnf.print_analysis() 