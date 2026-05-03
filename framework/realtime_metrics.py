# -*- coding: utf-8 -*-
"""
Real-time Metrics Calculation and Summary Module

Provides comprehensive metrics calculation based on results_summary/analyze_results.py,
integrated into the run.py execution flow for real-time updates.

Metrics include:
- Pass@K (Overall/Memory/Standard)
- IRR (Information Retention Rate)
- FRR (Failure Recovery Rate)
- MTPR (Memory Task Performance Ratio)
- Difficulty-based grouping
- App count-based grouping
- Token statistics
- BadCase statistics
"""

import os
import re
import json
import warnings
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from collections import defaultdict

import pandas as pd
import numpy as np

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)


# ============================================================================
# File Paths
# ============================================================================
METRICS_SUMMARY_FILE = "metrics_summary.json"
METRICS_SUMMARY_CSV = "metrics_summary.csv"
METRICS_HISTORY_FILE = "metrics_history.jsonl"
# Leaderboard result files are now saved with agent-specific names
# e.g., "qwen3vl.json", "agent-s2.json" (matching data/agents/*.json format)


# ============================================================================
# Helper Functions
# ============================================================================
def _get_numeric_value(df: pd.DataFrame, col_name: str, idx: int, default: float = 0) -> float:
    """Safe get numeric value from DataFrame."""
    if col_name and col_name in df.columns:
        val = df.at[idx, col_name]
        if pd.notna(val):
            try:
                return float(val)
            except (ValueError, TypeError):
                pass
    return default


def _extract_agent_name(df: pd.DataFrame) -> str:
    """Extract agent name from DataFrame columns."""
    for col in df.columns:
        match = re.match(r"^(.+?)_attempt_1_completion$", col)
        if match:
            return match.group(1)
        match = re.match(r"^(.+?)_successful_attempts$", col)
        if match:
            return match.group(1)
    return "Unknown"


# ============================================================================
# Core Metrics Calculation
# ============================================================================
class MetricsCalculator:
    """
    Calculates all metrics from results.csv file.
    
    This class consolidates the logic from results_summary/analyze_results.py
    for real-time metric calculation during run.py execution.
    """
    
    def __init__(self, csv_path: str, max_attempts: int = 3):
        """
        Initialize the calculator.
        
        Args:
            csv_path: Path to results.csv
            max_attempts: Maximum number of attempts per task
        """
        self.csv_path = csv_path
        self.max_attempts = max_attempts
        self.df = None
        self.agent_name = "Unknown"
        self.session_name = ""
        
    def load_data(self) -> bool:
        """Load and preprocess data from CSV."""
        if not os.path.exists(self.csv_path):
            return False
            
        try:
            self.df = pd.read_csv(self.csv_path)
        except Exception as e:
            print(f"  Warning: Failed to load {self.csv_path}: {e}")
            return False
            
        self.agent_name = _extract_agent_name(self.df)
        self.session_name = os.path.basename(os.path.dirname(self.csv_path))
        
        # Preprocess requires_ui_memory
        if "requires_ui_memory" in self.df.columns:
            self.df["requires_ui_memory"] = (
                self.df["requires_ui_memory"].astype(str).str.strip().str.upper() == "Y"
            )
        else:
            self.df["requires_ui_memory"] = False
            
        # Extract per-attempt fields
        self._extract_attempt_fields()
        
        return True
        
    def _extract_attempt_fields(self):
        """Extract key fields for each attempt."""
        for i in range(1, self.max_attempts + 1):
            # Evaluation result - try multiple possible column patterns
            eval_col = None
            eval_patterns = [
                f"{self.agent_name}_direct_with_action_attempt_{i}_evaluation",
                f"{self.agent_name}_attempt_{i}_evaluation",
            ]
            for pattern in eval_patterns:
                if pattern in self.df.columns:
                    eval_col = pattern
                    break
                    
            if eval_col:
                self.df[f"success_att_{i}"] = (
                    self.df[eval_col].astype(str).str.strip().str.upper() == "S"
                ).astype(int)
            else:
                self.df[f"success_att_{i}"] = 0
                
            # Steps
            steps_col = f"{self.agent_name}_attempt_{i}_total_steps"
            if steps_col in self.df.columns:
                self.df[f"steps_att_{i}"] = pd.to_numeric(self.df[steps_col], errors="coerce").fillna(0)
            else:
                self.df[f"steps_att_{i}"] = 0
            
            # Time
            time_col = f"{self.agent_name}_attempt_{i}_total_time"
            if time_col in self.df.columns:
                self.df[f"time_att_{i}"] = pd.to_numeric(self.df[time_col], errors="coerce").fillna(0)
            else:
                self.df[f"time_att_{i}"] = 0
            
            # Tokens - prefer total columns, fall back to avg * steps
            total_prompt_col = f"{self.agent_name}_attempt_{i}_total_prompt_tokens"
            total_completion_col = f"{self.agent_name}_attempt_{i}_total_completion_tokens"
            avg_prompt_col = f"{self.agent_name}_attempt_{i}_avg_prompt_tokens"
            avg_completion_col = f"{self.agent_name}_attempt_{i}_avg_completion_tokens"
            steps_col_name = f"{self.agent_name}_attempt_{i}_total_steps"

            if total_prompt_col in self.df.columns:
                self.df[f"total_prompt_tokens_att_{i}"] = pd.to_numeric(self.df[total_prompt_col], errors="coerce").fillna(0)
            elif avg_prompt_col in self.df.columns and steps_col_name in self.df.columns:
                avg_vals = pd.to_numeric(self.df[avg_prompt_col], errors="coerce").fillna(0)
                steps_vals = pd.to_numeric(self.df[steps_col_name], errors="coerce").fillna(0)
                self.df[f"total_prompt_tokens_att_{i}"] = avg_vals * steps_vals
            else:
                self.df[f"total_prompt_tokens_att_{i}"] = 0

            if total_completion_col in self.df.columns:
                self.df[f"total_completion_tokens_att_{i}"] = pd.to_numeric(self.df[total_completion_col], errors="coerce").fillna(0)
            elif avg_completion_col in self.df.columns and steps_col_name in self.df.columns:
                avg_vals = pd.to_numeric(self.df[avg_completion_col], errors="coerce").fillna(0)
                steps_vals = pd.to_numeric(self.df[steps_col_name], errors="coerce").fillna(0)
                self.df[f"total_completion_tokens_att_{i}"] = avg_vals * steps_vals
            else:
                self.df[f"total_completion_tokens_att_{i}"] = 0

            if avg_prompt_col in self.df.columns:
                self.df[f"prompt_tokens_att_{i}"] = pd.to_numeric(self.df[avg_prompt_col], errors="coerce").fillna(0)
            else:
                self.df[f"prompt_tokens_att_{i}"] = 0
            if avg_completion_col in self.df.columns:
                self.df[f"completion_tokens_att_{i}"] = pd.to_numeric(self.df[avg_completion_col], errors="coerce").fillna(0)
            else:
                self.df[f"completion_tokens_att_{i}"] = 0
            
            # IRR - try multiple possible column patterns
            irr_col = None
            irr_patterns = [
                f"{self.agent_name}_direct_with_action_attempt_{i}_irr_percentage",
                f"{self.agent_name}_attempt_{i}_irr_percentage",
            ]
            for pattern in irr_patterns:
                if pattern in self.df.columns:
                    irr_col = pattern
                    break
            
            if irr_col:
                self.df[f"irr_att_{i}"] = pd.to_numeric(self.df[irr_col], errors="coerce").fillna(0)
            else:
                self.df[f"irr_att_{i}"] = 0
            
            # BadCase
            badcase_col = f"{self.agent_name}_attempt_{i}_badcase_category"
            if badcase_col in self.df.columns:
                self.df[f"badcase_att_{i}"] = self.df[badcase_col].fillna("")
            else:
                self.df[f"badcase_att_{i}"] = ""
                
    def calculate_all_metrics(self) -> Dict[str, Any]:
        """Calculate all metrics and return as dictionary."""
        if self.df is None or self.df.empty:
            return self._empty_metrics()
            
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "session": self.session_name,
            "agent": self.agent_name,
            "max_attempts": self.max_attempts,
        }
        
        # Basic counts
        metrics.update(self._calculate_task_counts())
        
        # Pass@K metrics
        metrics.update(self._calculate_pass_at_k())
        
        # Memory-specific metrics (IRR, MTPR)
        metrics.update(self._calculate_memory_metrics())
        
        # FRR (Failure Recovery Rate)
        metrics.update(self._calculate_frr())
        
        # Efficiency metrics
        metrics.update(self._calculate_efficiency())
        
        # Difficulty grouping
        metrics.update(self._calculate_difficulty_metrics())
        
        # App count grouping
        metrics.update(self._calculate_app_count_metrics())
        
        # BadCase statistics
        metrics.update(self._calculate_badcase_stats())
        
        # Token statistics
        metrics.update(self._calculate_token_stats())
        
        return metrics
        
    def _empty_metrics(self) -> Dict[str, Any]:
        """Return empty metrics structure."""
        return {
            "timestamp": datetime.now().isoformat(),
            "session": self.session_name,
            "agent": self.agent_name,
            "total_tasks": 0,
            "memory_tasks": 0,
            "standard_tasks": 0,
            "executed_tasks": 0,
            "evaluated_tasks": 0,
        }
        
    def _calculate_task_counts(self) -> Dict[str, Any]:
        """Calculate basic task counts."""
        total = len(self.df)
        memory_df = self.df[self.df["requires_ui_memory"]]
        standard_df = self.df[~self.df["requires_ui_memory"]]
        
        # Count executed (has steps > 0 for attempt 1)
        executed = (self.df["steps_att_1"] > 0).sum()
        
        # Count evaluated - only count tasks with actual evaluation results (S/F/E)
        # CSV initializes evaluation column with "N" (not evaluated), so we must exclude it
        evaluated = 0
        eval_patterns = [
            f"{self.agent_name}_direct_with_action_attempt_1_evaluation",
            f"{self.agent_name}_attempt_1_evaluation",
        ]
        for eval_col in eval_patterns:
            if eval_col in self.df.columns:
                # Count only valid evaluation results: S (Success), F (Failure), E (Error)
                valid_results = self.df[eval_col].astype(str).str.strip().str.upper()
                evaluated = valid_results.isin(['S', 'F', 'E']).sum()
                break
            
        return {
            "total_tasks": total,
            "memory_tasks": len(memory_df),
            "standard_tasks": len(standard_df),
            "executed_tasks": int(executed),
            "evaluated_tasks": int(evaluated),
        }
        
    def _calculate_pass_at_k(self) -> Dict[str, Any]:
        """Calculate Pass@K rates for overall, memory, and standard tasks."""
        total = len(self.df)
        
        # First, create pass_at columns on the full DataFrame
        self.df["pass_at_1"] = self.df["success_att_1"]
        
        if self.max_attempts >= 2 and "success_att_2" in self.df.columns:
            self.df["pass_at_2"] = self.df[["success_att_1", "success_att_2"]].max(axis=1)
        
        if self.max_attempts >= 3:
            success_cols = [f"success_att_{i}" for i in range(1, self.max_attempts + 1) if f"success_att_{i}" in self.df.columns]
            if success_cols:
                self.df[f"pass_at_{self.max_attempts}"] = self.df[success_cols].max(axis=1)
        
        # Now create filtered DataFrames after columns exist
        executed_df = self.df[self.df["steps_att_1"] > 0]
        memory_df = self.df[self.df["requires_ui_memory"]]
        standard_df = self.df[~self.df["requires_ui_memory"]]
        executed_memory_df = executed_df[executed_df["requires_ui_memory"]]
        executed_standard_df = executed_df[~executed_df["requires_ui_memory"]]
        
        results = {}
        
        for k in range(1, self.max_attempts + 1):
            pass_col = f"pass_at_{k}"
            
            if pass_col not in self.df.columns:
                continue
            
            # Overall - use executed tasks as denominator
            count_all = executed_df[pass_col].sum() if not executed_df.empty else 0
            executed_count = len(executed_df)
            results[f"pass_at_{k}_count"] = int(count_all)
            results[f"pass_at_{k}_rate"] = count_all / executed_count * 100 if executed_count > 0 else 0
            results[f"pass_at_{k}_total"] = executed_count
            
            # Memory - use executed memory tasks as denominator
            count_mem = executed_memory_df[pass_col].sum() if not executed_memory_df.empty else 0
            executed_mem_count = len(executed_memory_df)
            results[f"pass_at_{k}_memory_count"] = int(count_mem)
            results[f"pass_at_{k}_memory_rate"] = (
                count_mem / executed_mem_count * 100 if executed_mem_count > 0 else 0
            )
            results[f"pass_at_{k}_memory_total"] = executed_mem_count
            
            # Standard - use executed standard tasks as denominator
            count_std = executed_standard_df[pass_col].sum() if not executed_standard_df.empty else 0
            executed_std_count = len(executed_standard_df)
            results[f"pass_at_{k}_standard_count"] = int(count_std)
            results[f"pass_at_{k}_standard_rate"] = (
                count_std / executed_std_count * 100 if executed_std_count > 0 else 0
            )
            results[f"pass_at_{k}_standard_total"] = executed_std_count
            
        return results
        
    def _calculate_memory_metrics(self) -> Dict[str, Any]:
        """Calculate IRR and MTPR."""
        memory_df = self.df[self.df["requires_ui_memory"]]
        standard_df = self.df[~self.df["requires_ui_memory"]]
        
        # Average IRR (only for memory tasks, attempt 1)
        # Include all tasks with IRR values (including IRR=0, which means no memory utilization)
        if not memory_df.empty:
            irr_values = memory_df["irr_att_1"]
            valid_irr = irr_values[irr_values.notna()]
            avg_irr = valid_irr.mean() if len(valid_irr) > 0 else 0
            irr_count = len(valid_irr)
        else:
            avg_irr = 0
            irr_count = 0
            
        # MTPR (Memory Task Performance Ratio)
        sr_memory_1 = (
            memory_df["success_att_1"].mean() * 100 if not memory_df.empty else 0
        )
        sr_standard_1 = (
            standard_df["success_att_1"].mean() * 100 if not standard_df.empty else 0
        )
        mtpr = sr_memory_1 / sr_standard_1 if sr_standard_1 > 0 else 0
        
        return {
            "avg_irr": avg_irr,
            "irr_count": irr_count,
            "sr_memory_at_1": sr_memory_1,
            "sr_standard_at_1": sr_standard_1,
            "mtpr": mtpr,
        }
        
    def _calculate_frr(self) -> Dict[str, Any]:
        """Calculate Failure Recovery Rate (FRR)."""
        # FRR = ((w_2 * R_2) + (w_3 * R_3) + ...) / N_failed_1 * 100
        # w_k = 1.0 / (2 ** (k - 2)) for k >= 2
        
        failed_on_1 = self.df[self.df["success_att_1"] == 0]
        n_failed_1 = len(failed_on_1)
        
        if n_failed_1 == 0:
            return {
                "frr": 0.0,
                "n_failed_1": 0,
                "recovery_at_2": 0,
                "recovery_at_3": 0,
            }
            
        # Recovery counts
        recovery_counts = {}
        weighted_sum = 0
        
        for k in range(2, self.max_attempts + 1):
            # Tasks that failed on all attempts 1 to k-1 but succeeded on k
            prev_failed_mask = failed_on_1[[f"success_att_{i}" for i in range(1, k)]].max(axis=1) == 0
            recovered_on_k = failed_on_1[prev_failed_mask][f"success_att_{k}"].sum()
            
            recovery_counts[k] = int(recovered_on_k)
            weight = 1.0 / (2 ** (k - 2))
            weighted_sum += weight * recovered_on_k
            
        frr = (weighted_sum / n_failed_1 * 100) if n_failed_1 > 0 else 0
        
        result = {
            "frr": frr,
            "n_failed_1": n_failed_1,
        }
        for k, count in recovery_counts.items():
            result[f"recovery_at_{k}"] = count
            
        return result
        
    def _calculate_efficiency(self) -> Dict[str, Any]:
        """Calculate efficiency metrics (step ratio, time/step, cost/step)."""
        results = {}
        
        # Cost calculation: Gemini-2.5-pro pricing ($1.25/M in, $10/M out)
        for k in range(1, self.max_attempts + 1):
            p1_df = self.df[self.df[f"pass_at_{k}"] == 1]
            
            if not p1_df.empty and "golden_steps" in p1_df.columns:
                valid_mask = p1_df["golden_steps"] > 0
                if valid_mask.sum() > 0:
                    step_ratios = (
                        p1_df.loc[valid_mask, f"steps_att_1"]
                        / p1_df.loc[valid_mask, "golden_steps"]
                    )
                    results[f"step_ratio_at_{k}"] = step_ratios.mean()
                else:
                    results[f"step_ratio_at_{k}"] = 0
            else:
                results[f"step_ratio_at_{k}"] = 0
                
            # Time and cost per step
            total_time = 0
            total_cost = 0
            total_steps = 0
            
            for _, row in self.df.iterrows():
                for att in range(1, k + 1):
                    steps = row.get(f"steps_att_{att}", 0)
                    if steps > 0:
                        total_steps += steps
                        total_time += row.get(f"time_att_{att}", 0)
                        prompt_tokens = row.get(f"prompt_tokens_att_{att}", 0)
                        completion_tokens = row.get(f"completion_tokens_att_{att}", 0)
                        total_cost += (
                            (prompt_tokens * steps / 1_000_000) * 1.25
                            + (completion_tokens * steps / 1_000_000) * 10
                        )
                        
            results[f"time_per_step_at_{k}"] = total_time / total_steps if total_steps > 0 else 0
            results[f"cost_per_step_at_{k}"] = total_cost / total_steps if total_steps > 0 else 0
            
        return results
        
    def _calculate_difficulty_metrics(self) -> Dict[str, Any]:
        """Calculate metrics grouped by task difficulty, including IRR."""
        results = {}
        
        if "task_difficulty" not in self.df.columns:
            return results
            
        for diff in sorted(self.df["task_difficulty"].dropna().unique()):
            diff_df = self.df[self.df["task_difficulty"] == diff]
            if diff_df.empty:
                continue
                
            diff_key = f"diff_{int(diff)}" if isinstance(diff, (int, float)) else f"diff_{diff}"
            
            results[f"count_{diff_key}"] = len(diff_df)
            for k in range(1, self.max_attempts + 1):
                rate = diff_df[f"pass_at_{k}"].mean() * 100
                results[f"pass_at_{k}_{diff_key}"] = rate
            
            # Calculate IRR for this difficulty level (only for memory tasks)
            # Include all tasks with IRR values (including IRR=0)
            memory_diff_df = diff_df[diff_df["requires_ui_memory"]]
            if not memory_diff_df.empty:
                irr_values = memory_diff_df["irr_att_1"]
                valid_irr = irr_values[irr_values.notna()]
                avg_irr = valid_irr.mean() if len(valid_irr) > 0 else 0
                results[f"irr_{diff_key}"] = avg_irr
            else:
                results[f"irr_{diff_key}"] = 0
                
        return results
        
    def _calculate_app_count_metrics(self) -> Dict[str, Any]:
        """Calculate metrics grouped by number of apps, including IRR."""
        results = {}
        
        if "num_apps" not in self.df.columns:
            return results
            
        for num_apps in sorted(self.df["num_apps"].dropna().unique()):
            apps_df = self.df[self.df["num_apps"] == num_apps]
            if apps_df.empty:
                continue
                
            apps_key = f"apps_{int(num_apps)}"
            
            results[f"count_{apps_key}"] = len(apps_df)
            for k in range(1, self.max_attempts + 1):
                rate = apps_df[f"pass_at_{k}"].mean() * 100
                results[f"pass_at_{k}_{apps_key}"] = rate
            
            # Calculate IRR for this app count (only for memory tasks)
            # Include all tasks with IRR values (including IRR=0)
            memory_apps_df = apps_df[apps_df["requires_ui_memory"]]
            if not memory_apps_df.empty:
                irr_values = memory_apps_df["irr_att_1"]
                valid_irr = irr_values[irr_values.notna()]
                avg_irr = valid_irr.mean() if len(valid_irr) > 0 else 0
                results[f"irr_{apps_key}"] = avg_irr
            else:
                results[f"irr_{apps_key}"] = 0
                
        return results
        
    def _calculate_badcase_stats(self) -> Dict[str, Any]:
        """Calculate BadCase category statistics."""
        results = {}
        
        for att in range(1, self.max_attempts + 1):
            failed_df = self.df[self.df[f"success_att_{att}"] == 0].copy()
            if failed_df.empty:
                continue
                
            final_categories = []
            for _, row in failed_df.iterrows():
                # Check IRR for partial_memory_hallucination
                irr_value = row.get(f"irr_att_{att}", 0)
                if isinstance(irr_value, (int, float)) and 0 < irr_value < 100:
                    final_categories.append("partial_memory_hallucination")
                    continue
                    
                badcase = row.get(f"badcase_att_{att}", "")
                if pd.notna(badcase) and str(badcase).strip():
                    final_categories.append(str(badcase).strip())
                    
            if final_categories:
                category_counts = pd.Series(final_categories).value_counts().to_dict()
                for cat, cnt in category_counts.items():
                    results[f"badcase_att{att}_{cat}"] = cnt
                    
        return results
        
    def _calculate_token_stats(self) -> Dict[str, Any]:
        """Calculate token usage statistics."""
        results = {}
        
        for att in range(1, self.max_attempts + 1):
            total_prompt_col = f"total_prompt_tokens_att_{att}"
            total_completion_col = f"total_completion_tokens_att_{att}"
            steps_col = f"steps_att_{att}"
            
            if total_prompt_col in self.df.columns and steps_col in self.df.columns:
                total_prompt = self.df[total_prompt_col].sum()
                total_completion = self.df[total_completion_col].sum()
                total_steps = self.df[steps_col].sum()
                
                results[f"token_att{att}_total_prompt"] = int(total_prompt)
                results[f"token_att{att}_total_completion"] = int(total_completion)
                results[f"token_att{att}_total_steps"] = int(total_steps)
                results[f"token_att{att}_avg_prompt"] = (
                    int(total_prompt / total_steps) if total_steps > 0 else 0
                )
                results[f"token_att{att}_avg_completion"] = (
                    int(total_completion / total_steps) if total_steps > 0 else 0
                )
                
        return results


# ============================================================================
# Metrics Saver
# ============================================================================
class MetricsSaver:
    """Handles saving metrics to various file formats."""
    
    def __init__(self, output_dir: str):
        """
        Initialize the saver.
        
        Args:
            output_dir: Base output directory (session directory)
        """
        self.output_dir = output_dir
        
    def save_json(self, metrics: Dict[str, Any]) -> str:
        """Save metrics to JSON file."""
        path = os.path.join(self.output_dir, METRICS_SUMMARY_FILE)
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2, ensure_ascii=False)
        except IOError as e:
            print(f"Warning: Failed to save metrics JSON: {e}")
        return path
        
    def save_csv(self, metrics: Dict[str, Any]) -> str:
        """Save key metrics to CSV file (single row, wide format)."""
        path = os.path.join(self.output_dir, METRICS_SUMMARY_CSV)
        try:
            df = pd.DataFrame([metrics])
            df.to_csv(path, index=False, float_format="%.4f")
        except Exception as e:
            print(f"Warning: Failed to save metrics CSV: {e}")
        return path
        
    def append_history(self, metrics: Dict[str, Any]) -> str:
        """Append metrics to history file (JSONL format)."""
        path = os.path.join(self.output_dir, METRICS_HISTORY_FILE)
        try:
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(metrics, ensure_ascii=False) + "\n")
        except IOError as e:
            print(f"Warning: Failed to append metrics history: {e}")
        return path


# ============================================================================
# Metrics Printer
# ============================================================================
class MetricsPrinter:
    """Handles printing metrics to console in a formatted way."""
    
    @staticmethod
    def print_summary(metrics: Dict[str, Any], trigger: str = ""):
        """Print comprehensive metrics summary."""
        print(f"\n{'=' * 100}")
        print(f"[*] METRICS SUMMARY [{trigger}]")
        print(f"{'=' * 100}")
        
        MetricsPrinter._print_progress(metrics)
        MetricsPrinter._print_pass_at_k(metrics)
        MetricsPrinter._print_core_metrics(metrics)
        MetricsPrinter._print_grouping_metrics(metrics)
        MetricsPrinter._print_token_stats(metrics)
        
        print(f"{'=' * 100}\n")
        
    @staticmethod
    def _print_progress(metrics: Dict[str, Any]):
        """Print progress section."""
        total = metrics.get("total_tasks", 0)
        memory = metrics.get("memory_tasks", 0)
        standard = metrics.get("standard_tasks", 0)
        executed = metrics.get("executed_tasks", 0)
        evaluated = metrics.get("evaluated_tasks", 0)
        
        exec_pct = executed / total * 100 if total > 0 else 0
        eval_pct = evaluated / total * 100 if total > 0 else 0
        
        print(f"\n[>] Task Progress:")
        print(f"    Total: {total} tasks (Memory: {memory}, Standard: {standard})")
        print(f"    Executed: {executed}/{total} ({exec_pct:.1f}%)")
        print(f"    Evaluated: {evaluated}/{total} ({eval_pct:.1f}%)")
        
    @staticmethod
    def _print_pass_at_k(metrics: Dict[str, Any]):
        """Print Pass@K section."""
        print(f"\n[>] Pass@K Results:")
        
        # Overall
        pass_str = " | ".join([
            f"@{k}: {metrics.get(f'pass_at_{k}_count', 0)}/{metrics.get(f'pass_at_{k}_total', 0)} ({metrics.get(f'pass_at_{k}_rate', 0):.1f}%)"
            for k in range(1, 4)
        ])
        print(f"    Overall:  {pass_str}")
        
        # Memory
        mem_total = metrics.get("pass_at_1_memory_total", 0)
        if mem_total > 0:
            mem_str = " | ".join([
                f"@{k}: {metrics.get(f'pass_at_{k}_memory_count', 0)}/{metrics.get(f'pass_at_{k}_memory_total', 0)} ({metrics.get(f'pass_at_{k}_memory_rate', 0):.1f}%)"
                for k in range(1, 4)
            ])
            print(f"    Memory:   {mem_str}")
            
        # Standard
        std_total = metrics.get("pass_at_1_standard_total", 0)
        if std_total > 0:
            std_str = " | ".join([
                f"@{k}: {metrics.get(f'pass_at_{k}_standard_count', 0)}/{metrics.get(f'pass_at_{k}_standard_total', 0)} ({metrics.get(f'pass_at_{k}_standard_rate', 0):.1f}%)"
                for k in range(1, 4)
            ])
            print(f"    Standard: {std_str}")
            
    @staticmethod
    def _print_core_metrics(metrics: Dict[str, Any]):
        """Print core metrics section (IRR, FRR, MTPR)."""
        print(f"\n[>] Core Metrics:")
        
        # IRR
        avg_irr = metrics.get("avg_irr", 0)
        irr_count = metrics.get("irr_count", 0)
        memory_tasks = metrics.get("memory_tasks", 0)
        print(f"    IRR: {avg_irr:.1f}% ({irr_count}/{memory_tasks} memory tasks evaluated)")
        
        # FRR
        frr = metrics.get("frr", 0)
        r2 = metrics.get("recovery_at_2", 0)
        r3 = metrics.get("recovery_at_3", 0)
        n_failed = metrics.get("n_failed_1", 0)
        print(f"    FRR: {frr:.1f}% (R2={r2}, R3={r3}, first_failures={n_failed})")
        
        # MTPR
        mtpr = metrics.get("mtpr", 0)
        sr_mem = metrics.get("sr_memory_at_1", 0)
        sr_std = metrics.get("sr_standard_at_1", 0)
        print(f"    MTPR: {mtpr:.3f} (Memory@1={sr_mem:.1f}%, Standard@1={sr_std:.1f}%)")
        
    @staticmethod
    def _print_grouping_metrics(metrics: Dict[str, Any]):
        """Print difficulty and app count grouping metrics."""
        # Difficulty grouping
        diff_keys = [k for k in metrics.keys() if k.startswith("count_diff_")]
        if diff_keys:
            print(f"\n[>] By Difficulty:")
            for key in sorted(diff_keys):
                diff = key.replace("count_diff_", "")
                count = metrics[key]
                rates = [f"@{k}={metrics.get(f'pass_at_{k}_diff_{diff}', 0):.1f}%" for k in range(1, 4)]
                print(f"    D{diff}: {' | '.join(rates)} (n={count})")
                
        # App count grouping
        apps_keys = [k for k in metrics.keys() if k.startswith("count_apps_")]
        if apps_keys:
            print(f"\n[>] By App Count:")
            for key in sorted(apps_keys):
                apps = key.replace("count_apps_", "")
                count = metrics[key]
                rates = [f"@{k}={metrics.get(f'pass_at_{k}_apps_{apps}', 0):.1f}%" for k in range(1, 4)]
                print(f"    {apps} Apps: {' | '.join(rates)} (n={count})")
                
    @staticmethod
    def _print_token_stats(metrics: Dict[str, Any]):
        """Print token statistics."""
        has_tokens = any(k.startswith("token_att1_") for k in metrics.keys())
        if not has_tokens:
            return
            
        print(f"\n[>] Token Statistics:")
        for att in range(1, 4):
            total_prompt = metrics.get(f"token_att{att}_total_prompt", 0)
            total_completion = metrics.get(f"token_att{att}_total_completion", 0)
            total_steps = metrics.get(f"token_att{att}_total_steps", 0)
            avg_prompt = metrics.get(f"token_att{att}_avg_prompt", 0)
            avg_completion = metrics.get(f"token_att{att}_avg_completion", 0)
            
            if total_steps > 0:
                print(f"    Att{att}: {total_steps:,} steps | "
                      f"In: {total_prompt:,} (avg {avg_prompt:,}) | "
                      f"Out: {total_completion:,} (avg {avg_completion:,})")


# ============================================================================
# Main Interface Functions
# ============================================================================
def calculate_and_save_metrics(
    output_dir: str,
    max_attempts: int = 3,
    trigger: str = "",
    print_summary: bool = True,
    save_to_file: bool = True,
) -> Dict[str, Any]:
    """
    Calculate all metrics from results.csv and save/print them.
    
    This is the main entry point for real-time metrics calculation.
    Call this function after each task execution/evaluation to update metrics.
    
    Args:
        output_dir: Session output directory (contains results.csv)
        max_attempts: Maximum number of attempts per task
        trigger: Description of what triggered this calculation
        print_summary: Whether to print metrics to console
        save_to_file: Whether to save metrics to files
        
    Returns:
        Dictionary containing all calculated metrics
    """
    csv_path = os.path.join(output_dir, "results.csv")
    
    # Calculate metrics
    calculator = MetricsCalculator(csv_path, max_attempts)
    if not calculator.load_data():
        return {}
        
    metrics = calculator.calculate_all_metrics()
    metrics["trigger"] = trigger
    
    # Save to files
    if save_to_file:
        saver = MetricsSaver(output_dir)
        saver.save_json(metrics)
        saver.save_csv(metrics)
        saver.append_history(metrics)
        
    # Print summary
    if print_summary:
        MetricsPrinter.print_summary(metrics, trigger)
        
    return metrics


def get_metrics_from_csv(csv_path: str, max_attempts: int = 3) -> Dict[str, Any]:
    """
    Calculate metrics from a specific results.csv file.
    
    Args:
        csv_path: Path to results.csv
        max_attempts: Maximum number of attempts
        
    Returns:
        Dictionary containing all calculated metrics
    """
    calculator = MetricsCalculator(csv_path, max_attempts)
    if not calculator.load_data():
        return {}
    return calculator.calculate_all_metrics()


# ============================================================================
# Leaderboard Format Generator
# ============================================================================
class LeaderboardResultGenerator:
    """
    Generates results in MemGUI-Bench leaderboard format.
    
    The leaderboard format follows the structure defined in:
    docs/data/agents/*.json
    
    Each agent is stored as a separate JSON file with the following structure:
    {
        "name": "Agent-Name",
        "backbone": "Model-Name",
        "type": "Agentic Workflow" | "Agent-as-a-Model",
        ...
    }
    """
    
    @staticmethod
    def generate(
        metrics: Dict[str, Any],
        agent_name: str = "Unknown",
    ) -> Dict[str, Any]:
        """
        Generate leaderboard-compatible result JSON.
        
        Metadata fields (backbone, type, institution, date, paperLink, 
        codeLink, hasUITree, hasLongTermMemory) are left as placeholders for the 
        user to fill in before submission.
        
        Args:
            metrics: Metrics dictionary from MetricsCalculator
            agent_name: Name of the agent
            
        Returns:
            Dictionary in leaderboard format (compatible with data/agents/*.json)
        """
        # Extract cross-app metrics (by num_apps)
        cross_app = {}
        for app_count in [1, 2, 3, 4]:
            app_key = f"apps_{app_count}"
            p1 = metrics.get(f"pass_at_1_{app_key}", 0)
            p3 = metrics.get(f"pass_at_3_{app_key}", 0)
            
            # IRR for this app count (use per-app-count IRR if available)
            irr = metrics.get(f"irr_{app_key}", metrics.get("avg_irr", 0))
            
            cross_app[f"app{app_count}"] = {
                "p1": round(p1, 1),
                "p3": round(p3, 1),
                "irr": round(irr, 1),
            }
        
        # Extract difficulty metrics (including IRR per difficulty level)
        difficulty = {}
        diff_mapping = {"1": "easy", "2": "medium", "3": "hard"}
        for diff_num, diff_name in diff_mapping.items():
            diff_key = f"diff_{diff_num}"
            p1 = metrics.get(f"pass_at_1_{diff_key}", 0)
            p3 = metrics.get(f"pass_at_3_{diff_key}", 0)
            irr = metrics.get(f"irr_{diff_key}", 0)
            difficulty[diff_name] = {
                "p1": round(p1, 1),
                "p3": round(p3, 1),
                "irr": round(irr, 2),
            }
        
        # Overall average
        avg_p1 = metrics.get("pass_at_1_rate", 0)
        avg_p3 = metrics.get("pass_at_3_rate", 0)
        
        # Core metrics
        irr = metrics.get("avg_irr", 0)
        mtpr = metrics.get("mtpr", 0)
        frr = metrics.get("frr", 0)
        step_ratio = metrics.get("step_ratio_at_1", 0)
        time_per_step = metrics.get("time_per_step_at_1", 0)
        cost_per_step = metrics.get("cost_per_step_at_1", 0)
        
        # Long-term metrics (using pass@3 data)
        step_ratio_p3 = metrics.get("step_ratio_at_3", step_ratio)
        time_per_step_p3 = metrics.get("time_per_step_at_3", time_per_step)
        cost_per_step_p3 = metrics.get("cost_per_step_at_3", cost_per_step)
        
        # Build result in the exact format of data/agents/*.json
        # Metadata fields are left as placeholders for user to fill in
        result = {
            "name": agent_name,
            "backbone": "-",  # TODO: Fill in backbone model name (e.g., "Gemini-2.5-Pro", or "-" for Agent-as-a-Model)
            "type": "",  # TODO: Fill in "Agentic Workflow" or "Agent-as-a-Model"
            "institution": "",  # TODO: Fill in institution name
            "date": "",  # TODO: Fill in submission date (YYYY-MM-DD)
            "paperLink": "",  # TODO: Fill in paper link
            "codeLink": "",  # TODO: Fill in code repository link
            "hasUITree": False,  # TODO: Set to true if agent uses UI tree
            "hasLongTermMemory": False,  # TODO: Set to true if agent has long-term memory
            "crossApp": cross_app,
            "difficulty": difficulty,
            "avg": {
                "p1": round(avg_p1, 1),
                "p3": round(avg_p3, 1),
            },
            "metrics": {
                "shortTerm": {
                    "irr": round(irr, 1),
                    "mtpr": round(mtpr, 2),
                    "stepRatio": round(step_ratio, 2) if step_ratio else None,
                    "timePerStep": round(time_per_step, 1),
                    "costPerStep": round(cost_per_step, 4) if cost_per_step else None,
                },
                "longTerm": {
                    "frr": round(frr, 1),
                    "stepRatio": round(step_ratio_p3, 2) if step_ratio_p3 else None,
                    "timePerStep": round(time_per_step_p3, 1),
                    "costPerStep": round(cost_per_step_p3, 4) if cost_per_step_p3 else None,
                },
            },
        }
        
        return result
    
    @staticmethod
    def get_filename(agent_name: str) -> str:
        """
        Generate the filename for the agent JSON file.
        
        Converts agent name to lowercase, replaces spaces/underscores with hyphens.
        Example: "Agent-S2" -> "agent-s2.json"
                 "UI_TARS_1.5_7B" -> "ui-tars-1.5-7b.json"
        """
        # Convert to lowercase and replace spaces/underscores with hyphens
        filename = agent_name.lower()
        filename = filename.replace(" ", "-").replace("_", "-")
        # Remove consecutive hyphens
        while "--" in filename:
            filename = filename.replace("--", "-")
        return f"{filename}.json"


def save_leaderboard_result(
    output_dir: str,
    metrics: Dict[str, Any],
    agent_name: str = "Unknown",
) -> str:
    """
    Save metrics in MemGUI-Bench leaderboard format.
    
    This generates a JSON file that can be directly submitted to the leaderboard.
    The file format matches docs/data/agents/*.json
    
    Note: Metadata fields (backbone, type, institution, date, paperLink,
    codeLink, hasUITree, hasLongTermMemory) are left as placeholders for the user
    to fill in before submission.
    
    Args:
        output_dir: Session output directory
        metrics: Metrics dictionary from calculate_and_save_metrics()
        agent_name: Name of the agent
        
    Returns:
        Path to the saved leaderboard result file
    """
    # Generate leaderboard format (matching data/agents/*.json structure)
    leaderboard_result = LeaderboardResultGenerator.generate(
        metrics=metrics,
        agent_name=agent_name,
    )
    
    # Generate filename based on agent name (e.g., "qwen3vl.json")
    filename = LeaderboardResultGenerator.get_filename(agent_name)
    path = os.path.join(output_dir, filename)
    
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(leaderboard_result, f, indent=2, ensure_ascii=False)
    except IOError as e:
        print(f"Warning: Failed to save leaderboard result: {e}")
        
    return path

