"""
Convergence monitoring for Deep CFR training.

This module tracks strategy evolution, regret convergence, and training stability
to determine when the algorithm has sufficiently converged.
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict, deque
import torch


class ConvergenceMonitor:
    """
    Monitors convergence of MCCFR training across multiple metrics.
    
    Tracks:
    - Strategy stability over time
    - Regret magnitude convergence  
    - External regret vs network strategy agreement
    - Information set visitation patterns
    - Exploitability trends
    """
    
    def __init__(self, 
                 window_size: int = 100,
                 stability_threshold: float = 0.01,
                 convergence_patience: int = 50):
        """
        Initialize convergence monitor.
        
        Args:
            window_size: Number of recent iterations to analyze
            stability_threshold: Threshold for considering strategies stable
            convergence_patience: Number of stable iterations required for convergence
        """
        self.window_size = window_size
        self.stability_threshold = stability_threshold
        self.convergence_patience = convergence_patience
        
        # Strategy tracking
        self.strategy_history = defaultdict(deque)  # info_set -> deque of strategies
        self.strategy_variance_history = deque(maxlen=window_size)
        
        # Regret tracking
        self.regret_magnitude_history = deque(maxlen=window_size)
        self.regret_convergence_history = deque(maxlen=window_size)
        
        # Agreement tracking
        self.strategy_agreement_history = deque(maxlen=window_size)
        self.kl_divergence_history = deque(maxlen=window_size)
        
        # Exploitability tracking
        self.exploitability_history = deque(maxlen=window_size)
        
        # Information set statistics
        self.info_set_visitation = defaultdict(int)
        self.info_set_first_seen = {}
        
        # Convergence state
        self.stable_iterations = 0
        self.is_converged = False
        self.convergence_iteration = None
        
        self.logger = logging.getLogger(__name__)
    
    def track_iteration(self, 
                       iteration: int,
                       info_set_strategies: Dict[str, np.ndarray],
                       regret_statistics: Dict[str, Any],
                       strategy_agreement: Optional[Dict[str, float]] = None,
                       exploitability: Optional[float] = None) -> Dict[str, Any]:
        """
        Track metrics for one training iteration.
        
        Args:
            iteration: Current training iteration
            info_set_strategies: Dictionary of info_set -> strategy
            regret_statistics: Regret statistics from InfoSetManager
            strategy_agreement: Agreement metrics between external/network strategies
            exploitability: Current exploitability measurement
            
        Returns:
            Dictionary with convergence metrics for this iteration
        """
        # Update strategy history
        strategy_variance = self._update_strategy_tracking(info_set_strategies)
        
        # Update regret tracking
        regret_convergence = self._update_regret_tracking(regret_statistics)
        
        # Update agreement tracking
        agreement_metrics = self._update_agreement_tracking(strategy_agreement)
        
        # Update exploitability tracking
        exploitability_metrics = self._update_exploitability_tracking(exploitability)
        
        # Update information set visitation
        self._update_visitation_tracking(info_set_strategies, iteration)
        
        # Check convergence
        convergence_status = self._check_convergence(iteration)
        
        # Compile metrics
        metrics = {
            'iteration': iteration,
            'strategy_variance': strategy_variance,
            'regret_convergence': regret_convergence,
            'agreement_metrics': agreement_metrics,
            'exploitability_metrics': exploitability_metrics,
            'convergence_status': convergence_status,
            'info_set_stats': self._get_info_set_statistics(),
            'is_converged': self.is_converged,
            'stable_iterations': self.stable_iterations
        }
        
        # Log significant events
        self._log_convergence_events(iteration, metrics)
        
        return metrics
    
    def _update_strategy_tracking(self, info_set_strategies: Dict[str, np.ndarray]) -> float:
        """Update strategy stability tracking."""
        total_variance = 0.0
        valid_info_sets = 0
        
        for info_set, strategy in info_set_strategies.items():
            if info_set not in self.strategy_history:
                self.strategy_history[info_set] = deque(maxlen=self.window_size)
            
            self.strategy_history[info_set].append(strategy.copy())
            
            # Calculate variance if we have enough history
            if len(self.strategy_history[info_set]) >= 2:
                recent_strategies = list(self.strategy_history[info_set])[-10:]  # Last 10
                if len(recent_strategies) >= 2:
                    strategy_variance = self._calculate_strategy_variance(recent_strategies)
                    total_variance += strategy_variance
                    valid_info_sets += 1
        
        # Average strategy variance across all information sets
        avg_variance = total_variance / max(1, valid_info_sets)
        self.strategy_variance_history.append(avg_variance)
        
        return avg_variance
    
    def _calculate_strategy_variance(self, strategies: List[np.ndarray]) -> float:
        """Calculate variance of strategies over time."""
        try:
            if len(strategies) < 2:
                return 0.0
            
            # Convert to numpy array for easier calculation
            strategy_matrix = np.array(strategies)  # shape: (time, actions)
            
            # Calculate variance across time for each action
            action_variances = np.var(strategy_matrix, axis=0)
            
            # Return average variance across actions
            return float(np.mean(action_variances))
            
        except Exception as e:
            self.logger.debug(f"Error calculating strategy variance: {e}")
            return 0.0
    
    def _update_regret_tracking(self, regret_statistics: Dict[str, Any]) -> Dict[str, float]:
        """Update regret convergence tracking."""
        regret_metrics = {}
        
        # Extract regret magnitude
        avg_regret_magnitude = regret_statistics.get('avg_regret_magnitude', 0.0)
        max_regret_magnitude = regret_statistics.get('max_regret_magnitude', 0.0)
        
        self.regret_magnitude_history.append(avg_regret_magnitude)
        
        # Calculate regret convergence rate
        if len(self.regret_magnitude_history) >= 2:
            recent_regrets = list(self.regret_magnitude_history)[-10:]
            regret_trend = self._calculate_trend(recent_regrets)
            regret_stability = self._calculate_stability(recent_regrets)
        else:
            regret_trend = 0.0
            regret_stability = 0.0
        
        regret_metrics = {
            'avg_magnitude': avg_regret_magnitude,
            'max_magnitude': max_regret_magnitude,
            'trend': regret_trend,
            'stability': regret_stability
        }
        
        self.regret_convergence_history.append(regret_metrics)
        
        return regret_metrics
    
    def _update_agreement_tracking(self, strategy_agreement: Optional[Dict[str, float]]) -> Dict[str, float]:
        """Update strategy agreement tracking."""
        if strategy_agreement is None:
            return {'agreement': 0.0, 'kl_divergence': float('inf')}
        
        agreement = strategy_agreement.get('strategy_similarity', 0.0)
        kl_div = strategy_agreement.get('kl_divergence', float('inf'))
        
        self.strategy_agreement_history.append(agreement)
        self.kl_divergence_history.append(kl_div)
        
        # Calculate agreement trends
        if len(self.strategy_agreement_history) >= 2:
            agreement_trend = self._calculate_trend(list(self.strategy_agreement_history)[-10:])
            kl_trend = self._calculate_trend(list(self.kl_divergence_history)[-10:])
        else:
            agreement_trend = 0.0
            kl_trend = 0.0
        
        return {
            'agreement': agreement,
            'kl_divergence': kl_div,
            'agreement_trend': agreement_trend,
            'kl_trend': kl_trend
        }
    
    def _update_exploitability_tracking(self, exploitability: Optional[float]) -> Dict[str, float]:
        """Update exploitability tracking."""
        if exploitability is None or exploitability == float('inf'):
            return {'current': float('inf'), 'trend': 0.0, 'improvement': 0.0}
        
        self.exploitability_history.append(exploitability)
        
        # Calculate exploitability trends
        if len(self.exploitability_history) >= 2:
            trend = self._calculate_trend(list(self.exploitability_history)[-10:])
            improvement = self.exploitability_history[0] - exploitability if len(self.exploitability_history) > 0 else 0.0
        else:
            trend = 0.0
            improvement = 0.0
        
        return {
            'current': exploitability,
            'trend': trend,
            'improvement': improvement,
            'min_seen': min(self.exploitability_history) if self.exploitability_history else exploitability
        }
    
    def _update_visitation_tracking(self, info_set_strategies: Dict[str, np.ndarray], iteration: int):
        """Update information set visitation tracking."""
        for info_set in info_set_strategies.keys():
            self.info_set_visitation[info_set] += 1
            if info_set not in self.info_set_first_seen:
                self.info_set_first_seen[info_set] = iteration
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate trend (slope) of recent values."""
        if len(values) < 2:
            return 0.0
        
        try:
            # Simple linear regression slope
            n = len(values)
            x = np.arange(n)
            y = np.array(values)
            
            # Calculate slope: (n*Σxy - Σx*Σy) / (n*Σx² - (Σx)²)
            slope = (n * np.sum(x * y) - np.sum(x) * np.sum(y)) / (n * np.sum(x**2) - np.sum(x)**2)
            return float(slope)
            
        except Exception:
            return 0.0
    
    def _calculate_stability(self, values: List[float]) -> float:
        """Calculate stability (inverse of variance) of recent values."""
        if len(values) < 2:
            return 1.0
        
        try:
            variance = np.var(values)
            # Return inverse variance (higher = more stable)
            return 1.0 / (1.0 + variance)
            
        except Exception:
            return 0.0
    
    def _check_convergence(self, iteration: int) -> Dict[str, Any]:
        """Check if training has converged."""
        convergence_signals = {}
        
        # Check strategy stability
        if len(self.strategy_variance_history) > 0:
            recent_variance = np.mean(list(self.strategy_variance_history)[-10:])
            strategy_stable = recent_variance < self.stability_threshold
            convergence_signals['strategy_stable'] = strategy_stable
        else:
            strategy_stable = False
            convergence_signals['strategy_stable'] = False
        
        # Check regret convergence
        if len(self.regret_magnitude_history) >= 10:
            recent_regrets = list(self.regret_magnitude_history)[-10:]
            regret_trend = self._calculate_trend(recent_regrets)
            regret_stable = abs(regret_trend) < self.stability_threshold
            convergence_signals['regret_stable'] = regret_stable
        else:
            regret_stable = False
            convergence_signals['regret_stable'] = False
        
        # Check exploitability improvement
        if len(self.exploitability_history) >= 10:
            recent_exploitability = list(self.exploitability_history)[-10:]
            exploit_trend = self._calculate_trend(recent_exploitability)
            exploit_stable = abs(exploit_trend) < self.stability_threshold / 10  # More sensitive
            convergence_signals['exploitability_stable'] = exploit_stable
        else:
            exploit_stable = False
            convergence_signals['exploitability_stable'] = False
        
        # Overall convergence check
        all_stable = strategy_stable and regret_stable and exploit_stable
        
        if all_stable:
            self.stable_iterations += 1
        else:
            self.stable_iterations = 0
        
        # Declare convergence if stable for enough iterations
        if self.stable_iterations >= self.convergence_patience and not self.is_converged:
            self.is_converged = True
            self.convergence_iteration = iteration
            self.logger.info(f"Training converged at iteration {iteration}")
        
        convergence_signals['all_stable'] = all_stable
        convergence_signals['stable_iterations'] = self.stable_iterations
        convergence_signals['convergence_patience'] = self.convergence_patience
        
        return convergence_signals
    
    def _get_info_set_statistics(self) -> Dict[str, Any]:
        """Get information set visitation statistics."""
        if not self.info_set_visitation:
            return {'total_info_sets': 0, 'avg_visitation': 0.0}
        
        visitations = list(self.info_set_visitation.values())
        
        return {
            'total_info_sets': len(self.info_set_visitation),
            'avg_visitation': np.mean(visitations),
            'min_visitation': np.min(visitations),
            'max_visitation': np.max(visitations),
            'visitation_variance': np.var(visitations)
        }
    
    def _log_convergence_events(self, iteration: int, metrics: Dict[str, Any]):
        """Log significant convergence events."""
        # Log convergence milestone
        if self.stable_iterations > 0 and self.stable_iterations % 10 == 0:
            self.logger.info(
                f"Iteration {iteration}: Stable for {self.stable_iterations} iterations "
                f"(need {self.convergence_patience} for convergence)"
            )
        
        # Log strategy variance milestone
        strategy_variance = metrics.get('strategy_variance', 0.0)
        if iteration % 50 == 0:
            self.logger.debug(
                f"Iteration {iteration}: Strategy variance = {strategy_variance:.6f}, "
                f"Regret convergence = {metrics.get('regret_convergence', {}).get('stability', 0.0):.6f}"
            )
    
    def get_convergence_summary(self) -> Dict[str, Any]:
        """Get comprehensive convergence summary."""
        summary = {
            'is_converged': self.is_converged,
            'convergence_iteration': self.convergence_iteration,
            'stable_iterations': self.stable_iterations,
            'total_iterations_tracked': len(self.strategy_variance_history),
        }
        
        # Strategy metrics
        if self.strategy_variance_history:
            summary['strategy_metrics'] = {
                'current_variance': self.strategy_variance_history[-1],
                'avg_variance': np.mean(self.strategy_variance_history),
                'min_variance': np.min(self.strategy_variance_history),
                'variance_trend': self._calculate_trend(list(self.strategy_variance_history)[-20:])
            }
        
        # Regret metrics
        if self.regret_magnitude_history:
            summary['regret_metrics'] = {
                'current_magnitude': self.regret_magnitude_history[-1],
                'avg_magnitude': np.mean(self.regret_magnitude_history),
                'min_magnitude': np.min(self.regret_magnitude_history),
                'magnitude_trend': self._calculate_trend(list(self.regret_magnitude_history)[-20:])
            }
        
        # Exploitability metrics
        if self.exploitability_history:
            summary['exploitability_metrics'] = {
                'current': self.exploitability_history[-1],
                'best': np.min(self.exploitability_history),
                'avg': np.mean(self.exploitability_history),
                'improvement': self.exploitability_history[0] - self.exploitability_history[-1],
                'trend': self._calculate_trend(list(self.exploitability_history)[-20:])
            }
        
        # Information set metrics
        summary['info_set_metrics'] = self._get_info_set_statistics()
        
        return summary
    
    def reset(self):
        """Reset convergence monitor state."""
        self.strategy_history.clear()
        self.strategy_variance_history.clear()
        self.regret_magnitude_history.clear()
        self.regret_convergence_history.clear()
        self.strategy_agreement_history.clear()
        self.kl_divergence_history.clear()
        self.exploitability_history.clear()
        self.info_set_visitation.clear()
        self.info_set_first_seen.clear()
        
        self.stable_iterations = 0
        self.is_converged = False
        self.convergence_iteration = None