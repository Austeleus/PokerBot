"""
Training Stage Controller for Deep CFR.

This module manages transitions between pure CFR, hybrid, and network training stages.
"""

from typing import Dict, Any
import logging


class TrainingStageController:
    """
    Controller for managing MCCFR training stage transitions.
    
    Handles automatic transitions between:
    1. Pure CFR (external regrets only)
    2. Hybrid (blend external regrets with network predictions) 
    3. Network (primarily network predictions)
    """
    
    def __init__(self, 
                 pure_cfr_iterations: int = 500,
                 hybrid_iterations: int = 500,
                 auto_transition: bool = True):
        """
        Initialize training stage controller.
        
        Args:
            pure_cfr_iterations: Number of iterations for pure CFR stage
            hybrid_iterations: Number of iterations for hybrid stage
            auto_transition: Whether to automatically transition stages
        """
        self.pure_cfr_iterations = pure_cfr_iterations
        self.hybrid_iterations = hybrid_iterations
        self.auto_transition = auto_transition
        
        # Stage transition thresholds
        self.stage_transitions = {
            'pure_cfr': pure_cfr_iterations,
            'hybrid': pure_cfr_iterations + hybrid_iterations,
            'network': float('inf')
        }
        
        self.logger = logging.getLogger(__name__)
    
    def get_current_stage(self, iteration: int) -> str:
        """
        Get current training stage based on iteration.
        
        Args:
            iteration: Current training iteration
            
        Returns:
            Current stage ('pure_cfr', 'hybrid', 'network')
        """
        if iteration < self.pure_cfr_iterations:
            return 'pure_cfr'
        elif iteration < self.pure_cfr_iterations + self.hybrid_iterations:
            return 'hybrid'
        else:
            return 'network'
    
    def get_blend_factor(self, iteration: int, stage: str = None) -> float:
        """
        Calculate blend factor between external regrets and network predictions.
        
        Args:
            iteration: Current training iteration
            stage: Current stage (auto-detected if None)
            
        Returns:
            Blend factor (0.0 = pure external regrets, 1.0 = pure network)
        """
        if stage is None:
            stage = self.get_current_stage(iteration)
        
        if stage == 'pure_cfr':
            return 0.0
        elif stage == 'network':
            return 1.0
        else:  # hybrid stage
            # Gradual transition within hybrid stage
            hybrid_start = self.pure_cfr_iterations
            progress = max(0, iteration - hybrid_start)
            blend_factor = min(1.0, progress / self.hybrid_iterations)
            return blend_factor * 0.8  # Max 80% network in hybrid stage
    
    def should_update_network(self, iteration: int, stage: str = None) -> bool:
        """
        Decide whether to update network at current iteration.
        
        Args:
            iteration: Current training iteration
            stage: Current stage (auto-detected if None)
            
        Returns:
            Whether to update network
        """
        if stage is None:
            stage = self.get_current_stage(iteration)
        
        if stage == 'pure_cfr':
            # Update network occasionally even in pure CFR stage
            return iteration % 10 == 0 and iteration > 50
        elif stage == 'hybrid':
            # Regular network updates in hybrid stage
            return True
        else:  # network stage
            # Frequent network updates in network stage
            return True
    
    def get_network_update_frequency(self, stage: str) -> int:
        """
        Get how often to update network for given stage.
        
        Args:
            stage: Current training stage
            
        Returns:
            Update frequency (every N iterations)
        """
        if stage == 'pure_cfr':
            return 10  # Update every 10 iterations
        elif stage == 'hybrid':
            return 2   # Update every 2 iterations
        else:  # network stage
            return 1   # Update every iteration
    
    def get_stage_config(self, stage: str) -> Dict[str, Any]:
        """
        Get configuration parameters for a specific stage.
        
        Args:
            stage: Training stage
            
        Returns:
            Configuration dictionary
        """
        configs = {
            'pure_cfr': {
                'blend_factor': 0.0,
                'update_network': False,
                'focus': 'regret_accumulation',
                'network_loss_weight': 0.1,
                'external_regret_weight': 1.0
            },
            'hybrid': {
                'blend_factor': 'dynamic',
                'update_network': True,
                'focus': 'strategy_alignment',
                'network_loss_weight': 0.5,
                'external_regret_weight': 0.8
            },
            'network': {
                'blend_factor': 1.0,
                'update_network': True,
                'focus': 'network_refinement',
                'network_loss_weight': 1.0,
                'external_regret_weight': 0.2
            }
        }
        
        return configs.get(stage, configs['pure_cfr'])
    
    def log_stage_transition(self, old_stage: str, new_stage: str, iteration: int):
        """
        Log stage transition information.
        
        Args:
            old_stage: Previous training stage
            new_stage: New training stage
            iteration: Iteration at which transition occurred
        """
        self.logger.info(
            f"Training stage transition: {old_stage} -> {new_stage} at iteration {iteration}"
        )
        
        new_config = self.get_stage_config(new_stage)
        self.logger.info(f"New stage config: {new_config}")
    
    def get_statistics(self, iteration: int) -> Dict[str, Any]:
        """
        Get current stage statistics.
        
        Args:
            iteration: Current iteration
            
        Returns:
            Statistics dictionary
        """
        stage = self.get_current_stage(iteration)
        blend_factor = self.get_blend_factor(iteration, stage)
        
        return {
            'current_stage': stage,
            'blend_factor': blend_factor,
            'stage_progress': self._get_stage_progress(iteration, stage),
            'iterations_in_stage': self._get_iterations_in_stage(iteration, stage),
            'should_update_network': self.should_update_network(iteration, stage),
            'update_frequency': self.get_network_update_frequency(stage)
        }
    
    def _get_stage_progress(self, iteration: int, stage: str) -> float:
        """Get progress within current stage (0.0 to 1.0)."""
        if stage == 'pure_cfr':
            return min(1.0, iteration / self.pure_cfr_iterations)
        elif stage == 'hybrid':
            hybrid_start = self.pure_cfr_iterations
            progress = max(0, iteration - hybrid_start)
            return min(1.0, progress / self.hybrid_iterations)
        else:  # network stage
            return 1.0  # Always complete in network stage
    
    def _get_iterations_in_stage(self, iteration: int, stage: str) -> int:
        """Get number of iterations spent in current stage."""
        if stage == 'pure_cfr':
            return min(iteration, self.pure_cfr_iterations)
        elif stage == 'hybrid':
            hybrid_start = self.pure_cfr_iterations
            if iteration >= hybrid_start:
                return min(iteration - hybrid_start, self.hybrid_iterations)
            else:
                return 0
        else:  # network stage
            network_start = self.pure_cfr_iterations + self.hybrid_iterations
            return max(0, iteration - network_start)