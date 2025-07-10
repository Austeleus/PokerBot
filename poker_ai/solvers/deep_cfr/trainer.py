"""
Deep CFR trainer using external sampling and neural network approximation.

This implements the core Counterfactual Regret Minimization algorithm
with neural networks replacing tabular regret storage for scalability.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import copy
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
import logging

from poker_ai.memory.reservoir import ReservoirBuffer
from poker_ai.models.info_set_transformer import InfoSetTransformer
from poker_ai.encoders.card_utils import CardEncoder
from poker_ai.solvers.deep_cfr.info_set_manager import InfoSetManager
from poker_ai.solvers.deep_cfr.training_controller import TrainingStageController
from poker_ai.solvers.deep_cfr.exploitability import BestResponseCalculator
from poker_ai.solvers.deep_cfr.convergence_monitor import ConvergenceMonitor


class DeepCFRTrainer:
    """
    Deep CFR trainer using external sampling.
    
    This class implements the core CFR algorithm with neural network
    approximation for advantage and strategy networks.
    """
    
    def __init__(self, 
                 env,
                 network: InfoSetTransformer,
                 buffer_size: int = 1000000,
                 learning_rate: float = 1e-4,
                 batch_size: int = 512,
                 device: str = "cpu"):
        """
        Initialize the Deep CFR trainer.
        
        Args:
            env: Poker environment (HoldemWrapper)
            network: Transformer network for advantage/strategy estimation
            buffer_size: Size of experience replay buffer
            learning_rate: Learning rate for optimizer
            batch_size: Batch size for training
            device: Device to run on (cpu/cuda)
        """
        self.env = env
        self.network = network.to(device)
        self.device = device
        self.batch_size = batch_size
        
        # Optimizer for neural network
        self.optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)
        
        # Experience replay buffer
        self.buffer = ReservoirBuffer(buffer_size)
        
        # Card encoder for information set processing
        self.card_encoder = CardEncoder()
        
        # Information Set Manager for external regret storage
        self.info_set_manager = InfoSetManager(
            n_actions=5,
            regret_decay=0.99,
            strategy_decay=0.999,
            use_cfr_plus=True,
            regret_clip=10.0
        )
        
        # CFR statistics
        self.iteration = 0
        self.total_games = 0
        
        # Training statistics
        self.losses = []
        self.avg_regrets = []
        self.exploitabilities = []
        
        # Exploitability calculator
        self.best_response_calc = BestResponseCalculator(env, num_samples=100)
        
        # Convergence monitor
        self.convergence_monitor = ConvergenceMonitor(
            window_size=100,
            stability_threshold=0.01,
            convergence_patience=50
        )
        
        # Training stage controller
        self.stage_controller = TrainingStageController(
            pure_cfr_iterations=500,
            hybrid_iterations=500,
            auto_transition=True
        )
        
        # Training mode control
        self.training_stage = "pure_cfr"  # pure_cfr, hybrid, network
        self.transition_iteration = 500  # When to start blending with network
        
        # Logger
        self.logger = logging.getLogger(__name__)
        
    def train_iteration(self, num_traversals: int = 1000) -> Dict[str, float]:
        """
        Run one CFR iteration with external sampling MCCFR.
        
        Args:
            num_traversals: Number of game traversals per iteration
            
        Returns:
            Dictionary of training statistics
        """
        iteration_stats = {
            'games_played': 0,
            'experiences_collected': 0,
            'network_updates': 0,
            'avg_loss': 0.0,
            'traverser_stats': {},
            'training_stage': self.training_stage,
            'external_regret_stats': self.info_set_manager.get_statistics(),
            'network_blend_factor': self._calculate_current_blend_factor(),
            'stage_controller_stats': self.stage_controller.get_statistics(self.iteration)
        }
        
        # External Sampling MCCFR: Run traversals for each player as traverser
        players = list(range(self.env.num_players))
        
        for traverser in players:
            traverser_experiences = []
            
            # Run multiple traversals with this player as traverser
            for _ in range(num_traversals // len(players)):
                experiences = self._external_sampling_mccfr_traverse(traverser)
                if experiences:
                    traverser_experiences.extend(experiences)
                iteration_stats['games_played'] += 1
            
            # Add experiences to buffer
            for exp in traverser_experiences:
                self.buffer.add(exp)
            
            iteration_stats['experiences_collected'] += len(traverser_experiences)
            iteration_stats['traverser_stats'][f'player_{traverser}'] = len(traverser_experiences)
        
        # Update neural network based on stage controller decision
        should_update = self.stage_controller.should_update_network(self.iteration, self.training_stage)
        update_freq = self.stage_controller.get_network_update_frequency(self.training_stage)
        
        if should_update and len(self.buffer) >= self.batch_size and self.iteration % update_freq == 0:
            losses = []
            num_updates = max(1, len(self.buffer) // self.batch_size)
            
            # Adjust number of updates based on stage
            stage_config = self.stage_controller.get_stage_config(self.training_stage)
            max_updates = 10 if self.training_stage == 'network' else 5
            
            for _ in range(min(num_updates, max_updates)):
                loss = self._update_network()
                if loss is not None:
                    losses.append(loss)
                    iteration_stats['network_updates'] += 1
            
            if losses:
                iteration_stats['avg_loss'] = np.mean(losses)
                self.losses.append(iteration_stats['avg_loss'])
        
        # Auto-transition between training stages
        self._handle_stage_transitions()
        
        # Track convergence metrics (every 10 iterations to avoid overhead)
        if self.iteration % 10 == 0:
            convergence_metrics = self._track_convergence_metrics(iteration_stats)
            iteration_stats['convergence_metrics'] = convergence_metrics
        
        self.iteration += 1
        self.total_games += iteration_stats['games_played']
        
        return iteration_stats
    
    def _external_sampling_mccfr_traverse(self, traverser: int) -> List[Dict[str, Any]]:
        """
        Perform External Sampling MCCFR traversal for a specific traverser.
        
        In External Sampling MCCFR:
        - Sample chance events (cards) externally 
        - For traverser: explore ALL actions to compute counterfactual regrets
        - For opponents: sample actions according to current strategy
        
        Args:
            traverser: Player index who is the traverser (gets regret updates)
            
        Returns:
            List of experience dictionaries for the traverser
        """
        # Reset environment and start new game
        observations = self.env.reset()
        
        # Start recursive traversal from the root
        experiences = []
        
        try:
            # Get agent names mapping
            agent_names = self.env.agents
            traverser_name = agent_names[traverser] if traverser < len(agent_names) else None
            
            if traverser_name is None:
                return []
            
            # Perform recursive traversal
            traversal_result = self._mccfr_recursive_traverse(
                observations=observations,
                traverser=traverser,
                traverser_name=traverser_name,
                reach_prob_traverser=1.0,
                reach_prob_others=1.0,
                depth=0
            )
            
            if traversal_result and 'experiences' in traversal_result:
                experiences = traversal_result['experiences']
                
        except Exception as e:
            self.logger.error(f"Error in MCCFR traversal for traverser {traverser}: {e}")
            return []
        
        return experiences
    
    def _mccfr_recursive_traverse(self, 
                                 observations: Dict[str, Any],
                                 traverser: int,
                                 traverser_name: str,
                                 reach_prob_traverser: float,
                                 reach_prob_others: float,
                                 depth: int,
                                 max_depth: int = 100) -> Dict[str, Any]:
        """
        Recursive traversal for External Sampling MCCFR.
        
        Args:
            observations: Current game observations
            traverser: Index of traverser player
            traverser_name: Name of traverser player
            reach_prob_traverser: Reach probability for traverser
            reach_prob_others: Reach probability for other players
            depth: Current recursion depth
            max_depth: Maximum recursion depth
            
        Returns:
            Dictionary with utility and experiences
        """
        # Prevent infinite recursion
        if depth > max_depth:
            return {'utility': 0.0, 'experiences': []}
        
        # Check if game is terminal
        if self.env.is_terminal():
            final_rewards = self.env.get_final_rewards()
            utility = final_rewards.get(traverser_name, 0.0)
            return {'utility': utility, 'experiences': []}
        
        # Get current player
        current_player_name = self.env.current_player()
        if current_player_name is None:
            return {'utility': 0.0, 'experiences': []}
        
        # Get player index
        current_player = self.env.agents.index(current_player_name)
        
        # Get legal actions
        legal_actions = self.env.get_legal_actions(current_player_name)
        if not legal_actions:
            return {'utility': 0.0, 'experiences': []}
        
        # Get current observation
        obs = observations.get(current_player_name, {})
        
        # Encode information set
        info_set_encoding = self._encode_information_set(obs, current_player_name)
        if info_set_encoding is None:
            # Fallback: sample random action
            action = random.choice(legal_actions)
            next_obs, rewards, done, info = self.env.step(action)
            return self._mccfr_recursive_traverse(
                next_obs, traverser, traverser_name,
                reach_prob_traverser, reach_prob_others, depth + 1
            )
        
        # Get strategy using external regrets (primary) with optional network blending
        strategy = self._get_combined_strategy(info_set_encoding, legal_actions, observations, current_player_name)
        
        experiences = []
        
        if current_player == traverser:
            # Traverser player: compute counterfactual regrets for all actions
            action_utilities = {}
            
            # Use Monte Carlo sampling to estimate counterfactual utilities
            # This is the key fix: we sample outcomes instead of modifying environment state
            for action in legal_actions:
                action_utilities[action] = self._sample_counterfactual_utility(
                    action, observations, traverser, traverser_name,
                    strategy, legal_actions, depth
                )
            
            # Compute strategy utility (expected utility under current strategy)
            strategy_utility = sum(strategy[action] * action_utilities[action] 
                                 for action in legal_actions)
            
            # Generate information set key for external storage
            info_set_key = self._generate_info_set_key(observations, current_player_name)
            
            # Compute counterfactual regrets
            regrets = np.zeros(5)  # 5 actions total
            for action in legal_actions:
                # Counterfactual regret = counterfactual value - strategy value
                regrets[action] = action_utilities[action] - strategy_utility
            
            # Scale regrets by reach probability of other players
            regrets = regrets * reach_prob_others
            
            # Update external regrets in InfoSetManager
            self.info_set_manager.update_regrets(info_set_key, regrets)
            
            # Create enhanced experience with stage-specific metadata
            experience = {
                'info_set': info_set_encoding,
                'info_set_key': info_set_key,
                'strategy': strategy.copy(),  # Strategy from external regrets
                'external_strategy': strategy.copy(),
                'regrets': regrets.copy(),  # For analysis only, not training
                'reward': strategy_utility,
                'legal_actions': legal_actions.copy(),
                'reach_prob': reach_prob_traverser,
                'counterfactual_values': action_utilities.copy(),
                'betting_round': info_set_encoding.get('betting_round', 0),
                'enhanced_features': info_set_encoding.get('enhanced_features', {}),
                'collection_stage': self.training_stage,
                'iteration': self.iteration,
                'player': current_player_name,
                'action': None  # Will be set when action is selected
            }
            experiences.append(experience)
            
            # Sample action according to current strategy for game continuation
            legal_probs = strategy[legal_actions]
            legal_probs = legal_probs / np.sum(legal_probs)
            action_idx = np.random.choice(len(legal_actions), p=legal_probs)
            sampled_action = legal_actions[action_idx]
            
            # Continue game with sampled action
            try:
                next_obs, rewards, done, info = self.env.step(sampled_action)
                if not done:
                    next_result = self._mccfr_recursive_traverse(
                        next_obs, traverser, traverser_name,
                        reach_prob_traverser, reach_prob_others, depth + 1
                    )
                    experiences.extend(next_result.get('experiences', []))
                    utility = next_result.get('utility', 0.0)
                else:
                    # Terminal state reached
                    final_rewards = self.env.get_final_rewards()
                    utility = final_rewards.get(traverser_name, 0.0)
            except Exception as e:
                self.logger.warning(f"Error in traverser game continuation: {e}")
                utility = 0.0
            
        else:
            # Opponent player: sample action according to strategy
            legal_probs = strategy[legal_actions]
            legal_probs = legal_probs / np.sum(legal_probs)
            action_idx = np.random.choice(len(legal_actions), p=legal_probs)
            sampled_action = legal_actions[action_idx]
            
            # Update reach probability for others
            new_reach_prob_others = reach_prob_others * strategy[sampled_action]
            
            # Continue with sampled action
            next_obs, rewards, done, info = self.env.step(sampled_action)
            next_result = self._mccfr_recursive_traverse(
                next_obs, traverser, traverser_name,
                reach_prob_traverser, new_reach_prob_others, depth + 1
            )
            
            utility = next_result.get('utility', 0.0)
            experiences.extend(next_result.get('experiences', []))
        
        return {'utility': utility, 'experiences': experiences}
    
    def _sample_counterfactual_utility(self,
                                       action: int,
                                       observations: Dict[str, Any],
                                       traverser: int,
                                       traverser_name: str,
                                       strategy: np.ndarray,
                                       legal_actions: List[int],
                                       depth: int,
                                       num_samples: int = 3) -> float:
        """
        Estimate counterfactual utility using Monte Carlo sampling.
        
        Instead of modifying the environment state, we use sampling and
        heuristics to estimate what would happen if this action was taken.
        
        Args:
            action: Action to evaluate
            observations: Current observations
            traverser: Traverser player index
            traverser_name: Traverser player name
            strategy: Current strategy
            legal_actions: Legal actions available
            depth: Current depth
            num_samples: Number of samples for estimation
            
        Returns:
            Estimated counterfactual utility
        """
        try:
            # Use multiple estimation methods and average them
            utilities = []
            
            # Method 1: Strategy-based estimation
            strategy_utility = self._estimate_utility_from_strategy(
                action, observations, traverser_name, strategy, legal_actions
            )
            utilities.append(strategy_utility)
            
            # Method 2: Hand strength based estimation
            hand_strength_utility = self._estimate_utility_from_hand_strength(
                action, observations, traverser_name
            )
            utilities.append(hand_strength_utility)
            
            # Method 3: Position and action type estimation
            position_utility = self._estimate_utility_from_position_and_action(
                action, observations, traverser_name, depth
            )
            utilities.append(position_utility)
            
            # Average the estimates with weights
            weights = [0.5, 0.3, 0.2]  # Favor strategy-based estimation
            weighted_utility = sum(w * u for w, u in zip(weights, utilities))
            
            # Add some noise to prevent overfitting
            noise = np.random.normal(0, 0.1)
            
            return weighted_utility + noise
            
        except Exception as e:
            self.logger.warning(f"Error in counterfactual utility sampling: {e}")
            return 0.0
    
    def _estimate_utility_from_strategy(self, 
                                       action: int,
                                       observations: Dict[str, Any],
                                       player_name: str,
                                       strategy: np.ndarray,
                                       legal_actions: List[int]) -> float:
        """Estimate utility based on strategy distribution."""
        try:
            # If action has high probability in strategy, it's likely good
            if action in legal_actions:
                action_prob = strategy[action]
                # Convert probability to utility estimate
                base_utility = (action_prob - 0.2) * 2.0  # Scale to [-0.4, 1.6]
                
                # Adjust based on action type
                if action == 0:  # Fold
                    return min(base_utility, 0.0)  # Folding rarely has positive utility
                elif action in [2, 3, 4]:  # Aggressive actions
                    return base_utility * 1.2  # Slightly favor aggression
                else:  # Check/call
                    return base_utility * 0.9
            
            return 0.0
        except Exception:
            return 0.0
    
    def _estimate_utility_from_hand_strength(self,
                                           action: int,
                                           observations: Dict[str, Any],
                                           player_name: str) -> float:
        """Estimate utility based on hand strength."""
        try:
            hole_cards = observations.get('hole_cards', [])
            community_cards = observations.get('community_cards', [])
            
            # Get hand strength estimate
            hand_strength = self._estimate_hand_strength(hole_cards, community_cards)
            
            # Map hand strength and action to utility
            if action == 0:  # Fold
                # Folding is good with weak hands, bad with strong hands
                return (0.5 - hand_strength) * 2.0
            elif action in [2, 3, 4]:  # Aggressive actions
                # Aggression is good with strong hands
                return (hand_strength - 0.5) * 3.0
            else:  # Check/call
                # Passive play is neutral
                return hand_strength * 0.5
                
        except Exception:
            return 0.0
    
    def _estimate_utility_from_position_and_action(self,
                                                  action: int,
                                                  observations: Dict[str, Any],
                                                  player_name: str,
                                                  depth: int) -> float:
        """Estimate utility based on position and action type."""
        try:
            # Extract enhanced features if available
            enhanced_features = self._extract_poker_features(observations, player_name)
            
            position = enhanced_features.get('position', 0)
            betting_round = enhanced_features.get('betting_round', 0)
            pot_odds = enhanced_features.get('pot_odds', 1.0)
            
            # Position-based adjustments
            position_factor = 1.0 + (position * 0.1)  # Later position slightly better
            
            # Betting round adjustments
            round_factor = 1.0 + (betting_round * 0.05)  # Later rounds more important
            
            # Action type base utilities
            action_base = {
                0: -0.5,   # Fold
                1: 0.0,    # Check/call
                2: 0.3,    # Raise half
                3: 0.5,    # Raise full
                4: 0.7     # All-in
            }
            
            base_utility = action_base.get(action, 0.0)
            adjusted_utility = base_utility * position_factor * round_factor
            
            # Pot odds adjustment for calling/betting actions
            if action in [1, 2, 3] and pot_odds < 0.5:
                adjusted_utility *= 1.2  # Better pot odds increase utility
            
            return adjusted_utility
            
        except Exception:
            return 0.0
    
    def _encode_information_set(self, observation: Dict[str, Any], player: str) -> Optional[Dict[str, torch.Tensor]]:
        """
        Encode poker information set for neural network input with enhanced features.
        
        Args:
            observation: Game observation for the player
            player: Player identifier
            
        Returns:
            Encoded information set with enhanced poker features or None if encoding fails
        """
        try:
            # Extract basic card information
            hole_cards = observation.get('hole_cards', [])
            community_cards = observation.get('community_cards', [])
            action_history = observation.get('action_history', [])
            
            # Enhanced information set features
            enhanced_features = self._extract_poker_features(observation, player)
            
            # Use the transformer's encoding method with enhanced context
            tokens = self.network.encode_game_state(
                hole_cards=hole_cards,
                community_cards=community_cards,
                action_history=action_history[-20:],  # Last 20 actions for memory efficiency
                current_player=enhanced_features.get('position', 0)
            )
            
            # Create attention mask (all tokens are valid)
            seq_len = tokens.size(1)
            attention_mask = torch.ones((1, seq_len), dtype=torch.bool, device=self.device)
            
            # Apply device
            tokens = tokens.to(self.device)
            
            return {
                'tokens': tokens,
                'attention_mask': attention_mask,
                'enhanced_features': enhanced_features,  # Store for potential use
                'hole_cards': hole_cards,
                'community_cards': community_cards,
                'betting_round': enhanced_features.get('betting_round', 0)
            }
            
        except Exception as e:
            # Log detailed information for debugging
            self.logger.error(
                f"Failed to encode information set for player {player}: {e}\n"
                f"Observation: {observation}\n"
                f"Hole cards: {observation.get('hole_cards', [])}\n"
                f"Community cards: {observation.get('community_cards', [])}\n"
                f"Action history: {observation.get('action_history', [])}"
            )
            return None
    
    def _extract_poker_features(self, observation: Dict[str, Any], player: str) -> Dict[str, Any]:
        """
        Extract enhanced poker-specific features from observation.
        
        Args:
            observation: Game observation
            player: Player identifier
            
        Returns:
            Dictionary of enhanced poker features
        """
        features = {}
        
        try:
            # Betting round determination
            community_cards = observation.get('community_cards', [])
            if len(community_cards) == 0:
                features['betting_round'] = 0  # Preflop
            elif len(community_cards) == 3:
                features['betting_round'] = 1  # Flop
            elif len(community_cards) == 4:
                features['betting_round'] = 2  # Turn
            elif len(community_cards) >= 5:
                features['betting_round'] = 3  # River
            else:
                features['betting_round'] = 0
            
            # Player position (simplified)
            agent_names = getattr(self.env, 'agents', [])
            if player in agent_names:
                features['position'] = agent_names.index(player)
            else:
                features['position'] = 0
            
            # Pot odds estimation (simplified)
            observation_vector = observation.get('observation', np.zeros(54))
            if len(observation_vector) >= 54:
                # Last 2 dimensions often represent chip information
                chip_info = observation_vector[-2:]
                features['pot_size'] = float(chip_info[0]) if len(chip_info) > 0 else 0.0
                features['stack_size'] = float(chip_info[1]) if len(chip_info) > 1 else 100.0
                
                # Calculate simple pot odds
                if features['pot_size'] > 0:
                    features['pot_odds'] = features['stack_size'] / (features['stack_size'] + features['pot_size'])
                else:
                    features['pot_odds'] = 1.0
            else:
                features['pot_size'] = 0.0
                features['stack_size'] = 100.0
                features['pot_odds'] = 1.0
            
            # Action sequence analysis
            action_history = observation.get('action_history', [])
            features['num_actions'] = len(action_history)
            features['recent_aggression'] = self._calculate_aggression_factor(action_history[-5:])
            
            # Hand strength estimation (if cards are available)
            hole_cards = observation.get('hole_cards', [])
            if hole_cards and len(hole_cards) >= 2:
                features['hand_strength'] = self._estimate_hand_strength(hole_cards, community_cards)
            else:
                features['hand_strength'] = 0.5  # Neutral strength
            
        except Exception as e:
            self.logger.warning(f"Feature extraction failed for player {player}: {e}")
            # Return default features on error
            features = {
                'betting_round': 0,
                'position': 0,
                'pot_size': 0.0,
                'stack_size': 100.0,
                'pot_odds': 1.0,
                'num_actions': 0,
                'recent_aggression': 0.0,
                'hand_strength': 0.5
            }
        
        return features
    
    def _calculate_aggression_factor(self, recent_actions: List[int]) -> float:
        """
        Calculate aggression factor from recent actions.
        
        Args:
            recent_actions: List of recent action indices
            
        Returns:
            Aggression factor (0.0 = passive, 1.0 = aggressive)
        """
        if not recent_actions:
            return 0.0
        
        aggressive_count = 0
        for action in recent_actions:
            if action in [2, 3, 4]:  # Raise half, raise full, all-in
                aggressive_count += 1
        
        return aggressive_count / len(recent_actions)
    
    def _estimate_hand_strength(self, hole_cards: List[str], community_cards: List[str]) -> float:
        """
        Estimate hand strength using simple heuristics.
        
        Args:
            hole_cards: Player's hole cards
            community_cards: Community cards
            
        Returns:
            Hand strength estimate (0.0 = weak, 1.0 = strong)
        """
        try:
            # Use the card encoder for hand evaluation if available
            if hasattr(self, 'card_encoder') and hole_cards:
                try:
                    hand_strength = self.card_encoder.evaluate_hand(hole_cards, community_cards)
                    # Normalize to 0-1 range (7462 is maximum hand rank in eval7)
                    return 1.0 - (hand_strength / 7462.0)
                except Exception:
                    pass
            
            # Fallback: simple heuristics
            if not hole_cards or len(hole_cards) < 2:
                return 0.5
            
            # Simple pair detection
            card1, card2 = hole_cards[0], hole_cards[1]
            if len(card1) >= 2 and len(card2) >= 2:
                rank1, rank2 = card1[0], card2[0]
                if rank1 == rank2:
                    return 0.8  # Pocket pair
                elif rank1 in ['A', 'K', 'Q'] and rank2 in ['A', 'K', 'Q']:
                    return 0.7  # High cards
                elif rank1 in ['A', 'K', 'Q'] or rank2 in ['A', 'K', 'Q']:
                    return 0.6  # One high card
                else:
                    return 0.4  # Low cards
            
            return 0.5  # Default
            
        except Exception:
            return 0.5  # Safe default
    
    def _get_strategy(self, info_set_encoding: Dict[str, torch.Tensor], legal_actions: List[int]) -> np.ndarray:
        """
        Get strategy from neural network using regret matching.
        
        Args:
            info_set_encoding: Encoded information set
            legal_actions: List of legal action indices
            
        Returns:
            Strategy probability distribution
        """
        try:
            with torch.no_grad():
                # Forward pass through network
                tokens = info_set_encoding['tokens']
                attention_mask = info_set_encoding['attention_mask']
                
                advantages, policy_logits, values = self.network(tokens, attention_mask)
                advantages = advantages.squeeze().cpu().numpy()
            
            # Apply regret matching to get strategy
            strategy = self._regret_matching(advantages, legal_actions)
            
            return strategy
            
        except Exception as e:
            # Log detailed error information for debugging
            self.logger.error(
                f"Failed to get strategy from network: {e}\n"
                f"Info set encoding shapes: {[(k, v.shape) for k, v in info_set_encoding.items()]}\n"
                f"Legal actions: {legal_actions}\n"
                f"Network device: {next(self.network.parameters()).device}"
            )
            # Fallback to uniform strategy
            strategy = np.zeros(5)
            if legal_actions:
                prob = 1.0 / len(legal_actions)
                for action in legal_actions:
                    strategy[action] = prob
            return strategy
    
    def _regret_matching(self, advantages: np.ndarray, legal_actions: List[int], 
                        info_set_key: Optional[str] = None) -> np.ndarray:
        """
        Enhanced regret matching with CFR+ and strategy averaging.
        
        Args:
            advantages: Advantage values for each action
            legal_actions: List of legal action indices
            info_set_key: Optional key for information set (for strategy averaging)
            
        Returns:
            Strategy probability distribution
        """
        # Initialize strategy
        strategy = np.zeros(len(advantages))
        
        # Only consider legal actions
        if not legal_actions:
            return strategy
        
        # Get advantages for legal actions only
        legal_advantages = advantages[legal_actions]
        
        # Apply CFR+ enhancement: use maximum of 0 and discounted regrets
        if self.use_cfr_plus:
            # CFR+ uses positive part with discount factor
            positive_regrets = np.maximum(legal_advantages, 0)
            # Apply regret decay
            positive_regrets = positive_regrets * self.regret_decay
        else:
            # Standard CFR: just positive part
            positive_regrets = np.maximum(legal_advantages, 0)
        
        # Clip regrets to prevent explosion
        positive_regrets = np.clip(positive_regrets, 0, self.regret_clip)
        
        sum_positive_regrets = np.sum(positive_regrets)
        
        if sum_positive_regrets > 0:
            # Distribute probability proportional to positive regrets
            for i, action in enumerate(legal_actions):
                strategy[action] = positive_regrets[i] / sum_positive_regrets
        else:
            # Uniform distribution over legal actions with small smoothing
            prob = 1.0 / len(legal_actions)
            for action in legal_actions:
                strategy[action] = prob
        
        # Apply strategy smoothing to prevent pure strategies
        epsilon = 0.01  # Small smoothing factor
        uniform_strategy = np.zeros(len(advantages))
        for action in legal_actions:
            uniform_strategy[action] = 1.0 / len(legal_actions)
        
        strategy = (1 - epsilon) * strategy + epsilon * uniform_strategy
        
        # Update cumulative strategy if info_set_key is provided
        if info_set_key is not None:
            self._update_cumulative_strategy(info_set_key, strategy, legal_actions)
        
        return strategy
    
    def _update_cumulative_strategy(self, info_set_key: str, strategy: np.ndarray, legal_actions: List[int]):
        """
        Update cumulative strategy for average strategy computation.
        
        Args:
            info_set_key: Key identifying the information set
            strategy: Current strategy
            legal_actions: Legal actions for this information set
        """
        if info_set_key not in self.cumulative_strategies:
            self.cumulative_strategies[info_set_key] = np.zeros_like(strategy)
            self.strategy_counts[info_set_key] = 0
        
        # Update cumulative strategy with decay
        decay_factor = self.strategy_decay
        self.cumulative_strategies[info_set_key] = (
            decay_factor * self.cumulative_strategies[info_set_key] + 
            (1 - decay_factor) * strategy
        )
        self.strategy_counts[info_set_key] += 1
    
    def get_average_strategy(self, info_set_key: str, legal_actions: List[int]) -> np.ndarray:
        """
        Get the average strategy for an information set.
        
        Args:
            info_set_key: Key identifying the information set
            legal_actions: Legal actions for this information set
            
        Returns:
            Average strategy over all iterations
        """
        if info_set_key in self.cumulative_strategies:
            avg_strategy = self.cumulative_strategies[info_set_key].copy()
            
            # Normalize to legal actions only
            legal_sum = sum(avg_strategy[action] for action in legal_actions)
            if legal_sum > 0:
                normalized_strategy = np.zeros_like(avg_strategy)
                for action in legal_actions:
                    normalized_strategy[action] = avg_strategy[action] / legal_sum
                return normalized_strategy
        
        # Fallback to uniform strategy
        uniform_strategy = np.zeros(5)  # 5 total actions
        prob = 1.0 / len(legal_actions)
        for action in legal_actions:
            uniform_strategy[action] = prob
        
        return uniform_strategy
    
    def _enhanced_regret_update(self, current_regrets: np.ndarray, new_regrets: np.ndarray) -> np.ndarray:
        """
        Enhanced regret update with decay and CFR+ improvements.
        
        Args:
            current_regrets: Current regret values
            new_regrets: New regret updates
            
        Returns:
            Updated regret values
        """
        if self.use_cfr_plus:
            # CFR+ update: positive regrets are accumulated, negative are discounted
            updated_regrets = np.zeros_like(current_regrets)
            for i in range(len(current_regrets)):
                if new_regrets[i] > 0:
                    # Accumulate positive regrets
                    updated_regrets[i] = current_regrets[i] + new_regrets[i]
                else:
                    # Discount negative regrets
                    updated_regrets[i] = max(0, current_regrets[i] + new_regrets[i] * 0.5)
        else:
            # Standard CFR update
            updated_regrets = current_regrets + new_regrets
        
        # Apply regret decay
        updated_regrets = updated_regrets * self.regret_decay
        
        # Clip to prevent explosion
        updated_regrets = np.clip(updated_regrets, -self.regret_clip, self.regret_clip)
        
        return updated_regrets
    
    def _generate_info_set_key(self, observations: Dict[str, Any], player_name: str) -> str:
        """
        Generate information set key using InfoSetManager.
        
        Args:
            observations: Current game observations
            player_name: Current player name
            
        Returns:
            Information set key string
        """
        try:
            # Extract information from observations
            obs = observations.get(player_name, {})
            hole_cards = obs.get('hole_cards', [])
            community_cards = obs.get('community_cards', [])
            action_history = obs.get('action_history', [])
            
            # Extract enhanced features
            enhanced_features = self._extract_poker_features(obs, player_name)
            
            return self.info_set_manager.generate_info_set_key(
                hole_cards=hole_cards,
                community_cards=community_cards,
                action_history=action_history[-10:],  # Last 10 actions
                position=enhanced_features.get('position', 0),
                betting_round=enhanced_features.get('betting_round', 0),
                stack_size=enhanced_features.get('stack_size', 100.0),
                pot_size=enhanced_features.get('pot_size', 0.0)
            )
            
        except Exception as e:
            self.logger.warning(f"Failed to generate info set key: {e}")
            return f"fallback_{player_name}_{self.iteration}"
    
    def _get_combined_strategy(self, 
                              info_set_encoding: Dict[str, torch.Tensor],
                              legal_actions: List[int],
                              observations: Dict[str, Any],
                              player_name: str) -> np.ndarray:
        """
        Get strategy combining external regrets with optional network predictions.
        
        Args:
            info_set_encoding: Encoded information set
            legal_actions: Legal actions
            observations: Game observations
            player_name: Current player name
            
        Returns:
            Combined strategy
        """
        # Generate info set key
        info_set_key = self._generate_info_set_key(observations, player_name)
        
        # Get strategy from external regrets (InfoSetManager)
        external_strategy = self.info_set_manager.get_strategy(info_set_key, legal_actions)
        
        # Determine blending based on training stage
        if self.training_stage == "pure_cfr" or self.iteration < self.transition_iteration:
            # Use pure external regret matching
            return external_strategy
        
        elif self.training_stage == "hybrid":
            # Blend external regrets with network predictions
            try:
                network_strategy = self._get_network_strategy(info_set_encoding, legal_actions)
                
                # Calculate blend factor (gradually increase network influence)
                progress = min((self.iteration - self.transition_iteration) / self.transition_iteration, 1.0)
                blend_factor = 0.3 * progress  # Max 30% network influence
                
                combined_strategy = (1 - blend_factor) * external_strategy + blend_factor * network_strategy
                return combined_strategy
                
            except Exception as e:
                self.logger.warning(f"Network strategy failed, using external: {e}")
                return external_strategy
        
        else:  # network mode
            # Use primarily network with small external component
            try:
                network_strategy = self._get_network_strategy(info_set_encoding, legal_actions)
                combined_strategy = 0.9 * network_strategy + 0.1 * external_strategy
                return combined_strategy
            except Exception as e:
                self.logger.warning(f"Network strategy failed, fallback to external: {e}")
                return external_strategy
    
    def _get_network_strategy(self, info_set_encoding: Dict[str, torch.Tensor], legal_actions: List[int]) -> np.ndarray:
        """
        Get strategy from neural network policy head.
        
        Args:
            info_set_encoding: Encoded information set
            legal_actions: Legal actions
            
        Returns:
            Network strategy
        """
        try:
            with torch.no_grad():
                tokens = info_set_encoding['tokens']
                attention_mask = info_set_encoding['attention_mask']
                
                # Forward pass through network
                advantages, policy_logits, values = self.network(tokens, attention_mask)
                
                # Use policy head for strategy (not advantages)
                policy_logits = policy_logits.squeeze().cpu().numpy()
                
                # Mask illegal actions
                masked_logits = np.full(len(policy_logits), -1e9)
                for action in legal_actions:
                    masked_logits[action] = policy_logits[action]
                
                # Apply softmax to get strategy
                strategy = F.softmax(torch.tensor(masked_logits), dim=-1).numpy()
                
                return strategy
                
        except Exception as e:
            self.logger.warning(f"Network strategy computation failed: {e}")
            # Fallback to uniform strategy
            strategy = np.zeros(5)
            if legal_actions:
                prob = 1.0 / len(legal_actions)
                for action in legal_actions:
                    strategy[action] = prob
            return strategy
    
    def set_training_stage(self, stage: str):
        """
        Set training stage: 'pure_cfr', 'hybrid', or 'network'.
        
        Args:
            stage: Training stage name
        """
        if stage in ["pure_cfr", "hybrid", "network"]:
            self.training_stage = stage
            self.logger.info(f"Training stage set to: {stage}")
        else:
            self.logger.warning(f"Invalid training stage: {stage}")
    
    
    def _update_network(self) -> Optional[float]:
        """
        Update neural network with proper multi-head training targets.
        
        This method trains the transformer to:
        - Policy head: Predict strategies from external regret matching
        - Advantage head: Predict action values (Q-values)  
        - Value head: Predict state values
        
        Returns:
            Training loss or None if update failed
        """
        if len(self.buffer) < self.batch_size:
            return None
        
        try:
            # Sample batch from buffer
            batch = self.buffer.sample(self.batch_size)
            
            # Prepare multi-head training data
            training_data = self._prepare_multihead_training_data(batch)
            if training_data is None:
                return None
            
            tokens = training_data['tokens']
            attention_masks = training_data['attention_masks']
            policy_targets = training_data['policy_targets']
            value_targets = training_data['value_targets']
            advantage_targets = training_data['advantage_targets']
            sample_weights = training_data['sample_weights']
            
            # Forward pass
            advantages, policy_logits, values = self.network(tokens, attention_masks)
            
            # Compute multi-head losses
            losses = self._compute_multihead_losses(
                advantages, policy_logits, values,
                advantage_targets, policy_targets, value_targets,
                sample_weights
            )
            
            # Combined loss with head-specific weights
            total_loss = (
                0.5 * losses['policy_loss'] +      # Primary: learn strategies
                0.3 * losses['advantage_loss'] +   # Secondary: learn action values  
                0.2 * losses['value_loss']         # Tertiary: learn state values
            )
            
            # Backward pass
            self.optimizer.zero_grad()
            total_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Return detailed loss info
            loss_info = {
                'total_loss': total_loss.item(),
                'policy_loss': losses['policy_loss'].item(),
                'advantage_loss': losses['advantage_loss'].item(), 
                'value_loss': losses['value_loss'].item()
            }
            
            # Log detailed losses occasionally
            if self.iteration % 100 == 0:
                self.logger.info(f"Training losses: {loss_info}")
            
            return total_loss.item()
            
        except Exception as e:
            self.logger.error(f"Network update failed: {e}")
            return None
    
    def _prepare_multihead_training_data(self, batch: List[Dict[str, Any]]) -> Optional[Dict[str, torch.Tensor]]:
        """
        Prepare training data for multi-head transformer training.
        
        Args:
            batch: Batch of experiences
            
        Returns:
            Dictionary with training tensors or None if preparation fails
        """
        try:
            tokens_list = []
            attention_masks = []
            policy_targets = []
            value_targets = []
            advantage_targets = []
            sample_weights = []
            
            for experience in batch:
                info_set = experience['info_set']
                info_set_key = experience['info_set_key']
                legal_actions = experience['legal_actions']
                reward = experience['reward']
                
                # Get tokens and masks
                tokens_list.append(info_set['tokens'].squeeze())
                attention_masks.append(info_set['attention_mask'].squeeze())
                
                # Policy target: Get strategy from external regrets (ground truth)
                external_strategy = self.info_set_manager.get_strategy(info_set_key, legal_actions)
                policy_targets.append(torch.tensor(external_strategy, dtype=torch.float32, device=self.device))
                
                # Value target: Use reward as state value estimate
                value_targets.append(torch.tensor([reward], dtype=torch.float32, device=self.device))
                
                # Advantage target: Use counterfactual values as action value estimates
                counterfactual_values = experience.get('counterfactual_values', {})
                advantage_vector = np.zeros(5)
                for action in legal_actions:
                    if action in counterfactual_values:
                        advantage_vector[action] = counterfactual_values[action]
                
                advantage_targets.append(torch.tensor(advantage_vector, dtype=torch.float32, device=self.device))
                
                # Sample weight: Higher weight for more visited information sets
                visit_count = self.info_set_manager.info_set_counts.get(info_set_key, 1)
                weight = min(np.log(visit_count + 1), 3.0)  # Cap at 3.0
                sample_weights.append(weight)
            
            # Pad sequences to same length
            max_len = max(tokens.size(0) for tokens in tokens_list)
            
            padded_tokens = []
            padded_masks = []
            
            for tokens, mask in zip(tokens_list, attention_masks):
                if tokens.size(0) < max_len:
                    pad_len = max_len - tokens.size(0)
                    tokens = torch.cat([tokens, torch.full((pad_len,), 59, dtype=torch.long, device=self.device)])
                    mask = torch.cat([mask, torch.zeros(pad_len, dtype=torch.bool, device=self.device)])
                
                padded_tokens.append(tokens)
                padded_masks.append(mask)
            
            # Stack into batches
            return {
                'tokens': torch.stack(padded_tokens),
                'attention_masks': torch.stack(padded_masks),
                'policy_targets': torch.stack(policy_targets),
                'value_targets': torch.stack(value_targets),
                'advantage_targets': torch.stack(advantage_targets),
                'sample_weights': torch.tensor(sample_weights, dtype=torch.float32, device=self.device)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to prepare training data: {e}")
            return None
    
    def _compute_multihead_losses(self,
                                 advantages: torch.Tensor,
                                 policy_logits: torch.Tensor, 
                                 values: torch.Tensor,
                                 advantage_targets: torch.Tensor,
                                 policy_targets: torch.Tensor,
                                 value_targets: torch.Tensor,
                                 sample_weights: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute losses for each transformer head.
        
        Returns:
            Dictionary of losses for each head
        """
        # Policy loss: KL divergence between predicted and target strategies
        # Use log probabilities for numerical stability
        log_policy_probs = F.log_softmax(policy_logits, dim=-1)
        policy_loss = F.kl_div(log_policy_probs, policy_targets, reduction='none')
        policy_loss = policy_loss.sum(dim=-1)  # Sum over actions
        policy_loss = (policy_loss * sample_weights).mean()  # Weighted average
        
        # Value loss: MSE between predicted and target values
        value_loss = F.mse_loss(values.squeeze(), value_targets.squeeze(), reduction='none')
        value_loss = (value_loss * sample_weights).mean()
        
        # Advantage loss: MSE between predicted and target action values
        advantage_loss = F.mse_loss(advantages, advantage_targets, reduction='none')
        advantage_loss = advantage_loss.mean(dim=-1)  # Average over actions
        advantage_loss = (advantage_loss * sample_weights).mean()  # Weighted average
        
        return {
            'policy_loss': policy_loss,
            'value_loss': value_loss, 
            'advantage_loss': advantage_loss
        }
    
    def get_exploitability(self) -> float:
        """
        Calculate exploitability of current strategy.
        
        Returns:
            Exploitability estimate (lower is better)
        """
        try:
            # Use external regrets in early stages, network in later stages
            use_external_regrets = self.training_stage in ['pure_cfr', 'hybrid']
            
            exploitability = self.best_response_calc.calculate_exploitability(
                target_agent=self,
                info_set_manager=self.info_set_manager,
                use_external_regrets=use_external_regrets
            )
            
            # Store exploitability history
            self.exploitabilities.append(exploitability)
            
            # Log exploitability milestone
            if len(self.exploitabilities) % 10 == 0:
                recent_stats = self.best_response_calc.get_exploitability_statistics(
                    self.exploitabilities[-10:]
                )
                self.logger.info(
                    f"Exploitability stats (last 10): current={recent_stats['current']:.6f}, "
                    f"avg={recent_stats['average']:.6f}, trend={recent_stats['trend']:.6f}"
                )
            
            return exploitability
            
        except Exception as e:
            self.logger.warning(f"Error calculating exploitability: {e}")
            return float('inf')
    
    def save_checkpoint(self, filepath: str) -> None:
        """
        Save complete training checkpoint including external regrets.
        
        Args:
            filepath: Path to save checkpoint
        """
        checkpoint = {
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'iteration': self.iteration,
            'total_games': self.total_games,
            'losses': self.losses,
            'avg_regrets': self.avg_regrets,
            'training_stage': self.training_stage,
            'transition_iteration': self.transition_iteration
        }
        torch.save(checkpoint, filepath)
        
        # Save external regret state separately
        regret_filepath = filepath.replace('.pt', '_regrets.pkl')
        self.info_set_manager.save_state(regret_filepath)
        
        self.logger.info(f"Saved checkpoint: {filepath} and regrets: {regret_filepath}")
    
    def load_checkpoint(self, filepath: str) -> None:
        """
        Load complete training checkpoint including external regrets.
        
        Args:
            filepath: Path to load checkpoint from
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.iteration = checkpoint.get('iteration', 0)
        self.total_games = checkpoint.get('total_games', 0)
        self.losses = checkpoint.get('losses', [])
        self.avg_regrets = checkpoint.get('avg_regrets', [])
        self.training_stage = checkpoint.get('training_stage', 'pure_cfr')
        self.transition_iteration = checkpoint.get('transition_iteration', 500)
        
        # Load external regret state
        regret_filepath = filepath.replace('.pt', '_regrets.pkl')
        try:
            self.info_set_manager.load_state(regret_filepath)
            self.logger.info(f"Loaded checkpoint: {filepath} and regrets: {regret_filepath}")
        except Exception as e:
            self.logger.warning(f"Could not load regret state: {e}")
    
    def get_training_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive training statistics.
        
        Returns:
            Dictionary with training and regret statistics
        """
        stats = {
            'iteration': self.iteration,
            'total_games': self.total_games,
            'training_stage': self.training_stage,
            'recent_losses': self.losses[-10:] if self.losses else [],
            'avg_recent_loss': np.mean(self.losses[-100:]) if len(self.losses) >= 100 else 0.0
        }
        
        # Add information set manager statistics
        stats.update(self.info_set_manager.get_statistics())
        
        return stats
    
    def _calculate_current_blend_factor(self) -> float:
        """
        Calculate current blend factor between external regrets and network predictions.
        
        Returns:
            Blend factor (0.0 = pure external regrets, 1.0 = pure network)
        """
        return self.stage_controller.get_blend_factor(self.iteration, self.training_stage)
    
    def _handle_stage_transitions(self):
        """
        Handle automatic transitions between training stages.
        """
        new_stage = self.stage_controller.get_current_stage(self.iteration)
        if new_stage != self.training_stage:
            old_stage = self.training_stage
            self.training_stage = new_stage
            self.stage_controller.log_stage_transition(old_stage, new_stage, self.iteration)
    
    def set_training_stage(self, stage: str):
        """
        Manually set training stage.
        
        Args:
            stage: Training stage ('pure_cfr', 'hybrid', 'network')
        """
        if stage in ['pure_cfr', 'hybrid', 'network']:
            self.training_stage = stage
            self.logger.info(f"Training stage set to: {stage}")
        else:
            raise ValueError(f"Invalid training stage: {stage}")
    
    def _create_enhanced_experiences(self, game_history: List[Dict[str, Any]], 
                                  final_rewards: Dict[str, float]) -> List[Dict[str, Any]]:
        """
        Create enhanced experiences with stage-specific metadata.
        
        Args:
            game_history: List of game state dictionaries
            final_rewards: Final rewards for each player
            
        Returns:
            List of enhanced experience dictionaries
        """
        experiences = []
        
        for history_entry in game_history:
            if 'player' not in history_entry:
                continue
                
            player = history_entry['player']
            reward = final_rewards.get(player, 0.0)
            
            # Get network strategy if available for comparison
            network_strategy = None
            if 'info_set' in history_entry:
                try:
                    info_set = history_entry['info_set']
                    legal_actions = history_entry.get('legal_actions', [])
                    
                    # Get network strategy for comparison
                    with torch.no_grad():
                        advantages, policy_logits, values = self.network.forward(info_set['tokens'])
                        
                        # Convert to strategy
                        masked_logits = policy_logits.clone()
                        mask = torch.full_like(masked_logits, -1e9)
                        for action in legal_actions:
                            mask[0, action] = 0
                        masked_logits += mask
                        
                        network_strategy = F.softmax(masked_logits, dim=-1).cpu().numpy()[0]
                        
                except Exception as e:
                    self.logger.debug(f"Could not get network strategy: {e}")
                    network_strategy = None
            
            # Create enhanced experience
            experience = {
                'info_set': history_entry.get('info_set'),
                'strategy': history_entry.get('strategy'),
                'action': history_entry.get('action'),
                'legal_actions': history_entry.get('legal_actions'),
                'reward': reward,
                'regrets': self._compute_regret_update(history_entry, reward),
                'collection_stage': self.training_stage,
                'iteration': self.iteration,
                'network_strategy': network_strategy,
                'external_strategy': history_entry.get('strategy'),
                'strategy_comparison': self._compare_strategies(
                    history_entry.get('strategy'), network_strategy
                ) if network_strategy is not None else None
            }
            
            experiences.append(experience)
        
        return experiences
    
    def _compare_strategies(self, external_strategy: Optional[np.ndarray], 
                          network_strategy: Optional[np.ndarray]) -> Optional[Dict[str, float]]:
        """
        Compare external regret strategy with network strategy.
        
        Args:
            external_strategy: Strategy from external regrets
            network_strategy: Strategy from network
            
        Returns:
            Dictionary with comparison metrics
        """
        if external_strategy is None or network_strategy is None:
            return None
        
        try:
            # KL divergence
            kl_div = 0.0
            l1_distance = 0.0
            max_action_diff = 0.0
            
            for i in range(len(external_strategy)):
                if external_strategy[i] > 1e-10:  # Avoid log(0)
                    if network_strategy[i] > 1e-10:
                        kl_div += external_strategy[i] * np.log(external_strategy[i] / network_strategy[i])
                
                l1_distance += abs(external_strategy[i] - network_strategy[i])
                max_action_diff = max(max_action_diff, abs(external_strategy[i] - network_strategy[i]))
            
            return {
                'kl_divergence': kl_div,
                'l1_distance': l1_distance,
                'max_action_diff': max_action_diff,
                'strategy_similarity': 1.0 - (l1_distance / 2.0)  # Normalized similarity
            }
            
        except Exception as e:
            self.logger.debug(f"Strategy comparison failed: {e}")
            return None
    
    def _track_convergence_metrics(self, iteration_stats: Dict[str, Any]) -> Dict[str, Any]:
        """
        Track convergence metrics for current iteration.
        
        Args:
            iteration_stats: Current iteration statistics
            
        Returns:
            Convergence metrics dictionary
        """
        try:
            # Collect current strategies from information set manager
            info_set_strategies = {}
            for info_set_key in list(self.info_set_manager.strategies.keys())[:100]:  # Limit to prevent overhead
                strategy = self.info_set_manager.strategies[info_set_key]
                if len(strategy) > 0:
                    info_set_strategies[info_set_key] = strategy
            
            # Get regret statistics
            regret_statistics = self.info_set_manager.get_statistics()
            
            # Calculate strategy agreement (if in hybrid/network stage)
            strategy_agreement = None
            if self.training_stage in ['hybrid', 'network'] and len(info_set_strategies) > 0:
                # Sample a few strategies to compare
                sample_keys = list(info_set_strategies.keys())[:10]
                agreement_metrics = []
                
                for key in sample_keys:
                    external_strategy = self.info_set_manager.get_strategy(key, [0, 1, 2, 3, 4])
                    
                    # Get network strategy (simplified)
                    try:
                        # This is a simplified comparison - in practice would need full state
                        network_strategy = np.ones(5) / 5  # Placeholder
                        comparison = self._compare_strategies(external_strategy, network_strategy)
                        if comparison:
                            agreement_metrics.append(comparison)
                    except Exception:
                        continue
                
                if agreement_metrics:
                    # Average agreement metrics
                    strategy_agreement = {
                        'strategy_similarity': np.mean([m['strategy_similarity'] for m in agreement_metrics]),
                        'kl_divergence': np.mean([m['kl_divergence'] for m in agreement_metrics]),
                        'l1_distance': np.mean([m['l1_distance'] for m in agreement_metrics])
                    }
            
            # Get current exploitability (calculate every 50 iterations to avoid overhead)
            exploitability = None
            if self.iteration % 50 == 0:
                try:
                    exploitability = self.get_exploitability()
                except Exception as e:
                    self.logger.debug(f"Error calculating exploitability for convergence: {e}")
            
            # Track convergence metrics
            convergence_metrics = self.convergence_monitor.track_iteration(
                iteration=self.iteration,
                info_set_strategies=info_set_strategies,
                regret_statistics=regret_statistics,
                strategy_agreement=strategy_agreement,
                exploitability=exploitability
            )
            
            # Log convergence status
            if self.convergence_monitor.is_converged:
                self.logger.info(f"Training has converged at iteration {self.iteration}")
            
            return convergence_metrics
            
        except Exception as e:
            self.logger.warning(f"Error tracking convergence metrics: {e}")
            return {'error': str(e)}
    
    def get_convergence_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive convergence summary.
        
        Returns:
            Convergence summary with detailed metrics
        """
        return self.convergence_monitor.get_convergence_summary()
    
    def is_converged(self) -> bool:
        """
        Check if training has converged.
        
        Returns:
            True if training has converged
        """
        return self.convergence_monitor.is_converged