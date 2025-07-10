"""
PettingZoo wrapper for Texas Hold'em poker environment.
"""

from pettingzoo.classic import texas_holdem_no_limit_v6
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any


class HoldemWrapper:
    """
    3-player NLHE wrapper with discrete bet sizes.
    
    Wraps PettingZoo's Texas Hold'em environment and provides
    a simplified interface with discrete actions and standardized
    observations for neural network training.
    """
    
    def __init__(self, num_players: int = 3, render_mode: Optional[str] = None):
        """
        Initialize the poker environment wrapper.
        
        Args:
            num_players: Number of players (default 3)
            render_mode: Rendering mode for visualization
        """
        self.num_players = num_players
        self.render_mode = render_mode
        self.env = texas_holdem_no_limit_v6.env(
            num_players=num_players,
            render_mode=render_mode
        )
        
        # Discrete action mapping
        self.action_mapping = {
            0: "fold",
            1: "check_call", 
            2: "raise_half_pot",
            3: "raise_full_pot",
            4: "all_in"
        }
        
        # Game state tracking
        self.current_player = None
        self.agents = []
        self.game_over = False
        self._pot_size = 0
        self._last_bet = 0
        self._player_chips = {}
        self._action_history = []  # Track action history for CFR
        
        # Logger
        self.logger = logging.getLogger(__name__)
        
    def reset(self, seed: Optional[int] = None) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Reset environment and return initial observations.
        
        Args:
            seed: Random seed for reproducibility
            
        Returns:
            Dictionary of initial observations for each agent
        """
        self.env.reset(seed=seed)
        self.agents = self.env.agents[:]
        self.game_over = False
        self._pot_size = 0
        self._last_bet = 0
        self._action_history = []  # Reset action history for new game
        
        # Initialize player chips
        for agent in self.agents:
            self._player_chips[agent] = 200  # Starting stack
        
        # Get initial observations
        observations = {}
        for agent in self.env.agent_iter():
            obs, _, termination, truncation, _ = self.env.last()
            
            if termination or truncation:
                self.env.step(None)
                continue
                
            if agent in self.agents:
                observations[agent] = self._process_observation(obs, agent)
            
            # Take first action for initial state
            if obs is not None and 'action_mask' in obs:
                action = self._get_default_action(obs['action_mask'])
                self.env.step(action)
                break
        
        return observations
    
    def step(self, action: int) -> Tuple[Dict[str, Dict[str, np.ndarray]], Dict[str, float], bool, Dict[str, Any]]:
        """
        Execute discrete action and return next state.
        
        Args:
            action: Discrete action index (0-4)
            
        Returns:
            Tuple of (observations, rewards, done, info)
        """
        if self.game_over:
            return {}, {}, True, {}
        
        # Get current agent
        current_agent = self.env.agent_selection
        
        # Convert discrete action to environment action
        env_action = self._discrete_to_env_action(action, current_agent)
        
        # Record action in history
        self._action_history.append(action)
        
        # Execute action
        self.env.step(env_action)
        
        # Collect observations and rewards for all agents
        observations = {}
        rewards = {}
        info = {'current_player': None}
        
        # Process next state
        for agent in self.env.agent_iter():
            obs, reward, termination, truncation, agent_info = self.env.last()
            
            if termination or truncation:
                if agent in self.agents:
                    rewards[agent] = reward
                self.env.step(None)
                continue
            
            # Found the next active agent
            if obs is not None:
                observations[agent] = self._process_observation(obs, agent)
                rewards[agent] = reward
                info['current_player'] = agent
                info['legal_actions'] = self._get_legal_actions_from_mask(obs.get('action_mask'))
                break
        
        # Check if game is done
        self.game_over = all(agent not in self.env.agents for agent in self.agents)
        
        return observations, rewards, self.game_over, info
    
    def get_legal_actions(self, agent: Optional[str] = None) -> List[int]:
        """
        Get legal discrete actions for current or specified agent.
        
        Args:
            agent: Agent name (uses current agent if None)
            
        Returns:
            List of legal discrete action indices
        """
        if agent is None:
            agent = self.env.agent_selection
        
        if agent not in self.env.agents:
            return []
        
        obs, _, _, _, _ = self.env.last()
        if obs is None or 'action_mask' not in obs:
            return []
        
        return self._get_legal_actions_from_mask(obs['action_mask'])
    
    def _discrete_to_env_action(self, discrete_action: int, agent: str) -> int:
        """
        Convert discrete action to PettingZoo environment action.
        
        Args:
            discrete_action: Our discrete action (0-4)
            agent: Current agent
            
        Returns:
            PettingZoo environment action
        """
        obs, _, _, _, _ = self.env.last()
        if obs is None:
            return 0
        
        # Get current game state from observation
        observation = obs['observation']
        
        # PettingZoo action space:
        # 0: Fold
        # 1: Check
        # 2: Call  
        # 3: Raise (min)
        # 4: All-in
        
        action_mask = obs.get('action_mask', np.ones(5))
        
        if discrete_action == 0:  # Fold
            return 0 if action_mask[0] else 1
            
        elif discrete_action == 1:  # Check/Call
            if action_mask[1]:  # Can check
                return 1
            elif action_mask[2]:  # Can call
                return 2
            else:
                return 1  # Default to check
                
        elif discrete_action == 2:  # Raise half pot
            if action_mask[3]:  # Can raise
                return 3
            elif action_mask[2]:  # Fall back to call
                return 2
            else:
                return 1  # Fall back to check
                
        elif discrete_action == 3:  # Raise full pot
            if action_mask[3]:  # Can raise
                return 3  # PettingZoo uses min raise
            elif action_mask[2]:  # Fall back to call
                return 2
            else:
                return 1  # Fall back to check
                
        elif discrete_action == 4:  # All-in
            if action_mask[4]:  # Can all-in
                return 4
            elif action_mask[3]:  # Fall back to raise
                return 3
            elif action_mask[2]:  # Fall back to call
                return 2
            else:
                return 1  # Fall back to check
        
        return 1  # Default to check
    
    def _get_legal_actions_from_mask(self, action_mask: np.ndarray) -> List[int]:
        """
        Convert PettingZoo action mask to our discrete legal actions.
        
        Args:
            action_mask: PettingZoo's action mask
            
        Returns:
            List of legal discrete actions
        """
        if action_mask is None:
            return []
        
        legal_actions = []
        
        # Map PettingZoo actions to our discrete actions
        if action_mask[0]:  # Can fold
            legal_actions.append(0)
            
        if action_mask[1] or action_mask[2]:  # Can check or call
            legal_actions.append(1)
            
        if action_mask[3]:  # Can raise
            legal_actions.append(2)  # Half pot
            legal_actions.append(3)  # Full pot
            
        if action_mask[4]:  # Can all-in
            legal_actions.append(4)
        
        return legal_actions if legal_actions else [1]  # Default to check/call
    
    def _process_observation(self, obs: Dict[str, np.ndarray], agent: str) -> Dict[str, np.ndarray]:
        """
        Process raw observation into standardized format.
        
        Args:
            obs: Raw observation from PettingZoo
            agent: Agent receiving observation
            
        Returns:
            Processed observation dictionary with structured card information
        """
        if obs is None:
            return {
                'observation': np.zeros(54),
                'action_mask': np.zeros(5),
                'legal_actions': [],
                'hole_cards': [],
                'community_cards': [],
                'action_history': []
            }
        
        # PettingZoo observation is 54-dimensional:
        # - 52 for cards (one-hot)
        # - 2 for player chips
        observation_vector = obs['observation']
        
        # Extract card information from the observation vector
        hole_cards, community_cards = self._extract_cards_from_observation(observation_vector, agent)
        
        processed = {
            'observation': observation_vector.copy(),
            'action_mask': np.zeros(5),  # Our discrete action mask
            'legal_actions': self._get_legal_actions_from_mask(obs.get('action_mask')),
            'hole_cards': hole_cards,
            'community_cards': community_cards,
            'action_history': getattr(self, '_action_history', [])  # Track action history
        }
        
        # Create discrete action mask
        for action in processed['legal_actions']:
            processed['action_mask'][action] = 1
        
        return processed
    
    def _extract_cards_from_observation(self, observation: np.ndarray, agent: str) -> Tuple[List[str], List[str]]:
        """
        Extract hole cards and community cards from PettingZoo observation vector.
        
        PettingZoo texas_holdem_no_limit_v6 observation format:
        - First 52 dimensions: Card encoding (0-1 values)
        - Last 2 dimensions: Chip information (0-100 values)
        
        Args:
            observation: 54-dimensional observation vector
            agent: Agent name
            
        Returns:
            Tuple of (hole_cards, community_cards) as string lists
        """
        hole_cards = []
        community_cards = []
        
        try:
            # First try to access the underlying RLCard environment for direct card access
            if hasattr(self.env, '_env') and hasattr(self.env._env, 'env'):
                rlcard_env = self.env._env.env
                if hasattr(rlcard_env, 'get_state'):
                    agent_idx = self.env.agents.index(agent) if agent in self.env.agents else 0
                    state = rlcard_env.get_state(agent_idx)
                    
                    # Extract hand cards
                    if 'hand' in state:
                        hole_cards = self._convert_rlcard_hand(state['hand'])
                    
                    # Extract public cards
                    if 'public_cards' in state:
                        community_cards = self._convert_rlcard_cards(state['public_cards'])
                    
                    # If we got cards from RLCard, return them
                    if hole_cards or community_cards:
                        return hole_cards, community_cards
            
            # Fallback: decode from observation vector
            # The first 52 dimensions represent card presence
            card_vector = observation[:52]
            
            # Create standard deck mapping
            suits = ['S', 'H', 'D', 'C']  # Spades, Hearts, Diamonds, Clubs
            ranks = ['A', '2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K']
            
            # Generate all 52 cards in standard order
            deck = []
            for suit in suits:
                for rank in ranks:
                    deck.append(rank + suit)
            
            # Extract cards where observation indicates presence
            visible_cards = []
            for i, card_present in enumerate(card_vector):
                if card_present > 0.5:  # Threshold for card presence
                    visible_cards.append(deck[i])
            
            # For now, assume first 2 visible cards are hole cards, rest are community
            # This is a simplification - actual game state tracking would be more complex
            if len(visible_cards) >= 2:
                hole_cards = visible_cards[:2]
                community_cards = visible_cards[2:]
            else:
                hole_cards = visible_cards
                community_cards = []
                
        except Exception as e:
            # Fallback to empty lists on any error
            self.logger.debug(f"Card extraction failed for agent {agent}: {e}")
            hole_cards = []
            community_cards = []
        
        return hole_cards, community_cards
    
    def _convert_rlcard_hand(self, rlcard_hand) -> List[str]:
        """
        Convert RLCard hand format to standard card strings.
        
        RLCard typically uses Card objects with suit and rank attributes.
        """
        cards = []
        try:
            for card in rlcard_hand:
                if hasattr(card, 'suit') and hasattr(card, 'rank'):
                    # Convert RLCard format to standard format
                    suit_map = {0: 'S', 1: 'H', 2: 'D', 3: 'C'}
                    rank_map = {0: 'A', 1: '2', 2: '3', 3: '4', 4: '5', 5: '6', 6: '7', 
                               7: '8', 8: '9', 9: 'T', 10: 'J', 11: 'Q', 12: 'K'}
                    
                    suit = suit_map.get(card.suit, 'S')
                    rank = rank_map.get(card.rank, 'A')
                    cards.append(rank + suit)
                elif isinstance(card, str):
                    # Already in string format
                    cards.append(card)
        except Exception:
            pass
        
        return cards
    
    def _convert_rlcard_cards(self, rlcard_cards) -> List[str]:
        """
        Convert RLCard public cards format to standard card strings.
        """
        return self._convert_rlcard_hand(rlcard_cards)  # Same conversion logic
    
    def _get_default_action(self, action_mask: np.ndarray) -> int:
        """Get a default valid action given PettingZoo's action mask."""
        if action_mask[1]:  # Check
            return 1
        elif action_mask[2]:  # Call
            return 2
        elif action_mask[0]:  # Fold
            return 0
        return 1  # Default
    
    def get_current_player(self) -> Optional[str]:
        """Get the current player to act."""
        return self.env.agent_selection if not self.game_over else None
    
    def current_player(self) -> Optional[str]:
        """Alias for get_current_player() for backward compatibility."""
        return self.get_current_player()
    
    def is_terminal(self) -> bool:
        """Check if the game is in a terminal state."""
        return self.game_over
    
    def is_done(self) -> bool:
        """Alias for is_terminal() for backward compatibility."""
        return self.is_terminal()
    
    def get_final_rewards(self) -> Dict[str, float]:
        """
        Get final rewards for all players at end of game.
        
        Returns:
            Dictionary mapping player names to final rewards
        """
        # Extract final rewards from the last step
        final_rewards = {}
        
        if self.game_over:
            # Get rewards from PettingZoo environment
            for agent in self.agents:
                if agent in self.env.rewards:
                    final_rewards[agent] = self.env.rewards[agent]
                else:
                    final_rewards[agent] = 0.0
        else:
            # Game not finished, return zeros
            for agent in self.agents:
                final_rewards[agent] = 0.0
        
        return final_rewards
    
    def render(self):
        """Render the current game state."""
        if self.render_mode is not None:
            self.env.render()
    
    def close(self):
        """Close the environment."""
        self.env.close()