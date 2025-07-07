"""
PettingZoo wrapper for Texas Hold'em poker environment.
"""

from pettingzoo.classic import texas_holdem_no_limit_v6
import numpy as np
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
            Processed observation dictionary
        """
        if obs is None:
            return {
                'observation': np.zeros(54),
                'action_mask': np.zeros(5),
                'legal_actions': []
            }
        
        # PettingZoo observation is 54-dimensional:
        # - 52 for cards (one-hot)
        # - 2 for player chips
        processed = {
            'observation': obs['observation'].copy(),
            'action_mask': np.zeros(5),  # Our discrete action mask
            'legal_actions': self._get_legal_actions_from_mask(obs.get('action_mask'))
        }
        
        # Create discrete action mask
        for action in processed['legal_actions']:
            processed['action_mask'][action] = 1
        
        return processed
    
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
    
    def is_terminal(self) -> bool:
        """Check if the game is in a terminal state."""
        return self.game_over
    
    def render(self):
        """Render the current game state."""
        if self.render_mode is not None:
            self.env.render()
    
    def close(self):
        """Close the environment."""
        self.env.close()