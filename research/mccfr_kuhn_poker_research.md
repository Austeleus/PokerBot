# Monte Carlo CFR (MCCFR) for Two-Player Kuhn Poker - Research Summary

## Overview

Monte Carlo Counterfactual Regret Minimization (MCCFR) is a stochastic variant of CFR that uses sampling to reduce computational complexity while maintaining convergence guarantees to Nash equilibrium. For Kuhn poker, this approach is particularly effective due to the game's simplicity.

## Kuhn Poker Game Rules

**Setup:**
- 3 cards: King (K=3), Queen (Q=2), Jack (J=1)
- 2 players, each dealt 1 card
- 1 card remains as dead card
- Each player antes 1 chip

**Betting Round:**
- Player 1 acts first: Check or Bet (1 chip)
- If P1 checks: P2 can Check (showdown) or Bet (P1 can Call/Fold)
- If P1 bets: P2 can Call (showdown) or Fold
- If both check: pot = 2, winner gets 2 (net +1/-1)
- If one bets and other calls: pot = 4, winner gets 4 (net +2/-2)
- If bet and fold: bettor wins pot (net +1/-1)

## MCCFR Algorithm Components

### 1. Information Sets
Information sets represent what a player knows at a decision point:
- Player's private card
- Action history (sequence of bets/checks)

**Example Information Sets for Player 1:**
- `J` (Jack, first to act)
- `Q` (Queen, first to act) 
- `K` (King, first to act)

**Example Information Sets for Player 2:**
- `J/C` (Jack, opponent checked)
- `J/B` (Jack, opponent bet)
- `Q/C` (Queen, opponent checked)
- `Q/B` (Queen, opponent bet)
- `K/C` (King, opponent checked)
- `K/B` (King, opponent bet)

### 2. Regret Matching Algorithm

```python
def regret_matching(regrets):
    """Convert regrets to strategy via regret matching"""
    positive_regrets = max(0, regrets)
    sum_positive = sum(positive_regrets)
    
    if sum_positive > 0:
        strategy = positive_regrets / sum_positive
    else:
        strategy = uniform_random()  # Equal probability
    
    return strategy
```

### 3. Core MCCFR Algorithm Structure

```python
def mccfr_iteration(cards, history, player, reach_probabilities):
    """
    Monte Carlo CFR iteration using chance sampling
    
    Args:
        cards: [player1_card, player2_card] 
        history: string of actions taken (e.g., "CB" = check, bet)
        player: current player (0 or 1)
        reach_probabilities: [pi_1, pi_2] probability of reaching this node
    
    Returns:
        expected utility for current player
    """
    
    # Terminal node - return payoff
    if is_terminal(history):
        return get_payoff(cards, history, player)
    
    # Get information set for current player
    info_set = get_info_set(cards[player], history)
    
    # Get current strategy from regret matching
    strategy = regret_matching(regrets[info_set])
    
    # Initialize utilities for each action
    action_utilities = {}
    node_utility = 0
    
    # Sample all actions (chance sampling)
    for action in get_legal_actions():
        new_history = history + action
        new_reach_probs = reach_probabilities.copy()
        new_reach_probs[player] *= strategy[action]
        
        # Recursive call
        action_utilities[action] = -mccfr_iteration(
            cards, new_history, 1-player, new_reach_probs
        )
        
        # Accumulate expected utility
        node_utility += strategy[action] * action_utilities[action]
    
    # Update regrets
    for action in get_legal_actions():
        regret = action_utilities[action] - node_utility
        regrets[info_set][action] += reach_probabilities[1-player] * regret
        
        # Update strategy sum for average strategy
        strategy_sum[info_set][action] += reach_probabilities[player] * strategy[action]
    
    return node_utility
```

### 4. Chance Sampling vs Outcome Sampling

**Chance Sampling (Recommended for Kuhn Poker):**
- Sample all chance events (card deals)
- Explore all player actions
- Better for smaller games like Kuhn poker
- More stable convergence

**Outcome Sampling:**
- Sample both chance events and player actions
- Only one terminal node per iteration
- Better for very large games
- More variance but faster per iteration

## Implementation Architecture

### 1. Game State Representation

```python
class KuhnPokerState:
    def __init__(self):
        self.cards = None  # [p1_card, p2_card]
        self.history = ""  # Action sequence: "C"=check, "B"=bet, "F"=fold
        self.current_player = 0
        self.pot = 2  # Initial antes
        
    def is_terminal(self):
        # Terminal conditions:
        # - "CC" (both check)
        # - "BC" (bet-call) 
        # - "BF" (bet-fold)
        # - "CBF" (check-bet-fold)
        # - "CBC" (check-bet-call)
        pass
        
    def get_payoff(self, player):
        # Return payoff for specified player
        pass
        
    def get_info_set(self, player):
        # Return information set string
        return f"{self.cards[player]}/{self.history}"
```

### 2. Information Set Manager

```python
class InfoSetManager:
    def __init__(self):
        self.regret_sum = defaultdict(lambda: defaultdict(float))
        self.strategy_sum = defaultdict(lambda: defaultdict(float))
        self.num_actions = 2  # Check/Call (0), Bet/Raise (1)
        
    def get_strategy(self, info_set):
        """Get current strategy via regret matching"""
        regrets = self.regret_sum[info_set]
        return self.regret_matching(regrets)
        
    def get_average_strategy(self, info_set):
        """Get time-averaged strategy"""
        strategy_sum = self.strategy_sum[info_set]
        total = sum(strategy_sum.values())
        
        if total > 0:
            return {a: v/total for a, v in strategy_sum.items()}
        else:
            return {0: 0.5, 1: 0.5}  # Uniform default
```

### 3. MCCFR Trainer

```python
class MCCFRTrainer:
    def __init__(self):
        self.info_set_manager = InfoSetManager()
        self.cards = [1, 2, 3]  # J, Q, K
        
    def train(self, iterations):
        for i in range(iterations):
            # Sample random card deal
            cards = random.sample(self.cards, 2)
            
            # Run MCCFR iteration for both players
            for player in [0, 1]:
                self.mccfr(cards, "", player, [1.0, 1.0])
                
    def mccfr(self, cards, history, player, reach_probs):
        # Core MCCFR algorithm implementation
        pass
```

## Key Implementation Details

### 1. Information Set Encoding
- Format: `"{card}/{history}"` 
- Examples: `"1/"`, `"2/C"`, `"3/CB"`
- Separate info sets for each player

### 2. Action Encoding
- 0: Check/Call
- 1: Bet/Raise
- Actions depend on game state (check vs call, bet vs raise)

### 3. Terminal Node Payoffs
```python
def get_payoff(cards, history):
    """Calculate payoff for player 0"""
    if history == "CC":  # Both check
        return 1 if cards[0] > cards[1] else -1
    elif history == "BC":  # Bet-call
        return 2 if cards[0] > cards[1] else -2  
    elif history == "BF":  # Bet-fold
        return 1
    elif history == "CBF":  # Check-bet-fold
        return -1
    elif history == "CBC":  # Check-bet-call
        return 2 if cards[0] > cards[1] else -2
```

### 4. Convergence Monitoring
- Track exploitability over time
- Monitor average strategy stability
- Typical convergence: 10,000-100,000 iterations

## Expected Results

For optimal play in Kuhn poker:
- Player with King: Always bet/call
- Player with Queen: Mixed strategy (betting frequency ~0.33)
- Player with Jack: Mostly check/fold, sometimes bluff
- Game value: approximately -1/18 ≈ -0.0556 for player 1

## Performance Characteristics

- **Memory**: O(|I|) where |I| is number of information sets (~12 for Kuhn)
- **Time per iteration**: O(|I| × |A|) where |A| is actions per info set
- **Convergence**: Typically 10,000+ iterations for stable strategy
- **Variance**: Higher than vanilla CFR due to sampling, but manageable

This research provides the foundation for implementing a robust MCCFR solver for two-player Kuhn poker.