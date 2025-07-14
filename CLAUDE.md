# CLAUDE.md - Poker Bot Project Context

Rules for Claude:
1. First think through the problem, read the codebase for relevant files, and write a plan to tasks/todo.md. Specifically, read implementation.md for Technical Specifications. In the tasks file, you will find a TODO file.
2. The plan should have a list of todo items that you can check off as you complete them
3. Before you begin working, check in with me and I will verify the plan.
4. Then, begin working on the todo items, marking them as complete as you go.
5. Please every step of the way just give me a high level explanation of what changes you made
6. Make every task and code change you do as simple as possible. We want to avoid making any massive or complex changes. Every change should impact as little code as possible. Everything is about simplicity.
7. Finally, add a review section to the [todo.md](http://todo.md/) file with a summary of the changes you made and any other relevant information.

The todo.md file contains what you have done and what you should do.

## Project Overview

This is a sophisticated poker bot implementation that uses a two-stage approach combining **Deep Counterfactual Regret Minimization (CFR)** with **Reinforcement Learning fine-tuning** to create a game-theory optimal poker strategy.

### High-Level Architecture

1. **Stage 1: Deep CFR Blueprint** - Create an approximate equilibrium strategy using neural networks
2. **Stage 2: RL Fine-tuning** - Convert to Actor-Critic and improve through PPO self-play
3. **Stage 3: Evaluation** - Benchmark against baselines and measure exploitability

## Key Technical Components

### Environment Setup
- **Game**: 3-player No-Limit Hold'em with discrete bet sizes (½-pot, pot, all-in)
- **Framework**: PettingZoo `texas_holdem_no_limit_v6` environment
- **Action Space**: 5 discrete actions (Fold, Check/Call, Raise Half Pot, Raise Full Pot, All-In)
- **Observation Space**: 54-element vector (52 cards + 2 chip counts)

### Deep CFR Implementation
- **Algorithm**: External-sampling Monte Carlo CFR with neural network approximation
- **Network**: Transformer-based architecture for information set encoding
- **Memory**: Reservoir sampling buffer for experience replay (O(1) memory complexity)
- **Training**: Regret minimization through advantage network updates

### Transformer Architecture
- **Purpose**: Encode poker information sets (cards + betting history) into vector representations
- **Architecture**: Multi-layer Transformer encoder with self-attention mechanism
- **Heads**: Dual-head design (advantage estimation + policy distribution)
- **Input**: Tokenized sequence of cards and actions

### PPO Fine-tuning
- **Architecture**: Actor-Critic with shared Transformer backbone
- **Actor**: Policy network outputting action probabilities
- **Critic**: Value network estimating state values
- **Training**: Proximal Policy Optimization with KL regularization toward blueprint

### Reservoir Sampling
- **Algorithm**: Stochastic sampling maintaining equal probability (k/n) for each item
- **Complexity**: O(n) time, O(k) space
- **Usage**: Experience replay buffer for Deep CFR training

## Project Structure

```
poker_ai/
├── envs/
│   └── holdem_wrapper.py          # PettingZoo environment wrapper
├── encoders/
│   ├── info_set_transformer.py    # Transformer for information sets
│   └── card_utils.py              # Card indexing and utilities
├── memory/
│   └── reservoir.py               # O(1) reservoir sampling buffer
├── solvers/
│   └── deep_cfr/
│       ├── trainer.py             # External-sampling CFR trainer
│       └── evaluator.py           # Blueprint evaluation
├── agents/
│   ├── base_agent.py              # Abstract agent API
│   ├── cfr_agent.py               # Deep CFR policy wrapper
│   └── rl_agent.py                # Actor-Critic for PPO
└── utils/
    ├── logger.py                  # Logging utilities
    └── seed.py                    # Random seed management

scripts/
├── 10_run_deep_cfr.py            # Train Deep CFR blueprint
├── 20_eval_blueprint.py          # Evaluate blueprint strategy
├── 30_convert_to_rl.py           # Convert to Actor-Critic
├── 40_train_rl_finetune.py       # PPO self-play training
└── 50_eval_final.py              # Final tournament evaluation
```

## Implementation Order

1. **holdem_wrapper.py** - Environment abstraction with discrete betting
2. **card_utils.py** - Card encoding and hand evaluation utilities
3. **info_set_transformer.py** - Neural network for information set encoding
4. **reservoir.py** - Experience replay buffer implementation
5. **deep_cfr/trainer.py** - Core CFR algorithm with neural networks
6. **cfr_agent.py** - Agent wrapper for trained CFR policy
7. **rl_agent.py** - Actor-Critic architecture for fine-tuning
8. **Evaluation scripts** - Benchmarking and tournament evaluation

## Key Research Findings

### PettingZoo Integration
- Texas Hold'em No Limit v6 supports configurable number of players
- Action masking prevents illegal moves
- Observation includes full card state and chip counts
- Built on RLCard framework for card game environments

### Deep CFR Insights
- External sampling provides better performance than vanilla CFR
- Neural networks replace tabular regret storage for scalability
- Separate advantage networks for each player
- Theoretical convergence guarantees maintained

### Transformer Applications
- Self-attention mechanism captures card relationships
- Encoder architecture processes variable-length sequences
- Learned poker hand rankings after ~90M training hands
- Effective for information set representation in imperfect information games

### PPO Self-Play
- Actor-Critic framework suitable for imperfect information
- Dual-network approach (online/target) for stability
- Asynchronous policy updates for multi-agent training
- KL regularization prevents divergence from blueprint

### Reservoir Sampling
- Maintains uniform sampling probability as data streams
- O(1) memory complexity for large experience buffers
- Recent improvements with dynamic priority adjustment
- Essential for memory-efficient experience replay


## Dependencies

- PyTorch (neural networks)
- PettingZoo[classic] (poker environment)
- NumPy, Pandas (data processing)
- TensorBoard, WandB (logging and monitoring)
- tqdm (progress tracking)


# Using Gemini CLI for Large Codebase Analysis

When analyzing large codebases or multiple files that might exceed context limits, use the Gemini CLI with its massive
context window. Use `gemini -p` to leverage Google Gemini's large context capacity.

## File and Directory Inclusion Syntax

Use the `@` syntax to include files and directories in your Gemini prompts. The paths should be relative to WHERE you run the
  gemini command:

### Examples (some example files are not actually in this project):

**Single file analysis:**
gemini -p "@src/main.py Explain this file's purpose and structure"

Multiple files:
gemini -p "@package.json @src/index.js Analyze the dependencies used in the code"

Entire directory:
gemini -p "@src/ Summarize the architecture of this codebase"

Multiple directories:
gemini -p "@src/ @tests/ Analyze test coverage for the source code"

Current directory and subdirectories:
gemini -p "@./ Give me an overview of this entire project"

# Or use --all_files flag:
gemini --all_files -p "Analyze the project structure and dependencies"

Implementation Verification Examples

Check if a feature is implemented:
gemini -p "@src/ @lib/ Has dark mode been implemented in this codebase? Show me the relevant files and functions"

Verify authentication implementation:
gemini -p "@src/ @middleware/ Is JWT authentication implemented? List all auth-related endpoints and middleware"

Check for specific patterns:
gemini -p "@src/ Are there any React hooks that handle WebSocket connections? List them with file paths"

Verify error handling:
gemini -p "@src/ @api/ Is proper error handling implemented for all API endpoints? Show examples of try-catch blocks"

Check for rate limiting:
gemini -p "@backend/ @middleware/ Is rate limiting implemented for the API? Show the implementation details"

Verify caching strategy:
gemini -p "@src/ @lib/ @services/ Is Redis caching implemented? List all cache-related functions and their usage"

Check for specific security measures:
gemini -p "@src/ @api/ Are SQL injection protections implemented? Show how user inputs are sanitized"

Verify test coverage for features:
gemini -p "@src/payment/ @tests/ Is the payment processing module fully tested? List all test cases"

When to Use Gemini CLI

Use gemini -p when:
- Analyzing entire codebases or large directories
- Comparing multiple large files
- Need to understand project-wide patterns or architecture
- Current context window is insufficient for the task
- Working with files totaling more than 100KB
- Verifying if specific features, patterns, or security measures are implemented
- Checking for the presence of certain coding patterns across the entire codebase

Important Notes

- Paths in @ syntax are relative to your current working directory when invoking gemini
- The CLI will include file contents directly in the context
- No need for --yolo flag for read-only analysis
- Gemini's context window can handle entire codebases that would overflow Claude's context
- When checking implementations, be specific about what you're looking for to get accurate results


## Performance Targets

- **Deep CFR**: Convergence to approximate Nash equilibrium in 3-player NLHE
- **Blueprint**: Competitive against heuristic baselines
- **RL Fine-tuning**: Improved exploitability while maintaining robustness
- **Evaluation**: Positive win rate against previous versions and baselines

## References

- Brown et al. (2019) - Deep Counterfactual Regret Minimization
- Steinberger (2019) - Scalable Deep CFR Implementation
- Various 2024 research on PPO, reservoir sampling, and Transformer applications in poker