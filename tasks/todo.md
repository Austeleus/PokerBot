# TODO: Poker Bot Implementation Plan

## Analysis of Current Structure vs Implementation Plan

**Current Structure:**
- `poker_ai/` with `agents/`, `models/`, `utils/` folders
- Missing: `envs/`, `encoders/`, `memory/`, `solvers/`, `evaluation/`
- `scripts/` folder exists but empty

**Implementation Plan:**

### Phase 1: Core Directory Structure & Basic Files
- [ ] Create missing directory structure
- [ ] Create basic environment wrapper
- [ ] Create card utilities
- [ ] Create simple reservoir buffer
- [ ] Create basic transformer model

### Phase 2: Deep CFR Implementation
- [ ] Implement information set transformer
- [ ] Create Deep CFR trainer
- [ ] Create CFR agent wrapper
- [ ] Add basic training script

### Phase 3: PPO Implementation
- [ ] Create Actor-Critic agent
- [ ] Implement PPO trainer
- [ ] Add RL training script
- [ ] Create agent conversion utilities

### Phase 4: Evaluation & Testing
- [ ] Create evaluation framework
- [ ] Add baseline agents
- [ ] Create tournament evaluation
- [ ] Add configuration files

### Phase 5: Integration & Testing
- [ ] Create smoke test script
- [ ] Add unit tests
- [ ] Integration testing
- [ ] Performance optimization

## Detailed Task List

### 1. Directory Structure Setup
- [ ] Create `poker_ai/envs/` directory
- [ ] Create `poker_ai/encoders/` directory  
- [ ] Create `poker_ai/memory/` directory
- [ ] Create `poker_ai/solvers/` directory
- [ ] Create `poker_ai/solvers/deep_cfr/` directory
- [ ] Create `poker_ai/evaluation/` directory
- [ ] Create `configs/` subdirectories

### 2. Environment Implementation
- [ ] Create `poker_ai/envs/__init__.py`
- [ ] Create `poker_ai/envs/holdem_wrapper.py` (PettingZoo wrapper)
- [ ] Create `poker_ai/envs/tests/` directory
- [ ] Create `poker_ai/envs/tests/__init__.py`
- [ ] Create `poker_ai/envs/tests/test_env.py`

### 3. Card Utilities
- [ ] Create `poker_ai/encoders/__init__.py`
- [ ] Create `poker_ai/encoders/card_utils.py`
- [ ] Create `poker_ai/encoders/tests/` directory
- [ ] Create `poker_ai/encoders/tests/__init__.py`
- [ ] Create `poker_ai/encoders/tests/test_card_utils.py`

### 4. Memory System
- [ ] Create `poker_ai/memory/__init__.py`
- [ ] Create `poker_ai/memory/reservoir.py`
- [ ] Create `poker_ai/memory/tests/` directory
- [ ] Create `poker_ai/memory/tests/__init__.py`
- [ ] Create `poker_ai/memory/tests/test_reservoir.py`

### 5. Neural Network Models
- [ ] Create `poker_ai/models/__init__.py` (if not exists)
- [ ] Create `poker_ai/models/info_set_transformer.py`
- [ ] Create `poker_ai/models/tests/` directory
- [ ] Create `poker_ai/models/tests/__init__.py`
- [ ] Create `poker_ai/models/tests/test_transformer.py`

### 6. Deep CFR Solver
- [ ] Create `poker_ai/solvers/__init__.py`
- [ ] Create `poker_ai/solvers/deep_cfr/__init__.py`
- [ ] Create `poker_ai/solvers/deep_cfr/trainer.py`
- [ ] Create `poker_ai/solvers/deep_cfr/evaluator.py`
- [ ] Create `poker_ai/solvers/deep_cfr/tests/` directory
- [ ] Create `poker_ai/solvers/deep_cfr/tests/__init__.py`
- [ ] Create `poker_ai/solvers/deep_cfr/tests/test_trainer.py`

### 7. Agent Implementation
- [ ] Create `poker_ai/agents/__init__.py` (if not exists)
- [ ] Create `poker_ai/agents/base_agent.py`
- [ ] Create `poker_ai/agents/random_agent.py`
- [ ] Create `poker_ai/agents/cfr_agent.py`
- [ ] Create `poker_ai/agents/rl_agent.py`
- [ ] Create `poker_ai/agents/tests/` directory
- [ ] Create `poker_ai/agents/tests/__init__.py`
- [ ] Create `poker_ai/agents/tests/test_agents.py`

### 8. Evaluation Framework
- [ ] Create `poker_ai/evaluation/__init__.py`
- [ ] Create `poker_ai/evaluation/evaluator.py`
- [ ] Create `poker_ai/evaluation/metrics.py`
- [ ] Create `poker_ai/evaluation/tests/` directory
- [ ] Create `poker_ai/evaluation/tests/__init__.py`
- [ ] Create `poker_ai/evaluation/tests/test_evaluator.py`

### 9. Utility Functions
- [ ] Update `poker_ai/utils/__init__.py`
- [ ] Create `poker_ai/utils/logger.py`
- [ ] Create `poker_ai/utils/seed.py`
- [ ] Create `poker_ai/utils/config.py`

### 10. Training Scripts
- [ ] Create `scripts/00_env_smoke.py`
- [ ] Create `scripts/10_run_deep_cfr.py`
- [ ] Create `scripts/20_eval_blueprint.py`
- [ ] Create `scripts/30_convert_to_rl.py`
- [ ] Create `scripts/40_train_rl_finetune.py`
- [ ] Create `scripts/50_eval_final.py`

### 11. Configuration Files
- [ ] Create `configs/envs/` directory
- [ ] Create `configs/envs/holdem_3p.yaml`
- [ ] Create `configs/deep_cfr/` directory
- [ ] Create `configs/deep_cfr/default.yaml`
- [ ] Create `configs/ppo/` directory
- [ ] Create `configs/ppo/default.yaml`

### 12. Testing & Validation
- [ ] Run environment smoke test
- [ ] Run unit tests for all components
- [ ] Integration test Deep CFR training
- [ ] Integration test PPO training
- [ ] Performance benchmarking

## Implementation Strategy

1. **Start Simple**: Begin with minimal working versions of each component
2. **Test Early**: Create tests alongside implementation
3. **Incremental**: Build and test each component before moving to next
4. **Modular**: Keep components loosely coupled for easy testing
5. **Documentation**: Update docstrings and comments as we go

## Expected Challenges

1. **PettingZoo Integration**: Understanding the exact API and observation format
2. **Transformer Implementation**: Getting the tokenization and attention right
3. **CFR Algorithm**: Implementing external sampling correctly
4. **Memory Management**: Ensuring reservoir sampling works correctly
5. **Training Stability**: Getting PPO to converge properly

## Success Criteria

- [ ] Environment wrapper works with PettingZoo
- [ ] Card utilities correctly encode/decode poker hands
- [ ] Transformer can process poker information sets
- [ ] CFR trainer can run iterations without errors
- [ ] PPO agent can train on self-play data
- [ ] Evaluation framework provides meaningful metrics
- [ ] All unit tests pass
- [ ] Integration tests demonstrate learning

---

## Review Section

### Completed Tasks
**Phase 1: Foundation Components (COMPLETED)**

✅ **Directory Structure Setup**
- Created all missing directories (`envs/`, `encoders/`, `memory/`, `solvers/`, `evaluation/`)
- Added proper `__init__.py` files for module structure
- Updated requirements.txt with eval7 dependency

✅ **HoldemWrapper Implementation** 
- Built robust PettingZoo wrapper with discrete action mapping
- Handles 3+ player games with proper action translation
- Includes legal action masking and game state tracking
- Provides standardized observations for neural networks

✅ **Card Utilities with eval7**
- Complete card encoding/decoding system (52-card one-hot)
- Integrated eval7 for accurate hand strength evaluation  
- Hand bucketing and feature extraction for neural networks
- Card masking and random sampling utilities

✅ **Reservoir Sampling Buffer**
- O(1) insertion with uniform probability guarantees
- Support for both basic and prioritized experience replay
- Experience collector helpers for CFR and RL training
- Comprehensive statistics and batch sampling

✅ **Robust InfoSetTransformer Model**
- 4-layer transformer with configurable architecture
- Sophisticated tokenization (cards, actions, special tokens)
- Attention-weighted pooling for sequence representation
- Dual heads (advantage + policy) plus value head for RL
- Pre-norm layers for training stability
- Comprehensive encoding for poker information sets

✅ **Environment Smoke Test**
- Integration test covering all components
- Verifies environment, encoding, memory, and model work together
- Includes error handling and comprehensive assertions

### Changes Made
- Added prioritized reservoir buffer variant for future use
- Enhanced transformer with value head for RL training phase
- Implemented attention-weighted pooling instead of simple mean pooling
- Added comprehensive game state encoding with proper tokenization
- Made transformer more robust with pre-norm architecture

### Issues Encountered & Resolved
- **Import Structure**: Resolved circular import issues by careful module organization
- **PettingZoo API**: Had to study AEC API carefully for proper environment wrapping
- **Token Vocabulary**: Designed comprehensive token space (61 tokens) for all game elements
- **Memory Efficiency**: Used reservoir sampling to handle large experience streams
- **Dependency Issues**: Fixed missing dependencies (pygame, rlcard, Tuple import)
- **eval7 Integration**: Debugged hand evaluation with proper error handling and validation
- **Environment Testing**: Successfully ran smoke tests verifying all components work together

### Testing Results
✅ **Smoke Test Passed**: All Phase 1 components successfully tested
- CardEncoder: Card conversion, encoding, and hand evaluation working
- ReservoirBuffer: Memory system operational with proper sampling
- InfoSetTransformer: Model architecture functional with forward passes
- HoldemWrapper: Environment integration working with PettingZoo
- Component Integration: All systems work together in simulated poker gameplay

### Phase 2: Deep CFR Implementation (COMPLETED)
✅ **Base Agent Interface**
- Created abstract BaseAgent class with RandomAgent implementation
- Defined standard interface for all poker agents (get_action, reset, etc.)
- File: `poker_ai/agents/base_agent.py`

✅ **Deep CFR Trainer**
- Implemented external-sampling CFR with neural network approximation
- Features: regret matching, experience replay, proper counterfactual regret calculation
- Enhanced error logging and eliminated code duplication
- File: `poker_ai/solvers/deep_cfr/trainer.py`

✅ **CFR Agent Wrapper** 
- Created agent that uses trained CFR policy for gameplay
- Includes strategy extraction, temperature scaling, save/load functionality
- File: `poker_ai/agents/cfr_agent.py`

✅ **Training Script**
- Comprehensive training pipeline with evaluation, checkpointing, logging
- Configurable hyperparameters and automatic evaluation against baselines
- File: `scripts/10_run_deep_cfr.py`

✅ **Unit Tests & Bug Fixes**
- Complete test suites: 12 agent tests + 11 trainer tests (23 total)
- Fixed probability normalization and infinite loop issues
- Fixed hanging test problems in MockEnvironment
- All tests pass successfully

### CFR Reset - Starting Fresh with Simpler Implementation

**RESET COMPLETED**: Removed all complex CFR implementation files to start with simpler approach.

**Files Removed:**
- `poker_ai/solvers/` (entire directory)
- `poker_ai/evaluation/` (entire directory) 
- `data/` (training data)
- `poker_ai/agents/cfr_agent.py`
- CFR-related scripts: `03_debug_cfr.py`, `10_run_deep_cfr.py`, `test_cfr_logic.py`, `test_corrected_cfr.py`

**Files Preserved:**
- `poker_ai/envs/` (PettingZoo wrapper)
- `poker_ai/encoders/` (card utilities, transformer)
- `poker_ai/memory/` (reservoir buffer)
- `poker_ai/models/` (neural networks)
- `poker_ai/utils/` (general utilities)
- `poker_ai/agents/base_agent.py` (base agent class)

### Next Phase: Simple CFR Implementation
**Goal**: Implement basic CFR on a simplified poker variant first

**Proposed Simple Game**: Kuhn Poker (3-card, 1 bet round)
- [ ] Create simple Kuhn poker environment
- [ ] Implement basic tabular CFR (no neural networks)
- [ ] Verify CFR convergence on simple game
- [ ] Add basic evaluation metrics
- [ ] Once working, scale up complexity gradually

**Current Status**: Foundation components preserved. Ready to implement simple CFR on Kuhn poker.