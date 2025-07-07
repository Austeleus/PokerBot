"""
Transformer encoder for poker information sets.

This is the core neural network that processes poker game states
and outputs action advantages and policy distributions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Dict, Tuple, Optional, List


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for transformer inputs.
    """
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        """
        Initialize positional encoding.
        
        Args:
            d_model: Model dimension
            max_len: Maximum sequence length
            dropout: Dropout probability
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        # Register as buffer (not parameter)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input.
        
        Args:
            x: Input tensor of shape (seq_len, batch_size, d_model)
            
        Returns:
            Input with positional encoding added
        """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TokenEmbedding(nn.Module):
    """
    Embedding layer for poker tokens (cards, actions, special tokens).
    """
    
    def __init__(self, vocab_size: int, d_model: int, padding_idx: Optional[int] = None):
        """
        Initialize token embedding.
        
        Args:
            vocab_size: Size of vocabulary
            d_model: Model dimension
            padding_idx: Index used for padding
        """
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)
        
        # Initialize embeddings
        nn.init.normal_(self.embedding.weight, mean=0, std=d_model ** -0.5)
        if padding_idx is not None:
            nn.init.constant_(self.embedding.weight[padding_idx], 0)
    
    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Embed tokens.
        
        Args:
            tokens: Token indices of shape (batch_size, seq_len)
            
        Returns:
            Embeddings of shape (batch_size, seq_len, d_model)
        """
        return self.embedding(tokens) * math.sqrt(self.d_model)


class InfoSetTransformer(nn.Module):
    """
    Transformer encoder for poker information sets.
    
    This is the core model that processes sequences of cards and actions
    to produce advantage estimates and policy distributions for CFR and RL training.
    """
    
    def __init__(self,
                 d_model: int = 256,
                 n_heads: int = 8,
                 n_layers: int = 4,
                 n_actions: int = 5,
                 max_seq_len: int = 500,
                 dropout: float = 0.1,
                 activation: str = 'gelu',
                 layer_norm_eps: float = 1e-5):
        """
        Initialize information set transformer.
        
        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            n_layers: Number of transformer layers
            n_actions: Number of possible actions
            max_seq_len: Maximum sequence length
            dropout: Dropout probability
            activation: Activation function ('relu', 'gelu')
            layer_norm_eps: Layer norm epsilon
        """
        super().__init__()
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.n_actions = n_actions
        self.max_seq_len = max_seq_len
        
        # Vocabulary:
        # 0-51: Cards (52 cards)
        # 52-56: Actions (5 actions: fold, check/call, raise_half, raise_full, all_in)
        # 57: Start token
        # 58: Separator token (between rounds)
        # 59: Padding token
        # 60: Mask token
        self.vocab_size = 61
        self.card_tokens = list(range(52))
        self.action_tokens = list(range(52, 57))
        self.special_tokens = {
            'start': 57,
            'sep': 58,
            'pad': 59,
            'mask': 60
        }
        
        # Token embedding
        self.token_embedding = TokenEmbedding(
            vocab_size=self.vocab_size,
            d_model=d_model,
            padding_idx=self.special_tokens['pad']
        )
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len, dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation=activation,
            layer_norm_eps=layer_norm_eps,
            batch_first=True,
            norm_first=True  # Pre-norm for better training stability
        )
        
        self.transformer = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=n_layers,
            norm=nn.LayerNorm(d_model, eps=layer_norm_eps)
        )
        
        # Output projection layers
        self.advantage_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, n_actions),
        )
        
        self.policy_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, n_actions),
        )
        
        # Value head for RL training
        self.value_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.bias, 0)
                nn.init.constant_(module.weight, 1.0)
    
    def create_padding_mask(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Create padding mask for attention.
        
        Args:
            tokens: Token tensor of shape (batch_size, seq_len)
            
        Returns:
            Padding mask of shape (batch_size, seq_len)
        """
        return tokens == self.special_tokens['pad']
    
    def forward(self, 
                tokens: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through transformer.
        
        Args:
            tokens: Token sequence of shape (batch_size, seq_len)
            attention_mask: Optional attention mask
            
        Returns:
            Tuple of (advantages, policy_logits, values)
        """
        batch_size, seq_len = tokens.shape
        
        # Create padding mask if not provided
        if attention_mask is None:
            attention_mask = self.create_padding_mask(tokens)
        
        # Token embedding
        embeddings = self.token_embedding(tokens)  # (batch_size, seq_len, d_model)
        
        # Add positional encoding
        embeddings = self.pos_encoding(embeddings.transpose(0, 1)).transpose(0, 1)
        
        # Transformer encoding
        encoded = self.transformer(
            embeddings,
            src_key_padding_mask=attention_mask
        )  # (batch_size, seq_len, d_model)
        
        # Pool sequence representation
        # Use attention-weighted pooling to focus on important tokens
        pooled = self._attention_pool(encoded, attention_mask)
        
        # Output heads
        advantages = self.advantage_head(pooled)  # (batch_size, n_actions)
        policy_logits = self.policy_head(pooled)  # (batch_size, n_actions)
        values = self.value_head(pooled)  # (batch_size, 1)
        
        return advantages, policy_logits, values
    
    def _attention_pool(self, encoded: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Attention-weighted pooling of sequence.
        
        Args:
            encoded: Encoded sequence of shape (batch_size, seq_len, d_model)
            mask: Padding mask of shape (batch_size, seq_len)
            
        Returns:
            Pooled representation of shape (batch_size, d_model)
        """
        # Attention weights
        attention_weights = torch.tanh(torch.sum(encoded, dim=-1))  # (batch_size, seq_len)
        
        # Mask out padding tokens
        attention_weights = attention_weights.masked_fill(mask, -1e9)
        
        # Softmax over sequence dimension
        attention_weights = F.softmax(attention_weights, dim=-1)  # (batch_size, seq_len)
        
        # Weighted sum
        pooled = torch.sum(encoded * attention_weights.unsqueeze(-1), dim=1)  # (batch_size, d_model)
        
        return pooled
    
    def encode_game_state(self, 
                         hole_cards: List[str],
                         community_cards: List[str],
                         action_history: List[int],
                         current_player: int = 0) -> torch.Tensor:
        """
        Encode a poker game state into token sequence.
        
        Args:
            hole_cards: List of hole card strings (e.g., ['As', 'Kh'])
            community_cards: List of community card strings
            action_history: List of action indices
            current_player: Current player index
            
        Returns:
            Token sequence tensor
        """
        from ..encoders.card_utils import CardEncoder
        
        card_encoder = CardEncoder()
        tokens = [self.special_tokens['start']]
        
        # Add hole cards
        for card_str in hole_cards:
            if card_str and card_str.strip():
                try:
                    card_idx = card_encoder.card_string_to_index(card_str)
                    tokens.append(card_idx)
                except ValueError:
                    continue
        
        # Separator between hole and community cards
        if community_cards:
            tokens.append(self.special_tokens['sep'])
            
            # Add community cards
            for card_str in community_cards:
                if card_str and card_str.strip():
                    try:
                        card_idx = card_encoder.card_string_to_index(card_str)
                        tokens.append(card_idx)
                    except ValueError:
                        continue
        
        # Separator before action history
        if action_history:
            tokens.append(self.special_tokens['sep'])
            
            # Add action history
            for action in action_history:
                if 0 <= action < self.n_actions:
                    tokens.append(52 + action)  # Action tokens start at 52
        
        return torch.tensor(tokens, dtype=torch.long).unsqueeze(0)  # Add batch dimension
    
    def get_strategy(self, 
                    hole_cards: List[str],
                    community_cards: List[str],
                    action_history: List[int],
                    legal_actions: List[int],
                    temperature: float = 1.0) -> np.ndarray:
        """
        Get strategy (action probabilities) for given state.
        
        Args:
            hole_cards: Hole cards
            community_cards: Community cards
            action_history: Action history
            legal_actions: List of legal action indices
            temperature: Temperature for softmax (higher = more exploration)
            
        Returns:
            Strategy array of shape (n_actions,)
        """
        self.eval()
        
        with torch.no_grad():
            # Encode state
            tokens = self.encode_game_state(hole_cards, community_cards, action_history)
            
            # Forward pass
            _, policy_logits, _ = self.forward(tokens)
            
            # Apply temperature and softmax
            policy_logits = policy_logits.squeeze(0) / temperature
            
            # Mask illegal actions
            mask = torch.full_like(policy_logits, -1e9)
            for action in legal_actions:
                mask[action] = 0
            
            masked_logits = policy_logits + mask
            strategy = F.softmax(masked_logits, dim=-1).cpu().numpy()
            
            return strategy
    
    def get_advantages(self,
                      hole_cards: List[str],
                      community_cards: List[str],
                      action_history: List[int]) -> np.ndarray:
        """
        Get advantage estimates for actions.
        
        Args:
            hole_cards: Hole cards
            community_cards: Community cards  
            action_history: Action history
            
        Returns:
            Advantage array of shape (n_actions,)
        """
        self.eval()
        
        with torch.no_grad():
            # Encode state
            tokens = self.encode_game_state(hole_cards, community_cards, action_history)
            
            # Forward pass
            advantages, _, _ = self.forward(tokens)
            
            return advantages.squeeze(0).cpu().numpy()
    
    def save_checkpoint(self, filepath: str, optimizer_state: Optional[Dict] = None):
        """
        Save model checkpoint.
        
        Args:
            filepath: Path to save checkpoint
            optimizer_state: Optional optimizer state dict
        """
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'model_config': {
                'd_model': self.d_model,
                'n_heads': self.n_heads,
                'n_layers': self.n_layers,
                'n_actions': self.n_actions,
                'max_seq_len': self.max_seq_len
            }
        }
        
        if optimizer_state is not None:
            checkpoint['optimizer_state_dict'] = optimizer_state
            
        torch.save(checkpoint, filepath)
    
    @classmethod
    def load_checkpoint(cls, filepath: str, device: str = 'cpu'):
        """
        Load model from checkpoint.
        
        Args:
            filepath: Path to checkpoint file
            device: Device to load model on
            
        Returns:
            Loaded model instance
        """
        checkpoint = torch.load(filepath, map_location=device)
        
        # Create model with saved config
        model = cls(**checkpoint['model_config'])
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        
        return model, checkpoint.get('optimizer_state_dict')