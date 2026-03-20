#!/usr/bin/env python3
"""
Advanced Neural Personalization Engine for PeatLearn

Implements multiple deep learning models for personalized learning:
- Neural Collaborative Filtering for content recommendations
- LSTM-based learning trajectory modeling  
- Multi-task neural networks for quiz generation
- Deep reinforcement learning for content sequencing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import json
import asyncio

@dataclass
class UserInteraction:
    """Represents a user's interaction with content."""
    user_id: str
    content_id: str
    interaction_type: str  # 'question', 'quiz', 'browse'
    timestamp: datetime
    performance_score: float  # 0-1
    time_spent: float  # seconds
    difficulty_level: float  # 0-1
    topic_tags: List[str]
    context: Dict[str, Any]

@dataclass
class LearningState:
    """Current learning state of a user."""
    user_id: str
    topic_mastery: Dict[str, float]  # topic -> mastery level (0-1)
    learning_velocity: float  # how fast they learn
    preferred_difficulty: float  # 0-1
    learning_style_vector: np.ndarray  # learned embedding
    attention_span: float  # estimated in minutes
    last_active: datetime

class NeuralCollaborativeFilter(nn.Module):
    """
    Neural Collaborative Filtering for content recommendations.
    
    Uses deep learning to model user-content interactions and predict
    which content a user would find most valuable.
    """
    
    def __init__(
        self, 
        num_users: int, 
        num_content: int,
        embedding_dim: int = 128,
        hidden_dims: List[int] = [256, 128, 64]
    ):
        super().__init__()
        
        # Embedding layers
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.content_embedding = nn.Embedding(num_content, embedding_dim)
        
        # Deep neural network layers
        layers = []
        input_dim = embedding_dim * 2
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.BatchNorm1d(hidden_dim)
            ])
            input_dim = hidden_dim
            
        # Output layer for recommendation score
        layers.append(nn.Linear(input_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.deep_layers = nn.Sequential(*layers)
        
        # Initialize embeddings
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.content_embedding.weight, std=0.01)
    
    def forward(self, user_ids: torch.Tensor, content_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass to predict user-content interaction score."""
        user_emb = self.user_embedding(user_ids)
        content_emb = self.content_embedding(content_ids)
        
        # Concatenate embeddings
        x = torch.cat([user_emb, content_emb], dim=-1)
        
        # Pass through deep network
        score = self.deep_layers(x)
        
        return score.squeeze()

class LearningTrajectoryLSTM(nn.Module):
    """
    LSTM-based model for predicting learning trajectories.
    
    Models how a user's understanding evolves over time to predict
    optimal next learning steps.
    """
    
    def __init__(
        self,
        input_dim: int = 64,  # features per interaction
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_topics: int = 50,
        dropout: float = 0.2
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # LSTM for sequence modeling
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers,
            batch_first=True, dropout=dropout
        )
        
        # Attention mechanism for focusing on important interactions
        self.attention = nn.MultiheadAttention(
            hidden_dim, num_heads=8, dropout=dropout
        )
        
        # Output layers for different predictions
        self.mastery_predictor = nn.Linear(hidden_dim, num_topics)
        self.difficulty_predictor = nn.Linear(hidden_dim, 1)
        self.engagement_predictor = nn.Linear(hidden_dim, 1)
        
    def forward(
        self, 
        interaction_sequence: torch.Tensor,
        sequence_lengths: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for learning trajectory prediction.
        
        Args:
            interaction_sequence: [batch_size, max_seq_len, input_dim]
            sequence_lengths: [batch_size] actual lengths of sequences
            
        Returns:
            Dictionary with predictions for mastery, difficulty, engagement
        """
        batch_size = interaction_sequence.size(0)
        
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(interaction_sequence)
        
        # Apply attention to focus on important parts of the sequence
        # Reshape for attention: [seq_len, batch_size, hidden_dim]
        lstm_out_transposed = lstm_out.transpose(0, 1)
        
        attended_out, attention_weights = self.attention(
            lstm_out_transposed, lstm_out_transposed, lstm_out_transposed
        )
        
        # Get the final representation (last time step)
        # Use actual sequence lengths to get the right final state
        final_states = []
        for i, length in enumerate(sequence_lengths):
            final_states.append(attended_out[length-1, i, :])
        
        final_representation = torch.stack(final_states)
        
        # Make predictions
        predictions = {
            'topic_mastery': torch.sigmoid(self.mastery_predictor(final_representation)),
            'optimal_difficulty': torch.sigmoid(self.difficulty_predictor(final_representation)),
            'predicted_engagement': torch.sigmoid(self.engagement_predictor(final_representation)),
            'attention_weights': attention_weights
        }
        
        return predictions

class MultiTaskQuizGenerator(nn.Module):
    """
    Multi-task neural network for quiz generation and difficulty prediction.
    
    Simultaneously learns to:
    1. Generate relevant quiz questions from content
    2. Predict appropriate difficulty level
    3. Estimate user performance
    """
    
    def __init__(
        self,
        content_dim: int = 768,  # BERT/transformer embedding size
        user_dim: int = 128,
        hidden_dim: int = 256,
        num_topics: int = 50,
        vocab_size: int = 50000
    ):
        super().__init__()
        
        # Shared encoder
        self.content_encoder = nn.Sequential(
            nn.Linear(content_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.LayerNorm(hidden_dim)
        )
        
        self.user_encoder = nn.Sequential(
            nn.Linear(user_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.LayerNorm(hidden_dim)
        )
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Task-specific heads
        
        # Task 1: Question Generation (simplified as question type classification)
        self.question_type_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 10)  # 10 question types
        )
        
        # Task 2: Difficulty Prediction
        self.difficulty_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Task 3: Performance Prediction
        self.performance_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Task 4: Topic Relevance
        self.topic_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_topics),
            nn.Sigmoid()
        )
    
    def forward(
        self, 
        content_embedding: torch.Tensor,
        user_embedding: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Multi-task forward pass."""
        
        # Encode inputs
        content_encoded = self.content_encoder(content_embedding)
        user_encoded = self.user_encoder(user_embedding)
        
        # Fuse representations
        fused = self.fusion(torch.cat([content_encoded, user_encoded], dim=-1))
        
        # Multi-task predictions
        outputs = {
            'question_type': self.question_type_head(fused),
            'difficulty': self.difficulty_head(fused),
            'predicted_performance': self.performance_head(fused),
            'topic_relevance': self.topic_head(fused)
        }
        
        return outputs

class AdvancedPersonalizationEngine:
    """
    Main engine coordinating all advanced ML models for personalization.
    """
    
    def __init__(
        self,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        model_save_path: str = 'models/personalization/'
    ):
        self.device = device
        self.model_save_path = model_save_path
        
        # Initialize models (will be loaded or trained)
        self.ncf_model = None
        self.trajectory_model = None
        self.quiz_generator = None
        
        # User and content mappings
        self.user_to_id = {}
        self.content_to_id = {}
        self.id_to_user = {}
        self.id_to_content = {}
        
        # Learning state cache
        self.learning_states = {}
        
        print(f"Advanced Personalization Engine initialized on {device}")
    
    async def initialize_models(
        self,
        num_users: int = 1000,
        num_content: int = 5000,
        num_topics: int = 50
    ):
        """Initialize all neural models."""
        
        # Update numbers based on actual mappings if they exist
        actual_num_users = max(num_users, len(self.user_to_id) + 100) if self.user_to_id else num_users
        actual_num_content = max(num_content, len(self.content_to_id) + 100) if self.content_to_id else num_content
        
        # Neural Collaborative Filtering
        self.ncf_model = NeuralCollaborativeFilter(
            num_users=actual_num_users,
            num_content=actual_num_content,
            embedding_dim=128
        ).to(self.device)
        
        # Learning Trajectory LSTM
        self.trajectory_model = LearningTrajectoryLSTM(
            input_dim=64,
            hidden_dim=128,
            num_topics=num_topics
        ).to(self.device)
        
        # Multi-task Quiz Generator
        self.quiz_generator = MultiTaskQuizGenerator(
            content_dim=768,
            user_dim=128,
            num_topics=num_topics
        ).to(self.device)
        
        print("✅ All neural models initialized")
    
    async def get_content_recommendations(
        self,
        user_id: str,
        num_recommendations: int = 10,
        exclude_seen: bool = True
    ) -> List[Tuple[str, float]]:
        """
        Get personalized content recommendations using Neural Collaborative Filtering.
        
        Returns:
            List of (content_id, predicted_score) tuples
        """
        
        # Initialize mappings if they don't exist
        if user_id not in self.user_to_id:
            self.user_to_id[user_id] = len(self.user_to_id)
            self.id_to_user[len(self.id_to_user)] = user_id
        
        # Create some sample content if none exists
        if not self.content_to_id:
            for i in range(100):  # Sample 100 content items
                content_id = f"content_{i}"
                self.content_to_id[content_id] = i
                self.id_to_content[i] = content_id
        
        # Initialize models if needed, with correct dimensions
        if self.ncf_model is None:
            await self.initialize_models(
                num_users=len(self.user_to_id) + 100,  # Add buffer
                num_content=len(self.content_to_id) + 100  # Add buffer
            )
        
        user_idx = self.user_to_id[user_id]
        
        # Get all content IDs
        all_content_ids = list(range(len(self.content_to_id)))
        
        # Prepare batch prediction - ensure correct data types for embeddings
        user_tensor = torch.tensor([user_idx] * len(all_content_ids), dtype=torch.long, device=self.device)
        content_tensor = torch.tensor(all_content_ids, dtype=torch.long, device=self.device)
        
        # Get predictions
        with torch.no_grad():
            # Ensure indices are within bounds
            if user_idx >= self.ncf_model.user_embedding.num_embeddings:
                # If user index is out of bounds, use index 0
                user_idx = 0
                user_tensor = torch.tensor([user_idx] * len(all_content_ids), dtype=torch.long, device=self.device)
            
            # Check content indices
            max_content_idx = max(all_content_ids) if all_content_ids else 0
            if max_content_idx >= self.ncf_model.content_embedding.num_embeddings:
                # Truncate content IDs to fit embedding size
                valid_content_ids = [i for i in all_content_ids if i < self.ncf_model.content_embedding.num_embeddings]
                if not valid_content_ids:
                    valid_content_ids = [0]  # fallback
                content_tensor = torch.tensor(valid_content_ids, dtype=torch.long, device=self.device)
                user_tensor = torch.tensor([user_idx] * len(valid_content_ids), dtype=torch.long, device=self.device)
                all_content_ids = valid_content_ids
            
            scores = self.ncf_model(user_tensor, content_tensor)
        
        # Convert to content IDs with scores
        recommendations = []
        for content_idx, score in enumerate(scores.cpu().numpy()):
            content_id = self.id_to_content.get(content_idx, f"content_{content_idx}")
            recommendations.append((content_id, float(score)))
        
        # Sort by score and return top recommendations
        recommendations.sort(key=lambda x: x[1], reverse=True)
        
        return recommendations[:num_recommendations]
    
    async def predict_learning_trajectory(
        self,
        user_id: str,
        interaction_history: List[UserInteraction]
    ) -> Dict[str, Any]:
        """
        Predict future learning trajectory using LSTM model.
        
        Returns predictions for topic mastery, optimal difficulty, etc.
        """
        if self.trajectory_model is None:
            await self.initialize_models()
        
        # Convert interaction history to feature vectors
        features = []
        for interaction in interaction_history:
            # Create feature vector from interaction
            feature_vector = [
                interaction.performance_score,
                interaction.time_spent / 3600.0,  # normalize to hours
                interaction.difficulty_level,
                len(interaction.topic_tags),
                # Add more features as needed
            ]
            # Pad to 64 dimensions
            while len(feature_vector) < 64:
                feature_vector.append(0.0)
            features.append(feature_vector[:64])
        
        # Convert to tensor
        sequence = torch.tensor([features], dtype=torch.float32, device=self.device)
        lengths = torch.tensor([len(features)], device=self.device)
        
        # Get predictions
        with torch.no_grad():
            predictions = self.trajectory_model(sequence, lengths)
        
        # Convert to readable format
        result = {
            'topic_mastery': predictions['topic_mastery'][0].cpu().numpy().tolist(),
            'optimal_difficulty': float(predictions['optimal_difficulty'][0].cpu().numpy()),
            'predicted_engagement': float(predictions['predicted_engagement'][0].cpu().numpy()),
            'confidence': 0.85  # placeholder
        }
        
        return result
    
    async def generate_personalized_quiz(
        self,
        user_id: str,
        content_embedding: np.ndarray,
        user_state: LearningState
    ) -> Dict[str, Any]:
        """
        Generate personalized quiz using multi-task neural network.
        """
        if self.quiz_generator is None:
            await self.initialize_models()
        
        # Convert inputs to tensors
        content_tensor = torch.tensor(content_embedding, dtype=torch.float32, device=self.device).unsqueeze(0)
        user_tensor = torch.tensor(user_state.learning_style_vector, dtype=torch.float32, device=self.device).unsqueeze(0)
        
        # Get predictions
        with torch.no_grad():
            outputs = self.quiz_generator(content_tensor, user_tensor)
        
        # Convert to quiz specification
        quiz_spec = {
            'question_type': int(torch.argmax(outputs['question_type'][0]).cpu().numpy()),
            'difficulty': float(outputs['difficulty'][0].cpu().numpy()),
            'predicted_performance': float(outputs['predicted_performance'][0].cpu().numpy()),
            'topic_relevance': outputs['topic_relevance'][0].cpu().numpy().tolist(),
            'recommended_time_limit': 300  # 5 minutes base
        }
        
        return quiz_spec
    
    async def update_user_state(
        self,
        user_id: str,
        interaction: UserInteraction
    ):
        """Update user's learning state based on new interaction."""
        
        if user_id not in self.learning_states:
            # Initialize new user state
            self.learning_states[user_id] = LearningState(
                user_id=user_id,
                topic_mastery={},
                learning_velocity=0.5,
                preferred_difficulty=0.5,
                learning_style_vector=np.random.normal(0, 0.1, 128),
                attention_span=30.0,
                last_active=datetime.now()
            )
        
        state = self.learning_states[user_id]
        
        # Update topic mastery
        for topic in interaction.topic_tags:
            if topic not in state.topic_mastery:
                state.topic_mastery[topic] = 0.0
            
            # Simple update rule (could be made more sophisticated)
            mastery_change = (interaction.performance_score - 0.5) * 0.1
            state.topic_mastery[topic] = np.clip(
                state.topic_mastery[topic] + mastery_change, 0.0, 1.0
            )
        
        # Update learning velocity based on performance
        state.learning_velocity = 0.9 * state.learning_velocity + 0.1 * interaction.performance_score
        
        # Update preferred difficulty
        if interaction.performance_score > 0.8:
            state.preferred_difficulty = min(1.0, state.preferred_difficulty + 0.05)
        elif interaction.performance_score < 0.3:
            state.preferred_difficulty = max(0.0, state.preferred_difficulty - 0.05)
        
        state.last_active = interaction.timestamp
        
        print(f"[STAT] Updated learning state for user {user_id}")
    
    def get_user_analytics(self, user_id: str) -> Dict[str, Any]:
        """Get comprehensive analytics for a user."""
        
        if user_id not in self.learning_states:
            return {"error": "User not found"}
        
        state = self.learning_states[user_id]
        
        analytics = {
            'user_id': user_id,
            'total_topics_engaged': len(state.topic_mastery),
            'average_mastery': np.mean(list(state.topic_mastery.values())) if state.topic_mastery else 0.0,
            'learning_velocity': state.learning_velocity,
            'preferred_difficulty': state.preferred_difficulty,
            'estimated_attention_span': state.attention_span,
            'top_topics': sorted(
                state.topic_mastery.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:5],
            'learning_style_profile': {
                'analytical': float(np.mean(state.learning_style_vector[:43])),
                'practical': float(np.mean(state.learning_style_vector[43:86])),
                'conceptual': float(np.mean(state.learning_style_vector[86:]))
            },
            'last_active': state.last_active.isoformat()
        }
        
        return analytics

# Global instance
personalization_engine = AdvancedPersonalizationEngine()
