#!/usr/bin/env python3
"""
Advanced Knowledge Graph and Embedding System for Ray Peat Content

Implements sophisticated NLP and graph-based approaches:
- Fine-tuned domain-specific embeddings
- Knowledge graph construction and traversal
- Graph Neural Networks for concept relationships
- Hierarchical attention mechanisms
- Dynamic query expansion
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, TransformerConv
from torch_geometric.data import Data, Batch
from transformers import AutoModel, AutoTokenizer, Trainer, TrainingArguments
from sentence_transformers import SentenceTransformer
import networkx as nx
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass
from collections import defaultdict
import json
import asyncio
import re
from pathlib import Path
import pickle

@dataclass
class ConceptNode:
    """Represents a concept in the Ray Peat knowledge graph."""
    concept_id: str
    name: str
    description: str
    concept_type: str  # 'hormone', 'process', 'condition', 'treatment', etc.
    synonyms: List[str]
    importance_score: float  # 0-1
    frequency: int  # how often mentioned in corpus
    embedding: Optional[np.ndarray] = None

@dataclass
class ConceptRelation:
    """Represents a relationship between concepts."""
    source_concept: str
    target_concept: str
    relation_type: str  # 'causes', 'treats', 'inhibits', 'enhances', etc.
    strength: float  # 0-1 confidence in relationship
    evidence_passages: List[str]  # supporting text passages
    
class DomainSpecificEmbedder(nn.Module):
    """
    Fine-tuned embedding model specifically for Ray Peat biomedical content.
    
    Based on BioBERT but further fine-tuned on Ray Peat corpus for 
    domain-specific understanding.
    """
    
    def __init__(
        self,
        base_model: str = "dmis-lab/biobert-base-cased-v1.1",
        embedding_dim: int = 768,
        max_length: int = 512
    ):
        super().__init__()
        
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        self.bert_model = AutoModel.from_pretrained(base_model)
        
        # Add domain-specific projection layers
        self.domain_projection = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
            nn.LayerNorm(embedding_dim)
        )
        
        # Concept-aware attention
        self.concept_attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=12,
            dropout=0.1
        )
        
    def forward(self, input_texts: List[str]) -> torch.Tensor:
        """Generate embeddings for input texts."""
        
        # Tokenize inputs
        encoded = self.tokenizer(
            input_texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Get BERT embeddings
        with torch.no_grad():
            bert_outputs = self.bert_model(**encoded)
            hidden_states = bert_outputs.last_hidden_state
        
        # Apply concept attention
        attended_output, _ = self.concept_attention(
            hidden_states.transpose(0, 1),
            hidden_states.transpose(0, 1),
            hidden_states.transpose(0, 1)
        )
        
        # Pool and project
        pooled_output = attended_output.mean(dim=0)  # Mean pooling
        domain_embeddings = self.domain_projection(pooled_output)
        
        return domain_embeddings

class KnowledgeGraphGNN(nn.Module):
    """
    Graph Neural Network for learning concept relationships in Ray Peat knowledge.
    
    Uses Graph Attention Networks to capture complex concept interactions.
    """
    
    def __init__(
        self,
        node_feature_dim: int = 768,
        hidden_dim: int = 256,
        output_dim: int = 128,
        num_layers: int = 3,
        num_heads: int = 8
    ):
        super().__init__()
        
        self.num_layers = num_layers
        
        # Input projection
        self.input_projection = nn.Linear(node_feature_dim, hidden_dim)
        
        # GAT layers for learning concept relationships
        self.gat_layers = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                self.gat_layers.append(
                    GATConv(hidden_dim, hidden_dim, heads=num_heads, dropout=0.1)
                )
            else:
                self.gat_layers.append(
                    GATConv(hidden_dim * num_heads, hidden_dim, heads=num_heads, dropout=0.1)
                )
        
        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim * num_heads, output_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.LayerNorm(output_dim)
        )
        
        # Relationship prediction head
        self.relation_classifier = nn.Sequential(
            nn.Linear(output_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 10)  # 10 relationship types
        )
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Forward pass through the GNN."""
        
        # Project input features
        h = self.input_projection(x)
        
        # Apply GAT layers
        for i, gat_layer in enumerate(self.gat_layers):
            h = gat_layer(h, edge_index)
            if i < len(self.gat_layers) - 1:
                h = F.relu(h)
                h = F.dropout(h, training=self.training)
        
        # Final projection
        node_embeddings = self.output_projection(h)
        
        return node_embeddings
    
    def predict_relationship(
        self, 
        node_embeddings: torch.Tensor, 
        source_idx: int, 
        target_idx: int
    ) -> torch.Tensor:
        """Predict relationship type between two nodes."""
        
        source_emb = node_embeddings[source_idx]
        target_emb = node_embeddings[target_idx]
        
        # Concatenate embeddings
        relation_input = torch.cat([source_emb, target_emb], dim=-1)
        
        # Predict relationship
        relation_logits = self.relation_classifier(relation_input)
        
        return relation_logits

class HierarchicalAttentionRAG(nn.Module):
    """
    Advanced RAG with hierarchical attention mechanisms.
    
    Applies attention at multiple levels:
    1. Token-level attention within passages
    2. Passage-level attention across retrieved documents
    3. Concept-level attention for knowledge graph integration
    """
    
    def __init__(
        self,
        embedding_dim: int = 768,
        num_attention_heads: int = 12,
        num_passages: int = 10
    ):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.num_passages = num_passages
        
        # Token-level attention
        self.token_attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_attention_heads,
            dropout=0.1
        )
        
        # Passage-level attention
        self.passage_attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_attention_heads // 2,
            dropout=0.1
        )
        
        # Concept-level attention for knowledge graph integration
        self.concept_attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_attention_heads // 2,
            dropout=0.1
        )
        
        # Final fusion layer
        self.fusion_layer = nn.Sequential(
            nn.Linear(embedding_dim * 3, embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.LayerNorm(embedding_dim)
        )
    
    def forward(
        self,
        query_embedding: torch.Tensor,
        passage_embeddings: torch.Tensor,
        concept_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply hierarchical attention to combine query, passages, and concepts.
        
        Args:
            query_embedding: [1, embedding_dim]
            passage_embeddings: [num_passages, embedding_dim]
            concept_embeddings: [num_concepts, embedding_dim]
        """
        
        # Token-level attention (within passages)
        token_attended, _ = self.token_attention(
            query_embedding.unsqueeze(0),
            passage_embeddings.unsqueeze(1),
            passage_embeddings.unsqueeze(1)
        )
        
        # Passage-level attention
        passage_attended, passage_weights = self.passage_attention(
            query_embedding.unsqueeze(0),
            passage_embeddings.unsqueeze(1),
            passage_embeddings.unsqueeze(1)
        )
        
        # Concept-level attention
        concept_attended, concept_weights = self.concept_attention(
            query_embedding.unsqueeze(0),
            concept_embeddings.unsqueeze(1),
            concept_embeddings.unsqueeze(1)
        )
        
        # Fuse all attention outputs
        fused_representation = self.fusion_layer(torch.cat([
            token_attended.squeeze(0),
            passage_attended.squeeze(0),
            concept_attended.squeeze(0)
        ], dim=-1))
        
        return fused_representation, {
            'passage_weights': passage_weights,
            'concept_weights': concept_weights
        }

class AdvancedKnowledgeGraph:
    """
    Advanced knowledge graph system for Ray Peat content.
    
    Builds and maintains a sophisticated graph of biomedical concepts
    with learned relationships and embeddings.
    """
    
    def __init__(
        self,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.device = device
        self.graph = nx.DiGraph()
        
        # Neural models
        self.embedder = None
        self.gnn = None
        self.hierarchical_rag = None
        
        # Concept and relation storage
        self.concepts: Dict[str, ConceptNode] = {}
        self.relations: List[ConceptRelation] = []
        
        # Caching
        self.concept_embeddings_cache = {}
        self.relation_cache = {}
        
        print("🧠 Advanced Knowledge Graph initialized")
    
    async def initialize_models(self):
        """Initialize all neural models."""
        
        # Domain-specific embedder
        self.embedder = DomainSpecificEmbedder().to(self.device)
        
        # Knowledge graph GNN
        self.gnn = KnowledgeGraphGNN().to(self.device)
        
        # Hierarchical attention RAG
        self.hierarchical_rag = HierarchicalAttentionRAG().to(self.device)
        
        print("✅ All knowledge graph models initialized")
    
    async def extract_concepts_from_text(
        self, 
        text: str,
        min_frequency: int = 3
    ) -> List[ConceptNode]:
        """
        Extract biomedical concepts from Ray Peat text using NLP.
        
        Uses domain-specific patterns and medical entity recognition.
        """
        
        # Define biomedical concept patterns specific to Ray Peat
        concept_patterns = {
            'hormones': [
                r'\b(progesterone|estrogen|testosterone|cortisol|thyroid|T3|T4|TSH)\b',
                r'\b(insulin|adrenaline|serotonin|dopamine|GABA)\b'
            ],
            'processes': [
                r'\b(metabolism|oxidation|inflammation|stress response)\b',
                r'\b(glycolysis|oxidative phosphorylation|lipid peroxidation)\b'
            ],
            'conditions': [
                r'\b(hypothyroidism|diabetes|depression|anxiety|fatigue)\b',
                r'\b(autoimmune|cancer|cardiovascular disease)\b'
            ],
            'treatments': [
                r'\b(aspirin|vitamin E|coconut oil|orange juice|milk)\b',
                r'\b(light therapy|heat therapy|dietary intervention)\b'
            ]
        }
        
        extracted_concepts = []
        concept_frequencies = defaultdict(int)
        
        # Extract concepts using patterns
        for concept_type, patterns in concept_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    concept_name = match.group().lower()
                    concept_frequencies[concept_name] += 1
        
        # Create concept nodes for frequent concepts
        for concept_name, frequency in concept_frequencies.items():
            if frequency >= min_frequency:
                concept_id = f"concept_{len(self.concepts)}"
                
                concept_node = ConceptNode(
                    concept_id=concept_id,
                    name=concept_name,
                    description=f"Biomedical concept: {concept_name}",
                    concept_type=self._determine_concept_type(concept_name),
                    synonyms=self._find_synonyms(concept_name),
                    importance_score=min(1.0, frequency / 100.0),
                    frequency=frequency
                )
                
                extracted_concepts.append(concept_node)
                self.concepts[concept_id] = concept_node
        
        return extracted_concepts
    
    def _determine_concept_type(self, concept_name: str) -> str:
        """Determine the type of a biomedical concept."""
        
        hormone_keywords = ['progesterone', 'estrogen', 'testosterone', 'cortisol', 'thyroid']
        process_keywords = ['metabolism', 'oxidation', 'inflammation', 'stress']
        condition_keywords = ['hypothyroidism', 'diabetes', 'depression', 'fatigue']
        treatment_keywords = ['aspirin', 'vitamin', 'therapy', 'diet']
        
        concept_lower = concept_name.lower()
        
        if any(keyword in concept_lower for keyword in hormone_keywords):
            return 'hormone'
        elif any(keyword in concept_lower for keyword in process_keywords):
            return 'process'
        elif any(keyword in concept_lower for keyword in condition_keywords):
            return 'condition'
        elif any(keyword in concept_lower for keyword in treatment_keywords):
            return 'treatment'
        else:
            return 'general'
    
    def _find_synonyms(self, concept_name: str) -> List[str]:
        """Find synonyms for a concept (simplified implementation)."""
        
        synonym_map = {
            'progesterone': ['P4'],
            'thyroid': ['T3', 'T4', 'thyroxine', 'triiodothyronine'],
            'estrogen': ['estradiol', 'E2'],
            'cortisol': ['hydrocortisone', 'stress hormone'],
            'metabolism': ['metabolic rate', 'energy production'],
            'inflammation': ['inflammatory response', 'immune activation']
        }
        
        return synonym_map.get(concept_name.lower(), [])
    
    async def extract_relationships(
        self,
        text: str,
        concepts: List[ConceptNode]
    ) -> List[ConceptRelation]:
        """
        Extract relationships between concepts from text.
        
        Uses pattern matching and dependency parsing to identify
        relationships like "X causes Y", "X treats Y", etc.
        """
        
        relationship_patterns = {
            'causes': [
                r'(\w+)\s+(?:causes?|leads? to|results? in|triggers?)\s+(\w+)',
                r'(\w+)\s+(?:is responsible for|brings about)\s+(\w+)'
            ],
            'treats': [
                r'(\w+)\s+(?:treats?|cures?|helps with|ameliorates?)\s+(\w+)',
                r'(\w+)\s+(?:is effective for|therapy for)\s+(\w+)'
            ],
            'inhibits': [
                r'(\w+)\s+(?:inhibits?|blocks?|suppresses?|reduces?)\s+(\w+)',
                r'(\w+)\s+(?:prevents?|interferes with)\s+(\w+)'
            ],
            'enhances': [
                r'(\w+)\s+(?:enhances?|improves?|boosts?|increases?)\s+(\w+)',
                r'(\w+)\s+(?:promotes?|stimulates?|activates?)\s+(\w+)'
            ]
        }
        
        concept_names = {concept.name.lower(): concept.concept_id for concept in concepts}
        extracted_relations = []
        
        for relation_type, patterns in relationship_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                
                for match in matches:
                    source_term = match.group(1).lower()
                    target_term = match.group(2).lower()
                    
                    # Check if both terms are recognized concepts
                    if source_term in concept_names and target_term in concept_names:
                        relation = ConceptRelation(
                            source_concept=concept_names[source_term],
                            target_concept=concept_names[target_term],
                            relation_type=relation_type,
                            strength=0.7,  # Default confidence
                            evidence_passages=[match.group(0)]
                        )
                        
                        extracted_relations.append(relation)
                        self.relations.append(relation)
        
        return extracted_relations
    
    async def build_graph_from_corpus(self, corpus_texts: List[str]):
        """Build knowledge graph from entire Ray Peat corpus."""
        
        print("🔨 Building knowledge graph from corpus...")
        
        all_concepts = []
        all_relations = []
        
        # Process each document
        for i, text in enumerate(corpus_texts):
            if i % 50 == 0:
                print(f"Processing document {i+1}/{len(corpus_texts)}")
            
            # Extract concepts
            concepts = await self.extract_concepts_from_text(text)
            all_concepts.extend(concepts)
            
            # Extract relationships
            relations = await self.extract_relationships(text, concepts)
            all_relations.extend(relations)
        
        # Add nodes to NetworkX graph
        for concept in all_concepts:
            self.graph.add_node(
                concept.concept_id,
                name=concept.name,
                concept_type=concept.concept_type,
                importance=concept.importance_score,
                frequency=concept.frequency
            )
        
        # Add edges to NetworkX graph
        for relation in all_relations:
            self.graph.add_edge(
                relation.source_concept,
                relation.target_concept,
                relation_type=relation.relation_type,
                strength=relation.strength
            )
        
        print(f"OK: Knowledge graph built: {len(self.concepts)} concepts, {len(self.relations)} relations")
    
    async def generate_concept_embeddings(self):
        """Generate embeddings for all concepts using domain-specific embedder."""
        
        if self.embedder is None:
            await self.initialize_models()
        
        concept_texts = []
        concept_ids = []
        
        for concept_id, concept in self.concepts.items():
            # Create rich text description for embedding
            text = f"{concept.name}. {concept.description}. "
            if concept.synonyms:
                text += f"Also known as: {', '.join(concept.synonyms)}. "
            
            concept_texts.append(text)
            concept_ids.append(concept_id)
        
        # Generate embeddings in batches
        batch_size = 32
        all_embeddings = []
        
        for i in range(0, len(concept_texts), batch_size):
            batch_texts = concept_texts[i:i+batch_size]
            
            with torch.no_grad():
                batch_embeddings = self.embedder(batch_texts)
                all_embeddings.append(batch_embeddings.cpu().numpy())
        
        # Store embeddings
        all_embeddings = np.vstack(all_embeddings)
        
        for i, concept_id in enumerate(concept_ids):
            self.concepts[concept_id].embedding = all_embeddings[i]
            self.concept_embeddings_cache[concept_id] = all_embeddings[i]
        
        print(f"OK: Generated embeddings for {len(concept_ids)} concepts")
    
    async def find_related_concepts(
        self,
        query_concept: str,
        max_concepts: int = 10,
        relation_types: Optional[List[str]] = None
    ) -> List[Tuple[str, str, float]]:
        """
        Find concepts related to a query concept using graph traversal.
        
        Returns:
            List of (concept_id, relation_type, strength) tuples
        """
        
        related_concepts = []
        
        if query_concept not in self.graph:
            return []
        
        # Direct neighbors
        for neighbor in self.graph.neighbors(query_concept):
            edge_data = self.graph[query_concept][neighbor]
            relation_type = edge_data.get('relation_type', 'unknown')
            strength = edge_data.get('strength', 0.5)
            
            if relation_types is None or relation_type in relation_types:
                related_concepts.append((neighbor, relation_type, strength))
        
        # Second-order neighbors (concepts connected to direct neighbors)
        for neighbor in self.graph.neighbors(query_concept):
            for second_neighbor in self.graph.neighbors(neighbor):
                if second_neighbor != query_concept:
                    edge_data = self.graph[neighbor][second_neighbor]
                    relation_type = edge_data.get('relation_type', 'unknown')
                    strength = edge_data.get('strength', 0.5) * 0.7  # Decay for distance
                    
                    if relation_types is None or relation_type in relation_types:
                        related_concepts.append((second_neighbor, f"indirect_{relation_type}", strength))
        
        # Sort by strength and return top concepts
        related_concepts.sort(key=lambda x: x[2], reverse=True)
        return related_concepts[:max_concepts]
    
    async def expand_query_with_concepts(
        self,
        original_query: str,
        max_expansions: int = 5
    ) -> Dict[str, Any]:
        """
        Expand a user query with related concepts from the knowledge graph.
        
        This enables more comprehensive retrieval by including semantically
        related concepts that the user might not have explicitly mentioned.
        """
        
        # Extract concepts from original query
        query_concepts = await self.extract_concepts_from_text(original_query, min_frequency=1)
        
        expanded_terms = set()
        expansion_sources = {}
        
        for concept in query_concepts:
            # Find related concepts
            related = await self.find_related_concepts(
                concept.concept_id,
                max_concepts=max_expansions
            )
            
            for related_concept_id, relation_type, strength in related:
                if related_concept_id in self.concepts:
                    related_concept = self.concepts[related_concept_id]
                    expanded_terms.add(related_concept.name)
                    expansion_sources[related_concept.name] = {
                        'source_concept': concept.name,
                        'relation_type': relation_type,
                        'strength': strength
                    }
        
        # Create expanded query
        expanded_query = original_query
        if expanded_terms:
            expanded_query += " " + " ".join(expanded_terms)
        
        return {
            'original_query': original_query,
            'expanded_query': expanded_query,
            'expansion_terms': list(expanded_terms),
            'expansion_sources': expansion_sources,
            'num_expansions': len(expanded_terms)
        }
    
    def save_knowledge_graph(self, filepath: str):
        """Save the knowledge graph to disk."""
        
        graph_data = {
            'concepts': {cid: {
                'concept_id': c.concept_id,
                'name': c.name,
                'description': c.description,
                'concept_type': c.concept_type,
                'synonyms': c.synonyms,
                'importance_score': c.importance_score,
                'frequency': c.frequency,
                'embedding': c.embedding.tolist() if c.embedding is not None else None
            } for cid, c in self.concepts.items()},
            'relations': [{
                'source_concept': r.source_concept,
                'target_concept': r.target_concept,
                'relation_type': r.relation_type,
                'strength': r.strength,
                'evidence_passages': r.evidence_passages
            } for r in self.relations],
            'graph_edges': list(self.graph.edges(data=True))
        }
        
        with open(filepath, 'w') as f:
            json.dump(graph_data, f, indent=2)
        
        print(f"Saved: Knowledge graph saved to {filepath}")
    
    def get_graph_analytics(self) -> Dict[str, Any]:
        """Get comprehensive analytics about the knowledge graph."""
        
        analytics = {
            'num_concepts': len(self.concepts),
            'num_relations': len(self.relations),
            'num_edges': self.graph.number_of_edges(),
            'graph_density': nx.density(self.graph),
            'average_degree': sum(dict(self.graph.degree()).values()) / len(self.graph) if len(self.graph) > 0 else 0,
            'concept_types': {},
            'relation_types': {},
            'top_concepts_by_importance': [],
            'top_concepts_by_frequency': []
        }
        
        # Concept type distribution
        for concept in self.concepts.values():
            concept_type = concept.concept_type
            analytics['concept_types'][concept_type] = analytics['concept_types'].get(concept_type, 0) + 1
        
        # Relation type distribution
        for relation in self.relations:
            relation_type = relation.relation_type
            analytics['relation_types'][relation_type] = analytics['relation_types'].get(relation_type, 0) + 1
        
        # Top concepts
        sorted_by_importance = sorted(self.concepts.values(), key=lambda x: x.importance_score, reverse=True)
        analytics['top_concepts_by_importance'] = [
            {'name': c.name, 'importance': c.importance_score} 
            for c in sorted_by_importance[:10]
        ]
        
        sorted_by_frequency = sorted(self.concepts.values(), key=lambda x: x.frequency, reverse=True)
        analytics['top_concepts_by_frequency'] = [
            {'name': c.name, 'frequency': c.frequency} 
            for c in sorted_by_frequency[:10]
        ]
        
        return analytics

# Global knowledge graph instance
ray_peat_knowledge_graph = AdvancedKnowledgeGraph()
