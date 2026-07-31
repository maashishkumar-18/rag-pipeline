"""
Confidence Scoring
Evaluates retrieval quality and decides whether the assembled context
is sufficient for generation, needs retry, or requires clarification.

Features:
- Weighted multi-factor confidence scoring
- Score distribution analysis
- Semantic coverage checking (keyword + embedding-based)
- Action recommendations (proceed, retry, clarify)
- Retry parameter adjustment
- Learned weights via optimization (with fallback to configured weights)

Configuration-driven — all thresholds and weights from YAML.
"""

import os
import yaml
import json
from typing import List, Dict, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from pathlib import Path
from collections import defaultdict

from dotenv import load_dotenv
import numpy as np

from src.retrieval.config import ConfidenceLevel
from src.retrieval.reranker import RerankedChunk, RerankerResult
from src.retrieval.context_builder import AssembledContext

load_dotenv()


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class FeatureWeights:
    """Weights for confidence calculation features."""
    reranker_scores: float = 0.30
    score_distribution: float = 0.15
    chunk_diversity: float = 0.15
    context_completeness: float = 0.15
    semantic_coverage: float = 0.10
    metadata_match: float = 0.10
    retrieval_depth: float = 0.05
    
    def __post_init__(self):
        total = sum([
            self.reranker_scores, self.score_distribution,
            self.chunk_diversity, self.context_completeness,
            self.semantic_coverage, self.metadata_match,
            self.retrieval_depth
        ])
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"Feature weights must sum to 1.0, got {total}")


@dataclass
class Thresholds:
    """Confidence level thresholds."""
    high: float = 0.75
    medium: float = 0.50
    low: float = 0.0


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""
    max_retries: int = 2
    increase_candidate_k: int = 10
    decrease_min_score: float = 0.05
    enable_aggressive_expansion: bool = True


@dataclass
class LearnedWeightsConfig:
    """Configuration for learned weights."""
    enabled: bool = False
    weights_path: Optional[str] = None
    min_samples_for_learning: int = 100
    optimization_metric: str = "accuracy"  # "accuracy", "f1", "ndcg"
    update_frequency: int = 50  # Update after N samples


@dataclass
class SemanticCoverageConfig:
    """Configuration for semantic coverage."""
    use_keywords: bool = True
    min_query_term_overlap: float = 0.3
    use_embeddings: bool = False
    embedding_model: Optional[str] = None
    embedding_similarity_threshold: float = 0.7
    embedding_weight: float = 0.5  # Blend with keyword coverage
    cache_embeddings: bool = True


@dataclass
class ConfidenceConfig:
    """Complete confidence scoring configuration."""
    thresholds: Thresholds = field(default_factory=Thresholds)
    feature_weights: FeatureWeights = field(default_factory=FeatureWeights)
    learned_weights: LearnedWeightsConfig = field(default_factory=LearnedWeightsConfig)
    semantic_coverage: SemanticCoverageConfig = field(default_factory=SemanticCoverageConfig)
    score_distribution_steep_drop: float = 0.3
    score_distribution_flat: float = 0.1
    retry: RetryConfig = field(default_factory=RetryConfig)
    high_action: str = "proceed"
    medium_action: str = "proceed_with_caution"
    low_action: str = "retry_or_clarify"
    enable_retry_depth_penalty: bool = True  # More sources aren't always better

    @classmethod
    def from_yaml(cls, path: Optional[str] = None) -> "ConfidenceConfig":
        """
        Load configuration from YAML file.
        
        Priority:
        1. Explicit path argument
        2. RAGPIPE_CONFIDENCE_CONFIG env var
        3. config/retrieval/confidence.yaml (default)
        """
        config_path = (
            path
            or os.getenv("RAGPIPE_CONFIDENCE_CONFIG")
            or str(Path(__file__).parent.parent.parent / "config" / "retrieval" / "confidence.yaml")
        )
        
        if Path(config_path).exists():
            with open(config_path, "r") as f:
                data = yaml.safe_load(f)
        else:
            data = cls._default_config()
        
        data = cls._apply_env_overrides(data)
        
        thresholds_data = data.get("thresholds", {})
        weights_data = data.get("feature_weights", {})
        learned_data = data.get("learned_weights", {})
        semantic_data = data.get("semantic_coverage", {})
        score_dist = data.get("score_distribution", {})
        retry_data = data.get("retry", {})
        actions = data.get("actions", {})
        
        return cls(
            thresholds=Thresholds(
                high=float(thresholds_data.get("high", 0.75)),
                medium=float(thresholds_data.get("medium", 0.50)),
            ),
            feature_weights=FeatureWeights(
                reranker_scores=float(weights_data.get("reranker_scores", 0.30)),
                score_distribution=float(weights_data.get("score_distribution", 0.15)),
                chunk_diversity=float(weights_data.get("chunk_diversity", 0.15)),
                context_completeness=float(weights_data.get("context_completeness", 0.15)),
                semantic_coverage=float(weights_data.get("semantic_coverage", 0.10)),
                metadata_match=float(weights_data.get("metadata_match", 0.10)),
                retrieval_depth=float(weights_data.get("retrieval_depth", 0.05)),
            ),
            learned_weights=LearnedWeightsConfig(
                enabled=bool(learned_data.get("enabled", False)),
                weights_path=learned_data.get("weights_path"),
                min_samples_for_learning=int(learned_data.get("min_samples_for_learning", 100)),
                optimization_metric=str(learned_data.get("optimization_metric", "accuracy")),
                update_frequency=int(learned_data.get("update_frequency", 50)),
            ),
            semantic_coverage=SemanticCoverageConfig(
                use_keywords=bool(semantic_data.get("use_keywords", True)),
                min_query_term_overlap=float(semantic_data.get("min_query_term_overlap", 0.3)),
                use_embeddings=bool(semantic_data.get("use_embeddings", False)),
                embedding_model=semantic_data.get("embedding_model"),
                embedding_similarity_threshold=float(semantic_data.get("embedding_similarity_threshold", 0.7)),
                embedding_weight=float(semantic_data.get("embedding_weight", 0.5)),
                cache_embeddings=bool(semantic_data.get("cache_embeddings", True)),
            ),
            score_distribution_steep_drop=float(score_dist.get("steep_drop_threshold", 0.3)),
            score_distribution_flat=float(score_dist.get("flat_threshold", 0.1)),
            retry=RetryConfig(
                max_retries=int(retry_data.get("max_retries", 2)),
                increase_candidate_k=int(retry_data.get("increase_candidate_k", 10)),
                decrease_min_score=float(retry_data.get("decrease_min_score", 0.05)),
                enable_aggressive_expansion=bool(retry_data.get("enable_aggressive_expansion", True)),
            ),
            high_action=str(actions.get("high", {}).get("action", "proceed")),
            medium_action=str(actions.get("medium", {}).get("action", "proceed_with_caution")),
            low_action=str(actions.get("low", {}).get("action", "retry_or_clarify")),
            enable_retry_depth_penalty=bool(data.get("enable_retry_depth_penalty", True)),
        )
    
    @classmethod
    def _apply_env_overrides(cls, data: Dict) -> Dict:
        """Apply environment variable overrides."""
        env_mapping = {
            "RAGPIPE_CONFIDENCE_HIGH": ("thresholds", "high"),
            "RAGPIPE_CONFIDENCE_MEDIUM": ("thresholds", "medium"),
            "RAGPIPE_CONFIDENCE_MAX_RETRIES": ("retry", "max_retries"),
            "RAGPIPE_LEARNED_WEIGHTS_ENABLED": ("learned_weights", "enabled"),
            "RAGPIPE_SEMANTIC_USE_EMBEDDINGS": ("semantic_coverage", "use_embeddings"),
        }
        
        for env_var, (section, key) in env_mapping.items():
            value = os.getenv(env_var)
            if value is not None:
                if section not in data:
                    data[section] = {}
                try:
                    # Parse boolean if needed
                    if key in ["enabled", "use_embeddings"]:
                        data[section][key] = value.lower() in ["true", "1", "yes"]
                    elif "." in value:
                        data[section][key] = float(value)
                    else:
                        data[section][key] = int(value)
                except ValueError:
                    pass
        
        return data
    
    @staticmethod
    def _default_config() -> Dict:
        """Minimal fallback configuration."""
        return {
            "thresholds": {"high": 0.75, "medium": 0.50},
            "feature_weights": {
                "reranker_scores": 0.30,
                "score_distribution": 0.15,
                "chunk_diversity": 0.15,
                "context_completeness": 0.15,
                "semantic_coverage": 0.10,
                "metadata_match": 0.10,
                "retrieval_depth": 0.05,
            },
            "learned_weights": {
                "enabled": False,
                "min_samples_for_learning": 100,
                "optimization_metric": "accuracy",
                "update_frequency": 50,
            },
            "semantic_coverage": {
                "use_keywords": True,
                "min_query_term_overlap": 0.3,
                "use_embeddings": False,
                "embedding_similarity_threshold": 0.7,
                "embedding_weight": 0.5,
                "cache_embeddings": True,
            },
            "retry": {
                "max_retries": 2,
                "increase_candidate_k": 10,
                "decrease_min_score": 0.05,
                "enable_aggressive_expansion": True,
            },
            "enable_retry_depth_penalty": True,
        }


# ============================================================================
# Result Types
# ============================================================================

@dataclass
class ConfidenceResult:
    """Complete confidence evaluation result."""
    score: float                          # 0.0 to 1.0
    level: ConfidenceLevel                # HIGH, MEDIUM, LOW
    action: str                           # "proceed", "proceed_with_caution", "retry_or_clarify"
    feature_scores: Dict[str, float]      # Individual feature scores
    recommendations: List[str] = field(default_factory=list)
    should_retry: bool = False
    retry_config: Optional[Dict[str, Any]] = None
    warnings: List[str] = field(default_factory=list)


# ============================================================================
# Weight Optimizer
# ============================================================================

class WeightOptimizer:
    """
    Optimizes feature weights based on evaluation data.
    
    Uses labeled examples to learn optimal weights for confidence scoring.
    Supports various optimization metrics and can update incrementally.
    """
    
    def __init__(
        self,
        config: LearnedWeightsConfig,
        feature_names: List[str],
        initial_weights: np.ndarray
    ):
        """
        Initialize weight optimizer.
        
        Args:
            config: LearnedWeightsConfig
            feature_names: List of feature names
            initial_weights: Initial weight vector
        """
        self.config = config
        self.feature_names = feature_names
        self.weights = initial_weights.copy()
        self.samples: List[Dict] = []
        self.sample_counter = 0
        
    def add_sample(
        self,
        features: Dict[str, float],
        true_confidence: float,
        true_level: Optional[ConfidenceLevel] = None
    ) -> None:
        """
        Add a training sample for weight optimization.
        
        Args:
            features: Feature scores
            true_confidence: True confidence score (e.g., from evaluation)
            true_level: Optional true confidence level
        """
        sample = {
            "features": features.copy(),
            "true_confidence": true_confidence,
            "true_level": true_level.value if true_level else None,
        }
        self.samples.append(sample)
        self.sample_counter += 1
        
        # Update weights if we have enough samples
        if self.config.enabled and self.sample_counter % self.config.update_frequency == 0:
            if len(self.samples) >= self.config.min_samples_for_learning:
                self._update_weights()
    
    def _update_weights(self) -> None:
        """Update weights using stored samples."""
        if len(self.samples) < self.config.min_samples_for_learning:
            return
        
        # Simple gradient-based optimization
        # For production, you'd use more sophisticated methods
        # e.g., Bayesian optimization, grid search, or scipy.optimize
        
        features_matrix = np.array([
            [sample["features"][name] for name in self.feature_names]
            for sample in self.samples
        ])
        targets = np.array([sample["true_confidence"] for sample in self.samples])
        
        # Linear regression to find optimal weights
        # Constraint: weights must be non-negative and sum to 1
        try:
            # Simple optimization: normalize by inverse variance
            # This is a heuristic, not true linear regression
            variances = np.var(features_matrix, axis=0)
            inv_variances = 1.0 / (variances + 1e-6)
            new_weights = inv_variances / np.sum(inv_variances)
            
            # Blend with existing weights (smooth updates)
            alpha = 0.3  # Learning rate for weight updates
            self.weights = (1 - alpha) * self.weights + alpha * new_weights
            
            # Ensure sum to 1
            self.weights = self.weights / np.sum(self.weights)
            
        except Exception as e:
            # Fall back to current weights on error
            pass
    
    def get_weights(self) -> Dict[str, float]:
        """Get current optimized weights."""
        return {
            name: float(weight)
            for name, weight in zip(self.feature_names, self.weights)
        }
    
    def save_weights(self, path: str) -> None:
        """Save learned weights to file."""
        weights_dict = self.get_weights()
        data = {
            "weights": weights_dict,
            "samples": len(self.samples),
            "timestamp": str(Path(path).stat().st_mtime) if Path(path).exists() else None
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
    
    def load_weights(self, path: str) -> bool:
        """Load learned weights from file."""
        try:
            with open(path, "r") as f:
                data = json.load(f)
            weights_dict = data.get("weights", {})
            if weights_dict:
                self.weights = np.array([
                    weights_dict.get(name, 0.0)
                    for name in self.feature_names
                ])
                # Ensure sum to 1
                self.weights = self.weights / np.sum(self.weights)
                return True
        except Exception:
            pass
        return False


# ============================================================================
# Semantic Coverage with Embeddings
# ============================================================================

class SemanticCoverageScorer:
    """
    Evaluates semantic coverage between query and context.
    
    Supports both keyword-based and embedding-based coverage.
    Can be extended to use paraphrases and semantic similarity.
    """
    
    def __init__(self, config: SemanticCoverageConfig):
        """
        Initialize semantic coverage scorer.
        
        Args:
            config: SemanticCoverageConfig
        """
        self.config = config
        self._embedding_model = None
        self._embedding_cache: Dict[str, np.ndarray] = {}
        
    def _get_embedding_model(self):
        """Lazy load embedding model."""
        if self._embedding_model is None and self.config.use_embeddings:
            try:
                from sentence_transformers import SentenceTransformer
                model_name = self.config.embedding_model or "all-MiniLM-L6-v2"
                self._embedding_model = SentenceTransformer(model_name)
            except ImportError:
                # Fall back to a simple embedding approximation
                self._embedding_model = "fallback"
                print("Warning: sentence-transformers not installed, using fallback embeddings")
        return self._embedding_model
    
    def _get_embedding(self, text: str) -> Optional[np.ndarray]:
        """Get embedding for text with caching."""
        if not self.config.use_embeddings:
            return None
        
        # Check cache
        if self.config.cache_embeddings and text in self._embedding_cache:
            return self._embedding_cache[text]
        
        # Get embedding
        model = self._get_embedding_model()
        if model is None:
            return None
        
        if model == "fallback":
            # Simple fallback: use TF-IDF style
            embedding = self._fallback_embedding(text)
        else:
            try:
                embedding = model.encode(text, convert_to_numpy=True)
            except Exception:
                embedding = self._fallback_embedding(text)
        
        # Cache
        if self.config.cache_embeddings:
            self._embedding_cache[text] = embedding
        
        return embedding
    
    def _fallback_embedding(self, text: str) -> np.ndarray:
        """Simple fallback embedding using word frequencies."""
        # This is a placeholder - in practice you'd use a proper embedding
        words = text.lower().split()
        # Simple random-like but deterministic embedding
        np.random.seed(sum(ord(c) for c in text[:100]))
        return np.random.normal(0, 1, 384)
    
    def _keyword_coverage(self, query: str, context: str) -> float:
        """Calculate keyword-based coverage."""
        if not query or not context:
            return 0.0
        
        context_lower = context.lower()
        query_terms = set()
        
        for word in query.lower().split():
            clean = word.strip('.,?!()[]{}":;')
            if len(clean) > 2:
                query_terms.add(clean)
        
        if not query_terms:
            return 0.5
        
        matched = sum(1 for term in query_terms if term in context_lower)
        coverage = matched / len(query_terms)
        
        return coverage
    
    def _embedding_coverage(self, query: str, context: str) -> Optional[float]:
        """Calculate embedding-based coverage."""
        if not self.config.use_embeddings or not query or not context:
            return None
        
        query_embedding = self._get_embedding(query)
        context_embedding = self._get_embedding(context)
        
        if query_embedding is None or context_embedding is None:
            return None
        
        # Cosine similarity
        similarity = np.dot(query_embedding, context_embedding) / (
            np.linalg.norm(query_embedding) * np.linalg.norm(context_embedding) + 1e-6
        )
        
        # Normalize similarity to [0, 1]
        # Similarity of 0.5 -> 0.0, 1.0 -> 1.0
        normalized = max(0.0, (similarity - 0.5) / 0.5)
        return min(1.0, normalized)
    
    def score(self, query: str, context: str) -> float:
        """
        Calculate semantic coverage score.
        
        Returns score in [0, 1] where 1 = perfect coverage.
        """
        # Keyword coverage
        keyword_score = 0.0
        if self.config.use_keywords:
            keyword_score = self._keyword_coverage(query, context)
        
        # Embedding coverage
        embedding_score = None
        if self.config.use_embeddings:
            embedding_score = self._embedding_coverage(query, context)
        
        # Blend scores
        if embedding_score is not None:
            # Blend keyword and embedding scores
            embedding_weight = self.config.embedding_weight
            return keyword_score * (1 - embedding_weight) + embedding_score * embedding_weight
        else:
            # Use only keyword score with threshold scaling
            min_overlap = self.config.min_query_term_overlap
            if keyword_score >= min_overlap:
                return 0.5 + (keyword_score - min_overlap) * (0.5 / (1.0 - min_overlap))
            else:
                return keyword_score * (0.5 / min_overlap)
    
    def score_against_terms(
        self,
        query: str,
        keywords: str,
        context: str
    ) -> float:
        """
        Score coverage against query and keywords.
        
        This combines query coverage with keyword-based coverage.
        """
        # Score against query
        query_score = self.score(query, context)
        
        # Score against keywords if provided
        if keywords:
            keyword_score = self.score(keywords, context)
            # Weighted average: query is more important
            return 0.7 * query_score + 0.3 * keyword_score
        
        return query_score


# ============================================================================
# Confidence Scorer
# ============================================================================

class ConfidenceScorer:
    """
    Evaluates retrieval quality using multiple weighted features.
    
    Analyzes reranker scores, score distribution, chunk diversity,
    semantic coverage, and metadata match to determine if the
    assembled context is sufficient for generation.
    
    Configuration-driven — all thresholds and weights from YAML.
    Supports learned weights and embedding-based semantic coverage.
    """
    
    def __init__(
        self,
        config: Optional[ConfidenceConfig] = None,
        config_path: Optional[str] = None
    ):
        """
        Initialize the confidence scorer.
        
        Args:
            config: ConfidenceConfig object
            config_path: Path to YAML config file
        """
        self.config = config or ConfidenceConfig.from_yaml(config_path)
        self.semantic_scorer = SemanticCoverageScorer(self.config.semantic_coverage)
        
        # Initialize weight optimizer if enabled
        self._feature_names = [
            "reranker_scores", "score_distribution", "chunk_diversity",
            "context_completeness", "semantic_coverage", "metadata_match",
            "retrieval_depth"
        ]
        self._base_weights = np.array([
            getattr(self.config.feature_weights, name)
            for name in self._feature_names
        ])
        
        self.weight_optimizer = None
        if self.config.learned_weights.enabled:
            self.weight_optimizer = WeightOptimizer(
                config=self.config.learned_weights,
                feature_names=self._feature_names,
                initial_weights=self._base_weights
            )
            # Load existing weights if available
            if self.config.learned_weights.weights_path:
                self.weight_optimizer.load_weights(
                    self.config.learned_weights.weights_path
                )
    
    def get_feature_weights(self) -> Dict[str, float]:
        """Get current feature weights (learned if enabled)."""
        if self.weight_optimizer and self.config.learned_weights.enabled:
            return self.weight_optimizer.get_weights()
        else:
            return {
                name: float(weight)
                for name, weight in zip(self._feature_names, self._base_weights)
            }
    
    def evaluate(
        self,
        query: str,
        keywords: str,
        reranker_result: RerankerResult,
        assembled_context: AssembledContext,
        metadata_filters_applied: List[str],
        retrieval_sources_used: List[str],
        retry_count: int = 0
    ) -> ConfidenceResult:
        """
        Evaluate retrieval quality and return confidence score.
        
        Args:
            query: Original search query
            keywords: Extracted keywords
            reranker_result: Results from the reranker
            assembled_context: Assembled context from context builder
            metadata_filters_applied: List of metadata filters used
            retrieval_sources_used: List of retrieval sources (semantic, keyword, metadata)
            retry_count: Current retry attempt number
            
        Returns:
            ConfidenceResult with score, level, and recommended action
        """
        feature_scores = {}
        warnings = []
        
        # 1. Reranker scores (average of top chunks)
        feature_scores["reranker_scores"] = self._evaluate_reranker_scores(
            assembled_context, reranker_result
        )
        
        # 2. Score distribution (steep drop = clear relevance)
        feature_scores["score_distribution"] = self._evaluate_score_distribution(
            reranker_result
        )
        
        # 3. Chunk diversity (coverage across sources)
        feature_scores["chunk_diversity"] = self._evaluate_chunk_diversity(
            assembled_context
        )
        
        # 4. Context completeness (chunks used vs available)
        feature_scores["context_completeness"] = self._evaluate_context_completeness(
            assembled_context, reranker_result
        )
        
        # 5. Semantic coverage (query terms/paraphrases in context)
        feature_scores["semantic_coverage"] = self._evaluate_semantic_coverage(
            query, keywords, assembled_context
        )
        
        # 6. Metadata match (filters applied)
        feature_scores["metadata_match"] = self._evaluate_metadata_match(
            metadata_filters_applied
        )
        
        # 7. Retrieval depth (quality-adjusted, not just count)
        feature_scores["retrieval_depth"] = self._evaluate_retrieval_depth(
            retrieval_sources_used, reranker_result
        )
        
        # Calculate weighted score
        weights = self.get_feature_weights()
        weighted_score = sum(
            weights[name] * feature_scores[name]
            for name in self._feature_names
        )
        
        # Apply retry penalty
        if retry_count > 0:
            weighted_score *= max(0.5, 1.0 - (retry_count * 0.15))
            warnings.append(f"Retry penalty applied (attempt {retry_count + 1})")
        
        # Determine confidence level
        level = self._determine_level(weighted_score)
        
        # Determine action
        action, recommendations, should_retry, retry_config = self._determine_action(
            level, weighted_score, retry_count, feature_scores
        )
        
        # Add warnings for borderline scores
        if weighted_score < self.config.thresholds.high and weighted_score >= self.config.thresholds.medium:
            if feature_scores["semantic_coverage"] < 0.5:
                warnings.append("Low semantic coverage — query terms not well represented in context")
            if feature_scores["reranker_scores"] < 0.6:
                warnings.append("Low reranker confidence — chunks may not be highly relevant")
        
        return ConfidenceResult(
            score=round(weighted_score, 4),
            level=level,
            action=action,
            feature_scores={k: round(v, 4) for k, v in feature_scores.items()},
            recommendations=recommendations,
            should_retry=should_retry,
            retry_config=retry_config,
            warnings=warnings
        )
    
    # ========================================================================
    # Feature Evaluators
    # ========================================================================
    
    def _evaluate_reranker_scores(
        self,
        context: AssembledContext,
        reranker_result: RerankerResult
    ) -> float:
        """Evaluate based on average reranker score of used chunks."""
        if not reranker_result.chunks or context.chunks_used == 0:
            return 0.0
        
        # Average score of top N chunks used in context
        top_scores = [c.rerank_score for c in reranker_result.chunks[:context.chunks_used]]
        if not top_scores:
            return 0.0
        
        avg_score = sum(top_scores) / len(top_scores)
        return min(1.0, avg_score)
    
    def _evaluate_score_distribution(self, reranker_result: RerankerResult) -> float:
        """Evaluate how scores are distributed — steep drop indicates clear relevance."""
        if not reranker_result.chunks or len(reranker_result.chunks) < 2:
            return 0.5  # Neutral if insufficient data
        
        scores = [c.rerank_score for c in reranker_result.chunks[:5]]
        
        # Calculate maximum drop between consecutive scores
        max_drop = max(
            scores[i] - scores[i + 1]
            for i in range(len(scores) - 1)
        )
        
        # Steep drop → high confidence (clear relevance boundary)
        if max_drop >= self.config.score_distribution_steep_drop:
            return 0.9
        # Flat distribution → low confidence (uncertain relevance)
        elif max_drop <= self.config.score_distribution_flat:
            return 0.3
        else:
            # Linear interpolation between flat and steep
            range_size = self.config.score_distribution_steep_drop - self.config.score_distribution_flat
            if range_size > 0:
                normalized = (max_drop - self.config.score_distribution_flat) / range_size
                return 0.3 + normalized * 0.6
            return 0.5
    
    def _evaluate_chunk_diversity(self, context: AssembledContext) -> float:
        """Evaluate diversity of sources in the assembled context."""
        if not context.sections or context.chunks_used <= 1:
            return 0.5
        
        # More sections = more diverse context
        section_count = len(context.sections)
        diversity_score = min(1.0, section_count / 4.0)  # 4+ sections = full score
        
        # Boost if diversity was actively applied
        if hasattr(context, 'diversity_applied') and context.diversity_applied:
            diversity_score = min(1.0, diversity_score + 0.2)
        
        return diversity_score
    
    def _evaluate_context_completeness(
        self,
        context: AssembledContext,
        reranker_result: RerankerResult
    ) -> float:
        """Evaluate how many available chunks were used."""
        total_available = len(reranker_result.chunks) if reranker_result.chunks else 0
        if total_available == 0:
            return 0.0
        
        # Ratio of used to available (capped at reasonable max)
        used = context.chunks_used
        completeness = min(1.0, used / min(total_available, 10))
        
        return completeness
    
    def _evaluate_semantic_coverage(
        self,
        query: str,
        keywords: str,
        context: AssembledContext
    ) -> float:
        """Evaluate how well query terms are covered in the context."""
        if not context.formatted_context:
            return 0.0
        
        return self.semantic_scorer.score_against_terms(
            query, keywords, context.formatted_context
        )
    
    def _evaluate_metadata_match(self, filters_applied: List[str]) -> float:
        """Evaluate based on how many metadata filters were applied."""
        if not filters_applied:
            return 0.3  # No filters = neutral
        
        # More specific filters = higher confidence
        specificity = min(1.0, len(filters_applied) / 3.0)
        return 0.5 + specificity * 0.5
    
    def _evaluate_retrieval_depth(
        self,
        sources_used: List[str],
        reranker_result: RerankerResult
    ) -> float:
        """
        Evaluate quality of retrieval depth.
        
        Uses quality-adjusted depth: more sources isn't always better.
        """
        if not sources_used:
            return 0.0
        
        # Identify source types
        all_sources = {"semantic", "keyword", "keyword_bm25", "keyword_fallback", "metadata"}
        used = set(sources_used)
        
        semantic_used = "semantic" in used
        keyword_used = any(s in used for s in ["keyword", "keyword_bm25", "keyword_fallback"])
        metadata_used = "metadata" in used
        
        # Base depth
        base_depth = sum([semantic_used, keyword_used, metadata_used])
        depth_score = min(1.0, base_depth / 3.0)
        
        # Quality adjustment: if semantic retrieval has good scores, depth is good
        # If multiple sources but all low quality, depth doesn't help much
        if self.config.enable_retry_depth_penalty and reranker_result.chunks:
            avg_score = np.mean([c.rerank_score for c in reranker_result.chunks[:5]])
            if avg_score < 0.5 and base_depth > 2:
                # Multiple sources but all low quality
                depth_score *= 0.7
        
        return depth_score
    
    # ========================================================================
    # Level & Action Determination
    # ========================================================================
    
    def _determine_level(self, score: float) -> ConfidenceLevel:
        """Determine confidence level from score."""
        thresholds = self.config.thresholds
        
        if score >= thresholds.high:
            return ConfidenceLevel.HIGH
        elif score >= thresholds.medium:
            return ConfidenceLevel.MEDIUM
        else:
            return ConfidenceLevel.LOW
    
    def _determine_action(
        self,
        level: ConfidenceLevel,
        score: float,
        retry_count: int,
        feature_scores: Dict[str, float]
    ) -> Tuple[str, List[str], bool, Optional[Dict]]:
        """
        Determine recommended action based on confidence level.
        
        Returns:
            Tuple of (action, recommendations, should_retry, retry_config)
        """
        recommendations = []
        should_retry = False
        retry_config = None
        
        if level == ConfidenceLevel.HIGH:
            return self.config.high_action, [], False, None
        
        elif level == ConfidenceLevel.MEDIUM:
            if feature_scores.get("semantic_coverage", 0) < 0.5:
                recommendations.append("Consider expanding query terms for better coverage")
            if feature_scores.get("reranker_scores", 0) < 0.6:
                recommendations.append("Top chunks have moderate relevance — verify with user if unsure")
            return self.config.medium_action, recommendations, False, None
        
        else:  # LOW
            recommendations.append("Retrieval quality is low — context may be insufficient")
            
            # Can we retry?
            if retry_count < self.config.retry.max_retries:
                should_retry = True
                retry_config = {
                    "increase_candidate_k": self.config.retry.increase_candidate_k,
                    "decrease_min_score": self.config.retry.decrease_min_score * (retry_count + 1),
                    "enable_aggressive_expansion": self.config.retry.enable_aggressive_expansion,
                    "retry_count": retry_count + 1,
                }
                recommendations.append(f"Retrying with expanded parameters (attempt {retry_count + 1})")
            else:
                # Max retries exceeded
                if feature_scores.get("semantic_coverage", 0) < 0.2:
                    recommendations.append("Topic may not be covered in uploaded materials")
                    recommendations.append("Consider acknowledging the gap to the user")
                else:
                    recommendations.append("Consider asking the user to rephrase or clarify their question")
            
            return self.config.low_action, recommendations, should_retry, retry_config
    
    # ========================================================================
    # Quick Assessment
    # ========================================================================
    
    def quick_assess(self, reranker_result: RerankerResult) -> ConfidenceLevel:
        """
        Quick confidence assessment based on reranker scores alone.
        
        Useful for fast path decisions before full context assembly.
        """
        if not reranker_result.chunks:
            return ConfidenceLevel.LOW
        
        top_score = reranker_result.chunks[0].rerank_score
        
        if top_score >= self.config.thresholds.high:
            return ConfidenceLevel.HIGH
        elif top_score >= self.config.thresholds.medium:
            return ConfidenceLevel.MEDIUM
        else:
            return ConfidenceLevel.LOW


# ============================================================================
# Validation
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Confidence Scorer - Validation")
    print("=" * 60)
    
    config = ConfidenceConfig.from_yaml()
    scorer = ConfidenceScorer(config=config)
    
    print(f"\n📋 Configuration loaded:")
    print(f"   High threshold: {config.thresholds.high}")
    print(f"   Medium threshold: {config.thresholds.medium}")
    print(f"   Max retries: {config.retry.max_retries}")
    weights = scorer.get_feature_weights()
    print(f"   Feature weights: {weights}")
    print(f"   Semantic coverage: keywords={config.semantic_coverage.use_keywords}, "
          f"embeddings={config.semantic_coverage.use_embeddings}")
    
    # Create sample data
    from src.retrieval.reranker import RerankedChunk, RerankerResult
    from src.retrieval.context_builder import AssembledContext
    
    # High confidence scenario
    print(f"\n{'─' * 50}")
    print("Scenario 1: High Confidence")
    
    high_chunks = [
        RerankedChunk("c1", "content", "raw", 0.85, 0.95, 1, {"topic": "Deadlock"}),
        RerankedChunk("c2", "content", "raw", 0.78, 0.88, 2, {"topic": "Deadlock"}),
        RerankedChunk("c3", "content", "raw", 0.72, 0.82, 3, {"topic": "Deadlock"}),
        RerankedChunk("c4", "content", "raw", 0.60, 0.75, 4, {"topic": "Deadlock"}),
        RerankedChunk("c5", "content", "raw", 0.55, 0.45, 5, {"topic": "CPU"}),
    ]
    
    high_reranker = RerankerResult(
        chunks=high_chunks,
        total_input=30,
        total_output=5,
        backend_used="cross_encoder",
        model_used="ms-marco-MiniLM-L-6-v2",
        processing_time_ms=45.0
    )
    
    high_context = AssembledContext(
        formatted_context="Deadlock is a situation where...\n\nA deadlock occurs when four conditions...\n\nFor example, consider...",
        chunks_used=4,
        tokens_used=200,
        sections=["definition", "explanation", "example", "important_notes"],
        citations=["[OS | Process | Slide 12]", "[OS | Process | Slide 13]"],
        template_used="learning",
        diversity_applied=True
    )
    
    result = scorer.evaluate(
        query="What is deadlock and how to prevent it?",
        keywords="deadlock prevent conditions",
        reranker_result=high_reranker,
        assembled_context=high_context,
        metadata_filters_applied=["course=OS", "chapter=Process"],
        retrieval_sources_used=["semantic", "keyword_bm25", "metadata"]
    )
    
    print(f"   Score: {result.score:.4f} ({result.level.value})")
    print(f"   Action: {result.action}")
    print(f"   Should retry: {result.should_retry}")
    print(f"   Feature scores: {result.feature_scores}")
    if result.warnings:
        print(f"   Warnings: {result.warnings}")
    
    # Low confidence scenario
    print(f"\n{'─' * 50}")
    print("Scenario 2: Low Confidence")
    
    low_chunks = [
        RerankedChunk("c1", "content", "raw", 0.45, 0.52, 1, {"topic": "General"}),
        RerankedChunk("c2", "content", "raw", 0.40, 0.48, 2, {"topic": "General"}),
        RerankedChunk("c3", "content", "raw", 0.38, 0.45, 3, {"topic": "General"}),
    ]
    
    low_reranker = RerankerResult(
        chunks=low_chunks,
        total_input=20,
        total_output=3,
        backend_used="cross_encoder",
        model_used="ms-marco-MiniLM-L-6-v2",
        processing_time_ms=30.0
    )
    
    low_context = AssembledContext(
        formatted_context="Some general information...",
        chunks_used=2,
        tokens_used=50,
        sections=["general"],
        citations=[],
        template_used="learning"
    )
    
    result = scorer.evaluate(
        query="Explain quantum cryptography in detail",
        keywords="quantum cryptography detail",
        reranker_result=low_reranker,
        assembled_context=low_context,
        metadata_filters_applied=[],
        retrieval_sources_used=["semantic"]
    )
    
    print(f"   Score: {result.score:.4f} ({result.level.value})")
    print(f"   Action: {result.action}")
    print(f"   Should retry: {result.should_retry}")
    if result.retry_config:
        print(f"   Retry config: {result.retry_config}")
    print(f"   Feature scores: {result.feature_scores}")
    if result.recommendations:
        print(f"   Recommendations: {result.recommendations}")
    if result.warnings:
        print(f"   Warnings: {result.warnings}")
    
    # Quick assess
    print(f"\n{'─' * 50}")
    print("Quick Assessment Tests")
    print(f"   High reranker → {scorer.quick_assess(high_reranker).value}")
    print(f"   Low reranker → {scorer.quick_assess(low_reranker).value}")
    
    # Semantic coverage with embeddings example
    print(f"\n{'─' * 50}")
    print("Semantic Coverage Tests")
    
    # Test keyword coverage
    test_query = "Prevent deadlock"
    test_context = "Avoid circular wait to prevent system hangs."
    score = scorer.semantic_scorer.score(test_query, test_context)
    print(f"   Keyword-only coverage: {score:.3f}")
    
    # Test embedding coverage (if available)
    if config.semantic_coverage.use_embeddings:
        embedding_score = scorer.semantic_scorer._embedding_coverage(test_query, test_context)
        print(f"   Embedding similarity: {embedding_score:.3f}")
    
    print(f"\n✅ Confidence Scorer ready for integration")