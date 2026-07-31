"""
Intent Router
Decides whether a query should be handled by deterministic rules
or delegated to the Agent layer for reasoning.

Decision logic:
- Rule rewriter returns needs_agent=False → go straight to retrieval
- Rule rewriter returns needs_agent=True → route to Agent
- Low confidence on rule rewrite → route to Agent
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum

from src.retrieval.query_rewriter import QueryRewriter, ConversationState


class RoutingDecision(str, Enum):
    """Where the query should be routed."""
    DIRECT_RETRIEVAL = "direct_retrieval"    # Rules handled it, go to hybrid search
    AGENT_REASONING = "agent_reasoning"       # Agent needs to understand query
    CLARIFICATION = "clarification"           # Ask user to clarify


@dataclass
class RoutingResult:
    """Result of the routing decision."""
    decision: RoutingDecision
    rewritten_query: str                     # Cleaned query (from rewriter)
    metadata: Dict[str, Any]                 # Full rewrite metadata
    agent_context: Dict[str, Any]            # Context to pass to Agent if needed
    
    @property
    def needs_agent(self) -> bool:
        return self.decision == RoutingDecision.AGENT_REASONING
    
    @property
    def needs_clarification(self) -> bool:
        return self.decision == RoutingDecision.CLARIFICATION


class IntentRouter:
    """
    Routes queries between deterministic rewriter and Agent layer.
    Thin decision layer — no complex logic.
    """
    
    def __init__(self, rewriter: Optional[QueryRewriter] = None):
        """
        Initialize the router.
        
        Args:
            rewriter: QueryRewriter instance. Creates default if None.
        """
        self.rewriter = rewriter or QueryRewriter()
        
        # Confidence threshold: below this, route to Agent
        self.agent_confidence_threshold = 0.5
        
        # If query is very short and no context → clarification
        self.clarification_min_length = 2
    
    def route(self, query: str, state: ConversationState) -> RoutingResult:
        """
        Process a query through the rewriter and decide routing.
        
        Args:
            query: Raw user query
            state: Current conversation state
            
        Returns:
            RoutingResult with decision and context
        """
        # Step 1: Run through lightweight rewriter
        rewritten, metadata = self.rewriter.rewrite(query, state)
        
        # Step 2: Decide routing
        
        # Clarification needed: very short query, no context
        if (len(query.split()) <= self.clarification_min_length 
            and not state.get_primary_concept()
            and not metadata.get("was_rewritten", False)):
            return RoutingResult(
                decision=RoutingDecision.CLARIFICATION,
                rewritten_query=query,
                metadata=metadata,
                agent_context={}
            )
        
        # Agent needed: rewriter explicitly flagged it
        if metadata.get("needs_agent", False):
            return RoutingResult(
                decision=RoutingDecision.AGENT_REASONING,
                rewritten_query=rewritten,
                metadata=metadata,
                agent_context=self._build_agent_context(query, rewritten, metadata, state)
            )
        
        # Agent needed: low confidence despite rewrite
        if (metadata.get("rewrite_confidence", 1.0) < self.agent_confidence_threshold
            and state.get_primary_concept()):
            return RoutingResult(
                decision=RoutingDecision.AGENT_REASONING,
                rewritten_query=rewritten,
                metadata=metadata,
                agent_context=self._build_agent_context(query, rewritten, metadata, state)
            )
        
        # Default: direct retrieval
        return RoutingResult(
            decision=RoutingDecision.DIRECT_RETRIEVAL,
            rewritten_query=rewritten,
            metadata=metadata,
            agent_context={}
        )
    
    def _build_agent_context(
        self,
        original_query: str,
        rewritten_query: str,
        metadata: Dict[str, Any],
        state: ConversationState
    ) -> Dict[str, Any]:
        """
        Build the context packet to pass to the Agent layer.
        """
        return {
            "original_query": original_query,
            "rewritten_query": rewritten_query,
            "rewrite_metadata": metadata,
            "conversation_hierarchy": {
                "subject": state.subject,
                "module": state.module,
                "chapter": state.chapter,
                "current_topic": state.current_topic,
                "current_concept": state.current_concept,
                "current_subconcept": state.current_subconcept,
            },
            "last_compared_pair": state.last_compared_pair,
            "recent_concepts": state.get_recent_concepts(3),
            "context_summary": state.get_context_summary(),
            "agent_reason": metadata.get("agent_reason", "Low confidence or novel query"),
        }


# ============================================================================
# Validation
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Intent Router - Validation")
    print("=" * 60)
    
    router = IntentRouter()
    
    state = ConversationState(
        session_id="test",
        subject="Computer Science",
        module="Operating Systems",
        chapter="Process Management",
        current_topic="Concurrency",
        current_concept="Deadlock"
    )
    
    state.add_turn("user", "What is deadlock?")
    state.add_turn("assistant", "Deadlock is a situation where...",
                   topic="Concurrency", concept="Deadlock")
    
    test_queries = [
        "Why?",
        "Give example",
        "Can this happen in real life?",
        "Make it suitable for a 5-mark answer",
        "What is virtual memory and how does it relate to paging?",
    ]
    
    for query in test_queries:
        result = router.route(query, state)
        
        emoji = {
            RoutingDecision.DIRECT_RETRIEVAL: "⚡",
            RoutingDecision.AGENT_REASONING: "🤖",
            RoutingDecision.CLARIFICATION: "❓",
        }
        
        print(f"\n{emoji[result.decision]} {result.decision.value.upper()}")
        print(f"   Query: \"{query}\"")
        print(f"   Rewritten: \"{result.rewritten_query}\"")
        print(f"   Confidence: {result.metadata.get('rewrite_confidence', 'N/A')}")
        if result.needs_agent:
            print(f"   Agent Reason: {result.agent_context.get('agent_reason', 'N/A')}")