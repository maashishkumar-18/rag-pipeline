"""
Retrieval Agent
LLM-powered reasoning layer for queries the rule-based rewriter cannot handle.

Responsibilities:
- Understand novel/unpredictable follow-up questions
- Handle exam-specific requests (marks-based, interview style, etc.)
- Decompose complex multi-part queries
- Detect ambiguity and request clarification
- Select appropriate retrieval strategy and pipeline
- Generate optimized search queries from conversation context

The Agent receives the full conversation state and returns a structured
retrieval plan that the orchestrator executes.
"""

import os
import json
import concurrent.futures
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum

from dotenv import load_dotenv

from src.common.llm_client import simple_generate
from src.retrieval.query_rewriter import ConversationState, IntentType

load_dotenv()

# Use DeepSeek's standard chat model for cost efficiency
AGENT_MODEL = "deepseek-chat"


# ============================================================================
# Agent Output Types
# ============================================================================

class AgentAction(str, Enum):
    """Actions the agent can decide to take."""
    REWRITE_AND_RETRIEVE = "rewrite_and_retrieve"    # Rewrite query and proceed
    DECOMPOSE = "decompose"                           # Break into sub-queries
    CLARIFY = "clarify"                               # Ask user for clarification
    SWITCH_PIPELINE = "switch_pipeline"               # Use a different pipeline
    DIRECT_RETRIEVE = "direct_retrieve"               # Query is fine as-is


class RetrievalStrategy(str, Enum):
    """Retrieval strategies the agent can select."""
    SEMANTIC = "semantic"              # Pure semantic search
    HYBRID = "hybrid"                  # Semantic + keyword + metadata
    METADATA_HEAVY = "metadata_heavy"  # Prioritize metadata filtering
    KEYWORD_HEAVY = "keyword_heavy"    # Prioritize keyword matching


@dataclass
class SubQuery:
    """A single sub-query from decomposition."""
    order: int
    query: str
    intent: str
    retrieval_strategy: RetrievalStrategy = RetrievalStrategy.HYBRID
    metadata_filters: Dict[str, str] = field(default_factory=dict)


@dataclass
class AgentDecision:
    """
    Structured output from the retrieval agent.
    This tells the orchestrator exactly what to do.
    """
    action: AgentAction
    reasoning: str                              # Why the agent made this decision
    
    # For REWRITE_AND_RETRIEVE / DIRECT_RETRIEVE
    search_query: str = ""
    retrieval_strategy: RetrievalStrategy = RetrievalStrategy.HYBRID
    metadata_filters: Dict[str, str] = field(default_factory=dict)
    target_pipeline: str = "learning"           # Which pipeline config to use
    
    # For DECOMPOSE
    sub_queries: List[SubQuery] = field(default_factory=list)
    
    # For CLARIFY
    clarification_question: str = ""
    
    # For SWITCH_PIPELINE
    suggested_pipeline: str = ""
    
    # Confidence
    confidence: float = 0.8
    
    # What the agent understood
    understood_intent: str = ""
    understood_topic: str = ""


# ============================================================================
# Retrieval Agent
# ============================================================================

class RetrievalAgent:
    """
    LLM-powered agent that reasons about how to retrieve information
    for queries the rule-based rewriter cannot handle.

    Uses DeepSeek for cost-efficient reasoning.
    """

    def __init__(self, api_key: Optional[str] = None, model: str = AGENT_MODEL):
        """
        Initialize the retrieval agent.

        Args:
            api_key: DeepSeek API key. If None, loads from env.
            model: DeepSeek model to use for reasoning
        """
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise ValueError("DEEPSEEK_API_KEY is required.")

        self.model_name = model

        # Single worker used to enforce a hard wall-clock timeout on
        # simple_generate, which has no built-in deadline of its own beyond
        # its internal HTTP timeout.
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="retrieval-agent"
        )

    def reason(
        self,
        query: str,
        rewritten_query: str,
        agent_context: Dict[str, Any],
        state: ConversationState,
        timeout_seconds: Optional[float] = None
    ) -> AgentDecision:
        """
        Analyze a query and decide the optimal retrieval approach.

        Args:
            query: Original user query
            rewritten_query: Best attempt from rule-based rewriter
            agent_context: Context packet from IntentRouter
            state: Full conversation state
            timeout_seconds: Max time to wait for the LLM call. If exceeded,
                falls back to direct retrieval instead of blocking the caller's
                own retrieval budget.

        Returns:
            AgentDecision with action and retrieval plan
        """
        # Build the reasoning prompt
        prompt = self._build_reasoning_prompt(query, rewritten_query, agent_context, state)

        try:
            if timeout_seconds is not None:
                future = self._executor.submit(simple_generate, prompt, self.model_name)
                response_text = future.result(timeout=timeout_seconds)
            else:
                response_text = simple_generate(prompt, self.model_name)
            decision = self._parse_response(response_text, query)
            return decision

        except concurrent.futures.TimeoutError:
            return AgentDecision(
                action=AgentAction.DIRECT_RETRIEVE,
                reasoning=f"Agent reasoning exceeded {timeout_seconds:.1f}s budget, falling back to direct retrieval",
                search_query=rewritten_query or query,
                retrieval_strategy=RetrievalStrategy.HYBRID,
                confidence=0.4,
                understood_intent="unknown",
                understood_topic=state.get_primary_concept() or "unknown"
            )

        except Exception as e:
            # Fallback: direct retrieval with the rewritten query
            return AgentDecision(
                action=AgentAction.DIRECT_RETRIEVE,
                reasoning=f"Agent error, falling back to direct retrieval: {e}",
                search_query=rewritten_query or query,
                retrieval_strategy=RetrievalStrategy.HYBRID,
                confidence=0.5,
                understood_intent="unknown",
                understood_topic=state.get_primary_concept() or "unknown"
            )
    
    def _build_reasoning_prompt(
        self,
        query: str,
        rewritten_query: str,
        agent_context: Dict[str, Any],
        state: ConversationState
    ) -> str:
        """Build the structured reasoning prompt for the agent."""
        
        # Format conversation history (last 5 turns)
        history_text = ""
        for turn in list(state.history)[-6:]:
            role_emoji = "🧑" if turn.role == "user" else "🤖"
            history_text += f"{role_emoji} [{turn.role}]: {turn.content[:200]}\n"
        
        # Format hierarchy
        hierarchy = agent_context.get("conversation_hierarchy", {})
        hierarchy_text = "\n".join(
            f"  {k}: {v}" for k, v in hierarchy.items() if v
        )
        
        prompt = f"""You are a retrieval planning agent for an AI study/exam-prep assistant.

## Your Job
Analyze the student's query and decide the BEST way to retrieve information from their study materials.

## Conversation Context
Current topic hierarchy:
{hierarchy_text or '  No active topic'}

Recent conversation:
{history_text or '  No history'}

## Student's Query
Original: "{query}"
Rule-based rewrite attempt: "{rewritten_query}"
Rewrite was: {"successful" if query != rewritten_query else "unchanged"}
Reason agent was invoked: {agent_context.get('agent_reason', 'Novel query')}

## Available Actions
1. **rewrite_and_retrieve**: Rewrite the query for better retrieval, then search
2. **decompose**: Break into multiple sub-queries (for complex/multi-part questions)
3. **clarify**: Ask the student a clarifying question (if intent is truly unclear)
4. **switch_pipeline**: Use a different retrieval pipeline (planner/revision/qa/quiz)
5. **direct_retrieve**: Query is fine, just search as-is

## Available Retrieval Strategies
- **semantic**: Pure vector similarity search
- **hybrid**: Semantic + keyword + metadata combined
- **metadata_heavy**: Prioritize filters (topic, chapter, document type)
- **keyword_heavy**: Prioritize exact term matching

## Available Pipelines
- **learning**: For explaining concepts (default)
- **planner**: For study planning, syllabus, weightage
- **revision**: For concise summaries, quick review
- **qa**: For previous year questions, exam-style answers
- **quiz**: For generating practice questions

## Important Rules
1. If the student asks about exam patterns, marks, PYQs, MCQs → use `rewrite_and_retrieve` with appropriate metadata filters
2. If the query is multi-part ("Compare X and Y. Give examples. Advantages.") → use `decompose`
3. If the query references "it", "this", "that" and context is clear → use `rewrite_and_retrieve` (resolve the reference)
4. If the intent is genuinely ambiguous AND no context helps → use `clarify`
5. If the student wants a study plan → `switch_pipeline` to planner
6. If the student wants revision notes/flashcards → `switch_pipeline` to revision
7. For analogies, real-world examples, "explain like I'm 5" → `rewrite_and_retrieve` (add those modifiers to search)
8. For "5-mark answer", "exam perspective", "interview style" → `rewrite_and_retrieve` with appropriate metadata filters

## Response Format
Return ONLY a valid JSON object. No markdown, no code blocks, no explanation.

{{
  "action": "rewrite_and_retrieve|decompose|clarify|switch_pipeline|direct_retrieve",
  "reasoning": "Brief explanation of your decision",
  "search_query": "Optimized search query (for rewrite/direct actions)",
  "retrieval_strategy": "semantic|hybrid|metadata_heavy|keyword_heavy",
  "metadata_filters": {{}},
  "target_pipeline": "learning|planner|revision|qa|quiz",
  "sub_queries": [
    {{"order": 1, "query": "...", "intent": "...", "retrieval_strategy": "hybrid", "metadata_filters": {{}}}}
  ],
  "clarification_question": "",
  "suggested_pipeline": "",
  "confidence": 0.0,
  "understood_intent": "What the student is trying to do",
  "understood_topic": "The topic/concept this is about"
}}
"""
        return prompt
    
    def _parse_response(self, response_text: str, original_query: str) -> AgentDecision:
        """Parse the DeepSeek response into an AgentDecision."""
        
        # Clean response
        text = response_text.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()
        
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            # Try to extract JSON
            import re
            match = re.search(r'\{.*\}', text, re.DOTALL)
            if match:
                data = json.loads(match.group())
            else:
                # Fallback
                return AgentDecision(
                    action=AgentAction.DIRECT_RETRIEVE,
                    reasoning="Failed to parse agent response",
                    search_query=original_query,
                    confidence=0.3
                )
        
        # Parse sub-queries if decomposing
        sub_queries = []
        for sq in data.get("sub_queries", []):
            sub_queries.append(SubQuery(
                order=sq.get("order", 0),
                query=sq.get("query", ""),
                intent=sq.get("intent", "general"),
                retrieval_strategy=RetrievalStrategy(sq.get("retrieval_strategy", "hybrid")),
                metadata_filters=sq.get("metadata_filters", {})
            ))
        
        # Parse action
        action_str = data.get("action", "direct_retrieve")
        try:
            action = AgentAction(action_str)
        except ValueError:
            action = AgentAction.DIRECT_RETRIEVE
        
        return AgentDecision(
            action=action,
            reasoning=data.get("reasoning", ""),
            search_query=data.get("search_query", original_query),
            retrieval_strategy=RetrievalStrategy(data.get("retrieval_strategy", "hybrid")),
            metadata_filters=data.get("metadata_filters", {}),
            target_pipeline=data.get("target_pipeline", "learning"),
            sub_queries=sub_queries,
            clarification_question=data.get("clarification_question", ""),
            suggested_pipeline=data.get("suggested_pipeline", ""),
            confidence=float(data.get("confidence", 0.7)),
            understood_intent=data.get("understood_intent", ""),
            understood_topic=data.get("understood_topic", ""),
        )


# ============================================================================
# Validation
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Retrieval Agent - Validation")
    print("=" * 60)
    
    agent = RetrievalAgent()
    
    # Setup conversation
    state = ConversationState(
        session_id="test",
        subject="Computer Science",
        module="Operating Systems",
        chapter="Process Management",
        current_topic="Concurrency",
        current_concept="Deadlock"
    )
    
    state.add_turn("user", "What is deadlock?")
    state.add_turn("assistant", "Deadlock is a situation where two or more processes are blocked indefinitely, each waiting for a resource held by another.",
                   topic="Concurrency", concept="Deadlock")
    
    agent_context = {
        "agent_reason": "Novel follow-up query",
        "conversation_hierarchy": {
            "subject": "Computer Science",
            "module": "Operating Systems",
            "chapter": "Process Management",
            "current_topic": "Concurrency",
            "current_concept": "Deadlock",
            "current_subconcept": None
        },
        "last_compared_pair": None,
        "recent_concepts": ["Deadlock"],
        "context_summary": "Topic: Concurrency | Concept: Deadlock"
    }
    
    test_queries = [
        ("Can this happen in real life?", "Can this happen in real life?"),
        ("Make it suitable for a 5-mark answer", "Make it suitable for a 5-mark answer"),
        ("Which one would you choose for an embedded system and why?", "Which one would you choose for an embedded system and why?"),
        ("Connect this with database transactions", "Connect this with database transactions"),
        ("Give me an analogy that a 10-year-old would understand", "Give me an analogy that a 10-year-old would understand"),
    ]
    
    for original, rewritten in test_queries:
        print(f"\n{'─' * 50}")
        print(f"📝 Query: \"{original}\"")
        
        decision = agent.reason(original, rewritten, agent_context, state)
        
        print(f"🎯 Action: {decision.action.value}")
        print(f"🧠 Reasoning: {decision.reasoning}")
        print(f"📊 Confidence: {decision.confidence:.2f}")
        print(f"💡 Understood as: {decision.understood_intent}")
        print(f"📚 Topic: {decision.understood_topic}")
        
        if decision.action == AgentAction.REWRITE_AND_RETRIEVE:
            print(f"🔍 Search query: \"{decision.search_query}\"")
            print(f"📂 Strategy: {decision.retrieval_strategy.value}")
            if decision.metadata_filters:
                print(f"🏷️  Filters: {decision.metadata_filters}")
        
        elif decision.action == AgentAction.DECOMPOSE:
            for sq in decision.sub_queries:
                print(f"   [{sq.order}] {sq.query}")
                print(f"       Intent: {sq.intent}, Strategy: {sq.retrieval_strategy.value}")
        
        elif decision.action == AgentAction.CLARIFY:
            print(f"❓ Clarification: \"{decision.clarification_question}\"")
        
        elif decision.action == AgentAction.SWITCH_PIPELINE:
            print(f"🔀 Switch to: {decision.suggested_pipeline}")