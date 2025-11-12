"""
Advanced Query Understanding Module
Provides complex query parsing, entity extraction, intent recognition, and query expansion
for improved semantic video search
"""

import re
import logging
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass
from datetime import datetime, timedelta
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class QueryIntent:
    """Structured representation of query intent"""
    primary_intent: str  # 'object_search', 'action_search', 'attribute_search', 'temporal_search'
    entities: List[str]  # Extracted entities (objects, people, etc.)
    attributes: List[str]  # Colors, sizes, materials, etc.
    actions: List[str]  # Verbs describing activities
    temporal_context: Optional[Dict[str, any]] = None  # Time-related filters
    spatial_context: Optional[str] = None  # Location information
    modifiers: List[str] = None  # Adjectives and modifiers
    negations: List[str] = None  # Things to exclude
    confidence: float = 1.0


@dataclass
class ExpandedQuery:
    """Query with expansions and variations"""
    original_query: str
    expanded_queries: List[str]  # Synonym-based expansions
    structured_intent: QueryIntent
    target_namespaces: List[str]  # Inferred search namespaces
    metadata_filters: Dict[str, any]  # Metadata-based filters
    boost_terms: List[str]  # Terms to boost in ranking


class QueryUnderstanding:
    """Advanced query understanding and expansion system"""
    
    def __init__(self):
        # Object taxonomy and synonyms
        self.object_synonyms = {
            'bag': ['bag', 'backpack', 'bagpack', 'bookbag', 'rucksack', 'knapsack', 'pack', 'satchel'],
            'duffel': ['duffel', 'duffle', 'duffel bag', 'duffle bag', 'gym bag', 'sports bag', 'travel bag'],
            'laptop': ['laptop', 'notebook', 'computer', 'macbook', 'chromebook', 'pc', 'notebook computer'],
            'person': ['person', 'man', 'woman', 'people', 'human', 'individual', 'someone', 'guy', 'girl'],
            'clothing': ['shirt', 'pants', 'jacket', 'coat', 'dress', 'top', 'sweater', 'hoodie', 'jeans'],
        }
        
        # Action verbs and their variations
        self.action_verbs = {
            'carry': ['carry', 'carrying', 'carries', 'held', 'holding', 'holds', 'bearing'],
            'walk': ['walk', 'walking', 'walks', 'stroll', 'strolling', 'stride', 'striding'],
            'wear': ['wear', 'wearing', 'wears', 'dressed in', 'has on', 'sporting'],
            'hold': ['hold', 'holding', 'holds', 'grip', 'gripping', 'grasp', 'grasping'],
            'place': ['place', 'placing', 'put', 'putting', 'set', 'setting', 'position'],
            'open': ['open', 'opening', 'opens', 'opened', 'unzip', 'unzipping'],
            'close': ['close', 'closing', 'closes', 'closed', 'zip', 'zipping'],
        }
        
        # Color terms
        self.colors = {
            'black', 'white', 'red', 'blue', 'green', 'yellow', 'orange', 'purple', 
            'pink', 'brown', 'gray', 'grey', 'navy', 'beige', 'tan', 'dark', 'light'
        }
        
        # Size/quantity modifiers
        self.size_modifiers = {
            'large', 'big', 'huge', 'small', 'tiny', 'medium', 'several', 'multiple', 'many', 'few'
        }
        
        # Temporal keywords
        self.temporal_keywords = {
            'today', 'yesterday', 'last week', 'last month', 'recent', 'recently',
            'morning', 'afternoon', 'evening', 'night', 'earlier', 'later'
        }
        
        # Negation words
        self.negation_words = {'not', 'no', 'without', 'except', 'excluding'}
        
        # Namespace mapping (aligned with embedding_generator)
        self.namespace_mapping = {
            'bag': ['bagpack', 'backpack', 'bag'],
            'duffel': ['duffel bag', 'duffel_bag'],
            'laptop': ['laptop'],
            'person': ['others'],  # Person-related queries might be in 'others'
        }
    
    def understand_query(self, query: str) -> ExpandedQuery:
        """
        Parse and understand a natural language query
        
        Args:
            query: Natural language search query
            
        Returns:
            ExpandedQuery object with structured understanding
        """
        query_lower = query.lower().strip()
        
        # Extract intent components
        entities = self._extract_entities(query_lower)
        attributes = self._extract_attributes(query_lower)
        actions = self._extract_actions(query_lower)
        temporal = self._extract_temporal_context(query_lower)
        negations = self._extract_negations(query_lower)
        modifiers = self._extract_modifiers(query_lower)
        
        # Determine primary intent
        primary_intent = self._determine_intent(entities, attributes, actions, temporal)
        
        # Create structured intent
        intent = QueryIntent(
            primary_intent=primary_intent,
            entities=entities,
            attributes=attributes,
            actions=actions,
            temporal_context=temporal,
            modifiers=modifiers,
            negations=negations,
            confidence=self._calculate_confidence(entities, attributes, actions)
        )
        
        # Generate query expansions
        expanded_queries = self._expand_query(query, intent)
        
        # Infer target namespaces
        target_namespaces = self._infer_namespaces(intent)
        
        # Build metadata filters
        metadata_filters = self._build_metadata_filters(intent)
        
        # Identify boost terms
        boost_terms = self._identify_boost_terms(intent)
        
        expanded = ExpandedQuery(
            original_query=query,
            expanded_queries=expanded_queries,
            structured_intent=intent,
            target_namespaces=target_namespaces,
            metadata_filters=metadata_filters,
            boost_terms=boost_terms
        )
        
        logger.info(f"Query understanding: '{query}' -> Intent: {primary_intent}, "
                   f"Entities: {entities}, Namespaces: {target_namespaces}")
        
        return expanded
    
    def _extract_entities(self, query: str) -> List[str]:
        """Extract object entities from query"""
        entities = []
        
        # Check against object synonyms
        for category, synonyms in self.object_synonyms.items():
            for synonym in synonyms:
                if re.search(r'\b' + re.escape(synonym) + r'\b', query):
                    entities.append(category)
                    break
        
        return list(set(entities))  # Remove duplicates
    
    def _extract_attributes(self, query: str) -> List[str]:
        """Extract attributes (colors, sizes, materials)"""
        attributes = []
        
        # Extract colors
        for color in self.colors:
            if re.search(r'\b' + color + r'\b', query):
                attributes.append(color)
        
        # Extract size modifiers
        for size in self.size_modifiers:
            if re.search(r'\b' + size + r'\b', query):
                attributes.append(size)
        
        return attributes
    
    def _extract_actions(self, query: str) -> List[str]:
        """Extract action verbs from query"""
        actions = []
        
        for action, variants in self.action_verbs.items():
            for variant in variants:
                if re.search(r'\b' + re.escape(variant) + r'\b', query):
                    actions.append(action)
                    break
        
        return list(set(actions))
    
    def _extract_temporal_context(self, query: str) -> Optional[Dict[str, any]]:
        """Extract temporal context from query"""
        temporal = {}
        
        # Check for temporal keywords
        for keyword in self.temporal_keywords:
            if keyword in query:
                temporal['keyword'] = keyword
                temporal['relative_time'] = self._resolve_relative_time(keyword)
                break
        
        # Check for specific dates (YYYY-MM-DD pattern)
        date_pattern = r'\b(\d{4}[-/]\d{2}[-/]\d{2})\b'
        date_match = re.search(date_pattern, query)
        if date_match:
            temporal['specific_date'] = date_match.group(1).replace('/', '-')
        
        # Check for time ranges
        if 'between' in query and 'and' in query:
            temporal['is_range'] = True
        
        return temporal if temporal else None
    
    def _extract_negations(self, query: str) -> List[str]:
        """Extract negated terms from query"""
        negations = []
        
        for neg_word in self.negation_words:
            pattern = rf'\b{neg_word}\s+(\w+)'
            matches = re.finditer(pattern, query)
            for match in matches:
                negations.append(match.group(1))
        
        return negations
    
    def _extract_modifiers(self, query: str) -> List[str]:
        """Extract descriptive modifiers"""
        # Simple approach: extract adjectives based on common patterns
        modifiers = []
        
        # Pattern: adjective + noun (e.g., "red bag")
        adj_pattern = r'\b(dark|light|bright|dull|new|old|clean|dirty)\b'
        matches = re.findall(adj_pattern, query)
        modifiers.extend(matches)
        
        return modifiers
    
    def _determine_intent(self, entities: List[str], attributes: List[str], 
                         actions: List[str], temporal: Optional[Dict]) -> str:
        """Determine the primary search intent"""
        if temporal:
            return 'temporal_search'
        elif actions:
            return 'action_search'
        elif attributes:
            return 'attribute_search'
        elif entities:
            return 'object_search'
        else:
            return 'general_search'
    
    def _calculate_confidence(self, entities: List[str], attributes: List[str], 
                             actions: List[str]) -> float:
        """Calculate confidence score for intent understanding"""
        score = 0.5  # Base score
        
        if entities:
            score += 0.2
        if attributes:
            score += 0.15
        if actions:
            score += 0.15
        
        return min(score, 1.0)
    
    def _expand_query(self, original_query: str, intent: QueryIntent) -> List[str]:
        """
        Generate expanded query variations with synonyms and paraphrases
        """
        expansions = [original_query]
        query_lower = original_query.lower()
        
        # Expand with entity synonyms
        for entity in intent.entities:
            if entity in self.object_synonyms:
                for synonym in self.object_synonyms[entity][:3]:  # Top 3 synonyms
                    # Simple word replacement
                    for orig_word in self.object_synonyms[entity]:
                        if orig_word in query_lower:
                            expanded = query_lower.replace(orig_word, synonym)
                            if expanded not in expansions:
                                expansions.append(expanded)
        
        # Expand with action variations
        for action in intent.actions:
            if action in self.action_verbs:
                for variant in self.action_verbs[action][:2]:
                    for orig_word in self.action_verbs[action]:
                        if orig_word in query_lower:
                            expanded = query_lower.replace(orig_word, variant)
                            if expanded not in expansions:
                                expansions.append(expanded)
        
        # Generate compositional queries
        if intent.entities and intent.attributes:
            # "black bag" -> ["bag that is black", "black colored bag"]
            for entity in intent.entities:
                for attr in intent.attributes:
                    compositional = f"{attr} {entity}"
                    if compositional not in expansions:
                        expansions.append(compositional)
        
        # Limit expansions to avoid query explosion
        return expansions[:5]
    
    def _infer_namespaces(self, intent: QueryIntent) -> List[str]:
        """
        Infer target namespaces based on query intent
        """
        namespaces = set()
        
        # Map entities to namespaces
        for entity in intent.entities:
            if entity in self.namespace_mapping:
                namespaces.update(self.namespace_mapping[entity])
        
        # If no specific entities found, search all namespaces
        if not namespaces:
            return ['bagpack', 'duffel_bag', 'laptop', 'others']
        
        return list(namespaces)
    
    def _build_metadata_filters(self, intent: QueryIntent) -> Dict[str, any]:
        """
        Build Pinecone metadata filters based on intent
        """
        filters = {}
        
        # Temporal filters
        if intent.temporal_context:
            if 'specific_date' in intent.temporal_context:
                filters['video_date'] = intent.temporal_context['specific_date']
            elif 'relative_time' in intent.temporal_context:
                date_range = intent.temporal_context['relative_time']
                if date_range:
                    filters['video_date'] = {'$gte': date_range['start'], '$lte': date_range['end']}
        
        return filters
    
    def _resolve_relative_time(self, keyword: str) -> Optional[Dict[str, str]]:
        """Resolve relative time keywords to date ranges"""
        today = datetime.now().date()
        
        if keyword == 'today':
            return {'start': today.isoformat(), 'end': today.isoformat()}
        elif keyword == 'yesterday':
            yesterday = today - timedelta(days=1)
            return {'start': yesterday.isoformat(), 'end': yesterday.isoformat()}
        elif keyword == 'last week':
            week_ago = today - timedelta(days=7)
            return {'start': week_ago.isoformat(), 'end': today.isoformat()}
        elif keyword in ['recent', 'recently']:
            three_days_ago = today - timedelta(days=3)
            return {'start': three_days_ago.isoformat(), 'end': today.isoformat()}
        
        return None
    
    def _identify_boost_terms(self, intent: QueryIntent) -> List[str]:
        """
        Identify terms that should be boosted in result ranking
        """
        boost_terms = []
        
        # Boost specific attributes
        boost_terms.extend(intent.attributes)
        
        # Boost specific entities
        boost_terms.extend(intent.entities)
        
        # Boost action verbs
        boost_terms.extend(intent.actions)
        
        return boost_terms
    
    def rerank_results(self, results: List[Dict], intent: QueryIntent) -> List[Dict]:
        """
        Re-rank search results based on query understanding
        
        Args:
            results: List of search results
            intent: Structured query intent
            
        Returns:
            Re-ranked results
        """
        if not results:
            return results
        
        for result in results:
            caption = result.get('caption', '').lower()
            score = result.get('similarity_score', 0.0)
            
            # Apply boosting based on query understanding
            boost_multiplier = 1.0
            
            # Boost if caption contains entities
            for entity in intent.entities:
                if entity in caption:
                    boost_multiplier += 0.1
            
            # Boost if caption contains attributes
            for attr in intent.attributes:
                if attr in caption:
                    boost_multiplier += 0.15
            
            # Boost if caption contains actions
            for action in intent.actions:
                if action in caption:
                    boost_multiplier += 0.1
            
            # Penalize if caption contains negated terms
            for negation in (intent.negations or []):
                if negation in caption:
                    boost_multiplier -= 0.2
            
            # Update score
            result['original_score'] = score
            result['boosted_score'] = min(score * boost_multiplier, 1.0)
            result['similarity_score'] = result['boosted_score']
        
        # Sort by boosted score
        results.sort(key=lambda x: x.get('boosted_score', x.get('similarity_score', 0)), reverse=True)
        
        return results
    
    def explain_query(self, expanded_query: ExpandedQuery) -> str:
        """
        Generate human-readable explanation of query understanding
        """
        intent = expanded_query.structured_intent
        
        explanation_parts = [f"Query: '{expanded_query.original_query}'"]
        explanation_parts.append(f"Intent: {intent.primary_intent}")
        
        if intent.entities:
            explanation_parts.append(f"Looking for: {', '.join(intent.entities)}")
        
        if intent.attributes:
            explanation_parts.append(f"With attributes: {', '.join(intent.attributes)}")
        
        if intent.actions:
            explanation_parts.append(f"Involving actions: {', '.join(intent.actions)}")
        
        if expanded_query.target_namespaces:
            explanation_parts.append(f"Searching in: {', '.join(expanded_query.target_namespaces)}")
        
        if expanded_query.expanded_queries and len(expanded_query.expanded_queries) > 1:
            explanation_parts.append(f"Query variations: {len(expanded_query.expanded_queries)}")
        
        return " | ".join(explanation_parts)


# Example usage and testing
if __name__ == "__main__":
    qu = QueryUnderstanding()
    
    # Test queries
    test_queries = [
        "person carrying a black backpack",
        "red duffel bag on the floor",
        "laptop computer on desk",
        "someone wearing a blue jacket without a bag",
        "large black bag from yesterday",
        "person walking with bagpack today"
    ]
    
    for query in test_queries:
        print(f"\n{'='*80}")
        expanded = qu.understand_query(query)
        print(qu.explain_query(expanded))
        print(f"Expanded queries: {expanded.expanded_queries[:3]}")
        print(f"Target namespaces: {expanded.target_namespaces}")
        if expanded.metadata_filters:
            print(f"Metadata filters: {expanded.metadata_filters}")
