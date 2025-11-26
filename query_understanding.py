import re
from typing import List, Dict, Any
from collections import defaultdict

# Simple synonym/action/attribute dictionaries (expand as needed)
SYNONYMS = {
    "backpack": ["bag", "rucksack", "bookbag"],
    "laptop": ["notebook", "computer"],
    "person": ["man", "woman", "individual"],
    "coat": ["jacket", "outerwear"],
    # ...
}
ACTIONS = {
    "carrying": ["holding", "bearing", "toting"],
    "walking": ["strolling", "moving", "striding"],
    "wearing": ["dressed in", "has on"],
    # ...
}
ATTRIBUTES = {
    "black": ["dark", "ebony"],
    "red": ["crimson", "scarlet"],
    # ...
}

def extract_entities(query: str) -> List[str]:
    # Naive: match known entities
    found = []
    for ent in SYNONYMS:
        if ent in query.lower():
            found.append(ent)
    return found

def extract_attributes(query: str) -> List[str]:
    found = []
    for attr in ATTRIBUTES:
        if attr in query.lower():
            found.append(attr)
    return found

def extract_actions(query: str) -> List[str]:
    found = []
    for act in ACTIONS:
        if act in query.lower():
            found.append(act)
    return found

def extract_temporal(query: str) -> List[str]:
    temporal_keywords = ["today", "yesterday", "last week", "this morning"]
    return [t for t in temporal_keywords if t in query.lower()]

def extract_negations(query: str) -> List[str]:
    negation_keywords = ["without", "not", "no", "except"]
    return [n for n in negation_keywords if n in query.lower()]

def classify_intent(query: str) -> str:
    if extract_entities(query):
        if extract_actions(query):
            return "action_search"
        if extract_attributes(query):
            return "attribute_search"
        return "object_search"
    if extract_temporal(query):
        return "temporal_search"
    return "general_search"

def expand_query(query: str) -> List[str]:
    # Generate variations: synonyms, actions, compositional
    entities = extract_entities(query)
    attributes = extract_attributes(query)
    actions = extract_actions(query)
    variations = set([query])

    # Synonym expansion
    for ent in entities:
        for syn in SYNONYMS.get(ent, []):
            variations.add(query.replace(ent, syn))

    # Action expansion
    for act in actions:
        for var in ACTIONS.get(act, []):
            variations.add(query.replace(act, var))

    # Attribute compositional
    for ent in entities:
        for attr in attributes:
            variations.add(f"{ent} that is {attr}")
            variations.add(f"{attr} colored {ent}")

    return list(variations)

def parse_query(query: str) -> Dict[str, Any]:
    return {
        "entities": extract_entities(query),
        "attributes": extract_attributes(query),
        "actions": extract_actions(query),
        "temporal": extract_temporal(query),
        "negations": extract_negations(query),
        "intent": classify_intent(query),
        "expanded_queries": expand_query(query)
    }