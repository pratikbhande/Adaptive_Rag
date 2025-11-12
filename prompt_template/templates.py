STRATEGY_PROMPTS = {
    "concise": """You are a helpful assistant focused on brevity and clarity. Answer the question concisely based on the context provided.

Context:
{context}

Question: {question}

Provide a brief, direct answer in 2-3 sentences. Focus only on the essential information.""",
    
    "detailed": """You are a knowledgeable assistant specializing in comprehensive explanations. Provide a thorough and detailed answer based on the context.

Context:
{context}

Question: {question}

Give a detailed, well-explained answer that covers:
- Main concepts and definitions
- Supporting details and examples
- Relevant background information
- Any important nuances or caveats

Use multiple paragraphs if needed to fully address the question.""",
    
    "structured": """You are an analytical assistant who organizes information systematically. Provide a well-structured answer based on the context.

Context:
{context}

Question: {question}

Structure your answer with clear organization:
1. Start with a brief overview
2. Break down the information into logical sections or points
3. Use clear transitions between ideas
4. Conclude with a summary if appropriate

Make the structure easy to follow and scan.""",

    "example_driven": """You are a practical assistant who explains through concrete examples. Answer the question by providing real-world examples and analogies based on the context.

Context:
{context}

Question: {question}

Provide an answer that:
- Uses specific examples to illustrate key points
- Includes analogies or comparisons to familiar concepts
- Shows practical applications or use cases
- Makes abstract concepts concrete and relatable

Focus on making the answer tangible and easy to understand through examples.""",

    "analytical": """You are a critical thinking assistant who provides in-depth analysis. Answer the question with analytical depth based on the context.

Context:
{context}

Question: {question}

Provide an analytical answer that:
- Examines the question from multiple angles
- Discusses implications and relationships
- Compares and contrasts different aspects
- Identifies patterns, causes, or effects
- Considers limitations or alternative perspectives

Prioritize depth of understanding and critical analysis."""
}

QUERY_ANALYSIS_PROMPT = """Analyze this query and determine its complexity level:

Query: {query}

Consider:
- Length and specificity of the query
- Technical depth required
- Whether it asks for definitions, explanations, analysis, or examples
- Scope (narrow vs broad topic)

Respond with ONLY ONE WORD: simple, moderate, or complex"""

QUERY_CLUSTERING_PROMPT = """Analyze and categorize this query into a semantic group.

Query: {query}

Previous query groups and examples:
{existing_groups}

Task:
1. Determine if this query belongs to an existing group based on semantic similarity (not just word matching)
2. If it's similar to an existing group, return that group name
3. If it's a new type of query, create a descriptive group name (2-4 words)

Consider:
- Intent of the query (what, why, how, when, etc.)
- Topic domain (technical, general knowledge, procedural, etc.)
- Level of detail requested (overview, specific detail, comparison, etc.)

Respond in this exact format:
GROUP: [group_name]
REASON: [one sentence explaining why]

Examples:
- "What is machine learning?" and "Explain ML" → GROUP: ml_definition
- "How does X work?" and "Explain the process of X" → GROUP: process_explanation
- "Compare A and B" and "What's the difference between A and B" → GROUP: comparison_query"""

SIMILARITY_EVALUATION_PROMPT = """Evaluate if these two queries are semantically similar (asking for the same type of information).

Query 1: {query1}
Query 2: {query2}

Consider:
- Do they ask about the same topic or concept?
- Do they have the same intent (definition, explanation, comparison, etc.)?
- Would the same answer strategy work well for both?

Ignore:
- Exact word matching
- Minor phrasing differences
- Word order

Respond with ONLY: SIMILAR or DIFFERENT"""