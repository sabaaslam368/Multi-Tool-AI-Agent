
import ast
import operator
from pathlib import Path

from langchain.tools import tool


# ==========================================
# TOOL 1: SAFE CALCULATOR
# ==========================================

ALLOWED_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

MAX_POWER = 100
MAX_RESULT = 10**50


def evaluate_expression(node):
    """Sirf approved arithmetic expressions evaluate karein."""

    if isinstance(node, ast.Constant):
        if (
            isinstance(node.value, (int, float))
            and not isinstance(node.value, bool)
        ):
            return node.value

        raise ValueError("Only numeric values are allowed.")

    if isinstance(node, ast.BinOp):
        left = evaluate_expression(node.left)
        right = evaluate_expression(node.right)

        operator_type = type(node.op)

        if operator_type not in ALLOWED_OPERATORS:
            raise ValueError("This operator is not allowed.")

        if isinstance(node.op, (ast.Div, ast.FloorDiv, ast.Mod)):
            if right == 0:
                raise ValueError("Cannot divide by zero.")

        if isinstance(node.op, ast.Pow):
            if abs(right) > MAX_POWER:
                raise ValueError("Exponent is too large.")

        result = ALLOWED_OPERATORS[operator_type](left, right)

        if isinstance(result, (int, float)):
            if abs(result) > MAX_RESULT:
                raise ValueError("Result exceeds the allowed limit.")

        return result

    if isinstance(node, ast.UnaryOp):
        operator_type = type(node.op)

        if operator_type not in ALLOWED_OPERATORS:
            raise ValueError("This unary operator is not allowed.")

        return ALLOWED_OPERATORS[operator_type](
            evaluate_expression(node.operand)
        )

    raise ValueError("Invalid mathematical expression.")


@tool
def calculator(expression: str) -> str:
    """Perform arithmetic calculations such as 125 * 24 + 50.
    Use this tool when the user asks for an exact calculation.
    """

    try:
        if not expression or len(expression) > 200:
            return "Calculator error: Invalid expression length."

        tree = ast.parse(expression, mode="eval")
        result = evaluate_expression(tree.body)

        return f"Calculation result: {result}"

    except Exception as error:
        return f"Calculator error: {error}"


# ==========================================
# RESEARCH DATA DIRECTORY
# ==========================================

DATA_DIR = (Path(__file__).resolve().parent / "data").resolve()


# ==========================================
# TOOL 2: READ RESEARCH FILE
# ==========================================

@tool
def read_research_file(filename: str = "research_notes.txt") -> str:
    """Read the local research TXT file.
    Use this tool when the user asks to read or summarize a file.
    Only TXT files inside the data folder are allowed.
    """

    try:
        if not filename or Path(filename).name != filename:
            return "File error: Please provide only a filename."

        if Path(filename).suffix.lower() != ".txt":
            return "File error: Only TXT files are allowed."

        file_path = (DATA_DIR / filename).resolve()

        if file_path.parent != DATA_DIR:
            return "File error: Access outside the data folder is denied."

        if not file_path.is_file():
            return f"File error: {filename} was not found."

        if file_path.stat().st_size > 100_000:
            return "File error: File is too large to read."

        content = file_path.read_text(encoding="utf-8")

        if not content.strip():
            return "File error: The file is empty."

        return content[:15000]

    except Exception as error:
        return f"File reading error: {error}"


# ==========================================
# TOOL 3: SEARCH RESEARCH NOTES
# ==========================================

@tool
def search_research_notes(keyword: str) -> str:
    """Search local research notes for a keyword.
    Use this tool to find information about a specific topic
    in research_notes.txt.
    """

    try:
        if not keyword or not keyword.strip():
            return "Search error: Please provide a keyword."

        file_path = (DATA_DIR / "research_notes.txt").resolve()

        if not file_path.is_file():
            return "Search error: research_notes.txt was not found."

        content = file_path.read_text(encoding="utf-8")

        matching_lines = [
            line.strip()
            for line in content.splitlines()
            if keyword.strip().lower() in line.lower()
        ]

        if not matching_lines:
            return f"No matching lines found for: {keyword}"

        return (
            f"Search results for '{keyword}':\n"
            + "\n".join(matching_lines[:30])
        )

    except Exception as error:
        return f"Search error: {error}"


# ==========================================
# TOOL 4: STUDY HELPER
# ==========================================

STUDY_TOPICS = {
    "python": (
        "Python is a general-purpose programming language. "
        "It is used in automation, web development, data analysis, "
        "and artificial intelligence."
    ),
    "machine learning": (
        "Machine Learning is a branch of AI in which systems learn "
        "patterns from data and use them to make predictions."
    ),
    "rag": (
        "RAG stands for Retrieval-Augmented Generation. "
        "It retrieves relevant information from a knowledge source "
        "and supplies it to an LLM to support grounded answers."
    ),
    "langgraph": (
        "LangGraph helps developers build stateful workflows using "
        "nodes, edges, shared state, and conditional routing."
    ),
    "langchain": (
        "LangChain provides components for building applications "
        "with language models, prompts, tools, and retrieval."
    ),
    "embeddings": (
        "Embeddings represent text as numerical vectors. "
        "They can be used for semantic similarity and retrieval."
    ),
    "generative ai": (
        "Generative AI creates new content such as text, images, "
        "audio, video, and code."
    ),
}


@tool
def study_helper(topic: str) -> str:
    """Explain a basic AI or Computer Science concept.
    Use this tool when the user requests a simple definition
    or beginner-friendly explanation of a supported topic.
    """

    if not topic or not topic.strip():
        return "Study helper error: Please provide a topic."

    normalized_topic = topic.strip().lower()

    for known_topic, explanation in STUDY_TOPICS.items():
        if (
            normalized_topic == known_topic
            or known_topic in normalized_topic
        ):
            return f"{known_topic.title()}:\n{explanation}"

    return (
        f"No predefined explanation is available for '{topic}'. "
        "Try Python, RAG, LangGraph, LangChain, embeddings, "
        "Machine Learning, or Generative AI."
    )


# ==========================================
# REGISTER ALL TOOLS
# ==========================================

tools = [
    calculator,
    read_research_file,
    search_research_notes,
    study_helper,
]
