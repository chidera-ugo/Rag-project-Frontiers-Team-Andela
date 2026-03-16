import os
import time
from pathlib import Path
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.messages import SystemMessage, HumanMessage, convert_to_messages
from langchain_core.documents import Document
from openai import RateLimitError, APIStatusError

from dotenv import load_dotenv

load_dotenv(override=True)

MODEL       = "gpt-4.1"
HELPER_MODEL = "gpt-4.1"  # use gpt-4.1-nano for quicker iteration runs
DB_NAME = str(Path(__file__).parent.parent / "vector_db")

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
RETRIEVAL_K = 20   # candidates retrieved per query
FINAL_K = 10       # kept after reranking

SYSTEM_PROMPT = """You are an expert assistant representing Insurellm, an insurance technology company.
Your answer will be evaluated for accuracy, relevance, and completeness — make sure it fully answers the question.
If the answer is directly stated in the context, report it exactly — do not paraphrase numbers, names, or dates.
If you cannot find the answer in the context, say so clearly.

Context:
{context}
"""

vectorstore  = Chroma(persist_directory=DB_NAME, embedding_function=embeddings)
llm          = ChatOpenAI(temperature=0, model_name=MODEL)
helper_llm   = ChatOpenAI(temperature=0, model_name=HELPER_MODEL)


def _is_rate_limit(exc: Exception) -> bool:
    if isinstance(exc, RateLimitError):
        return True
    if isinstance(exc, APIStatusError) and exc.status_code == 429:
        return True
    return False


def _invoke_with_retry(llm_instance, messages, **kwargs):
    """Invoke an LLM with exponential backoff on rate limit errors."""
    delay = 5
    for attempt in range(6):
        try:
            return llm_instance.invoke(messages, **kwargs)
        except Exception as e:
            if _is_rate_limit(e) and attempt < 5:
                time.sleep(delay)
                delay *= 2
            else:
                raise


def _search_with_retry(question: str, k: int) -> list[Document]:
    """similarity_search with backoff — embedding calls can also be rate limited."""
    delay = 5
    for attempt in range(6):
        try:
            return vectorstore.similarity_search(question, k=k)
        except Exception as e:
            if _is_rate_limit(e) and attempt < 5:
                time.sleep(delay)
                delay *= 2
            else:
                raise


# ── Pydantic schema for structured rerank output ───────────────────────────

class RankOrder(BaseModel):
    order: list[int] = Field(
        description="Chunk IDs ordered from most relevant to least relevant"
    )


# ── Query rewriting ────────────────────────────────────────────────────────

def rewrite_query(question: str, history: list[dict]) -> str:
    """Rewrite the user question into a focused KB search query."""
    prompt = f"""You are searching a Knowledge Base about Insurellm, an insurance technology company.
Rewrite the following question as a short, precise search query that will surface the most relevant content.
Focus on specific entities, names, values, or product names mentioned.
Respond ONLY with the rewritten query — nothing else.

Conversation history: {history}
Question: {question}
Rewritten query:"""
    response = _invoke_with_retry(helper_llm, [HumanMessage(content=prompt)])
    return response.content.strip()


# ── Reranking ──────────────────────────────────────────────────────────────

def rerank(question: str, docs: list[Document]) -> list[Document]:
    """Ask the LLM to reorder docs by relevance to the question."""
    if len(docs) <= 1:
        return docs

    chunks_text = ""
    for i, doc in enumerate(docs):
        chunks_text += f"# CHUNK ID: {i + 1}\n{doc.page_content}\n\n"

    prompt = f"""You are a document re-ranker.
Rank the following chunks by relevance to the question, most relevant first.
Reply ONLY with a JSON object: {{"order": [<chunk ids>]}}. Include every chunk ID.

Question: {question}

{chunks_text}"""

    response = _invoke_with_retry(
        helper_llm,
        [HumanMessage(content=prompt)],
        response_format={"type": "json_object"},
    )
    order = RankOrder.model_validate_json(response.content).order
    # guard against out-of-range ids
    reranked = [docs[i - 1] for i in order if 1 <= i <= len(docs)]
    # append any docs that were omitted
    seen = {id(d) for d in reranked}
    reranked += [d for d in docs if id(d) not in seen]
    return reranked


# ── Context retrieval ──────────────────────────────────────────────────────

def fetch_context(question: str, history: list[dict] = []) -> list[Document]:
    """
    Dual search (original + rewritten query), merge, rerank, return top FINAL_K.
    """
    rewritten = rewrite_query(question, history)

    docs1 = _search_with_retry(question, RETRIEVAL_K)
    docs2 = _search_with_retry(rewritten, RETRIEVAL_K)

    # merge, deduplicate by content
    seen_content = set()
    merged = []
    for doc in docs1 + docs2:
        if doc.page_content not in seen_content:
            seen_content.add(doc.page_content)
            merged.append(doc)

    reranked = rerank(question, merged)
    return reranked[:FINAL_K]


# ── Answer generation ──────────────────────────────────────────────────────

def answer_question(question: str, history: list[dict] = []) -> tuple[str, list[Document]]:
    """
    Answer the given question with RAG; return the answer and the context documents.
    """
    CONTEXT_CHAR_LIMIT = 5000
    docs = fetch_context(question, history)
    context_parts = []
    chars = 0
    for doc in docs:
        part = f"Extract from {doc.metadata.get('source', 'unknown')}:\n{doc.page_content}"
        if chars + len(part) > CONTEXT_CHAR_LIMIT:
            break
        context_parts.append(part)
        chars += len(part)
    context = "\n\n".join(context_parts)
    system_prompt = SYSTEM_PROMPT.format(context=context)
    messages = [SystemMessage(content=system_prompt)]
    messages.extend(convert_to_messages(history))
    messages.append(HumanMessage(content=question))
    response = _invoke_with_retry(llm, messages)
    return response.content, docs
