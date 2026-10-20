"""
RAG System for TechNova Solutions Policy Q&A
Built using LangChain + Groq (LLM) + HuggingFace Embeddings (local)

Assignment: Hands-on Assignment: Build a RAG System Using LangChain
"""

import os
import sys
from dotenv import load_dotenv

# Ensure UTF-8 output on Windows (handles special Unicode characters)
sys.stdout.reconfigure(encoding="utf-8")

# Load GROQ_API_KEY from .env file (never hardcode secrets in source code)
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError(
        "GROQ_API_KEY not found. "
        "Please create a .env file with: GROQ_API_KEY=your_key_here"
    )

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


# -----------------------------------------------------------------------------
# Task 1. Load the company policy document
# -----------------------------------------------------------------------------
POLICY_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "technova_policy.txt.md"
)

loader = TextLoader(POLICY_FILE, encoding="utf-8")
documents = loader.load()
print(f"[1] Loaded document: {POLICY_FILE}")
print(f"    Characters loaded: {len(documents[0].page_content)}\n")


# -----------------------------------------------------------------------------
# Task 2. Split the document into smaller chunks
# -----------------------------------------------------------------------------
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50,
    separators=["\n\n", "\n", ". ", " ", ""],
)
chunks = text_splitter.split_documents(documents)
print(f"[2] Document split into {len(chunks)} chunk(s):")
for i, chunk in enumerate(chunks):
    print(f"    Chunk {i + 1}: {repr(chunk.page_content[:80])}...")
print()


# -----------------------------------------------------------------------------
# Tasks 3 & 4. Generate embeddings and store in a vector store
# -----------------------------------------------------------------------------
# HuggingFace local embeddings are completely FREE - no API key required.
# The model (~90 MB) is downloaded automatically on first run.
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(chunks, embeddings)
print("[3&4] Embeddings generated and stored in FAISS vector store.\n")


# -----------------------------------------------------------------------------
# Task 5. Create a retriever (top-3 most relevant chunks)
# -----------------------------------------------------------------------------
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
print("[5] Retriever created (top-3 chunks).\n")


# -----------------------------------------------------------------------------
# Task 6. Build a LangChain prompt template and LLM chain
# -----------------------------------------------------------------------------
prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are a helpful HR assistant for TechNova Solutions.
Use ONLY the context provided below to answer the employee question.
If the answer is not present in the context, clearly state:
"The provided policy document does not contain information about this topic."

Context:
{context}

Question: {question}

Answer:""",
)

llm = ChatGroq(model="qwen/qwen3.8-27b", temperature=0, groq_api_key=GROQ_API_KEY)


def format_docs(docs):
    """Concatenate retrieved chunks into a single context string."""
    return "\n\n".join(doc.page_content for doc in docs)


# LCEL pipeline: retriever -> prompt -> LLM -> output parser
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt_template
    | llm
    | StrOutputParser()
)
print("[6] LangChain RAG pipeline initialised.\n")


# -----------------------------------------------------------------------------
# Task 7. Test with the three assignment questions
# -----------------------------------------------------------------------------
test_questions = [
    "How many paid annual leave days does a full-time employee receive per year?",
    "An employee wants to work remotely on Wednesday and take leave on Friday. "
    "What do the company policies say about these two requests?",
    "What is the company parental leave policy?",
]

print("=" * 70)
print("TEST RESULTS")
print("=" * 70)

for i, question in enumerate(test_questions, start=1):
    print(f"\nQuestion {i}: {question}")
    print("-" * 70)
    answer = rag_chain.invoke(question)
    print(f"Answer:\n{answer}")
    retrieved_docs = retriever.invoke(question)
    print("\nSource chunks used:")
    for j, doc in enumerate(retrieved_docs, start=1):
        print(f"  [{j}] {repr(doc.page_content[:120])}...")

print("\n" + "=" * 70)
print("Done.")
print("=" * 70)


# =============================================================================
# ACTUAL OUTPUTS FOR ALL THREE TEST QUESTIONS
# Model       : qwen/qwen3.8-27b via Groq
# Embeddings  : sentence-transformers/all-MiniLM-L6-v2 (local, HuggingFace)
# Vector Store: FAISS (in-memory)
# =============================================================================

# -----------------------------------------------------------------------------
# Question 1 - Direct retrieval
# Question: How many paid annual leave days does a full-time employee receive per year?
#
# Answer:
# Every full-time employee receives 18 paid annual leave days per calendar year.
#
# Source Chunks:
# [1] Annual Leave: 18 paid annual leave days per calendar year...
# [2] Sick Leave: 10 paid sick leave days per calendar year...
# [3] Working Hours: 9:30 AM to 5:30 PM, Monday through Friday...
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Question 2 - Combining information
# Question: An employee wants to work remotely on Wednesday and take leave on Friday.
#           What do the company policies say about these two requests?
#
# Answer:
# Based on the provided policy document, here is the information regarding the employee's requests:
#
# **1. Working Remotely on Wednesday:**
# *   **Eligibility:** Employees are permitted to work remotely on Wednesdays.
# *   **Requirements:** The employee must inform their manager by email at least one working day in advance.
# *   **Approval:** Remote work is subject to manager approval.
#
# **2. Taking Leave on Friday:**
# *   **Eligibility:** The policy states that the standard working day is Monday through Friday, and employees receive 18 paid annual leave days per calendar year. It does not explicitly prohibit taking leave on Fridays.
# *   **Requirements:** The employee must submit an annual leave request to their manager at least 3 calendar days before the requested leave date.
# *   **Approval:** Leave approval depends on the manager's confirmation.
#
# *Note: The policy does not specify any conflict between working remotely on one day and taking leave on another, provided both requests meet their respective advance notice and approval requirements.*
#
# Source Chunks:
# [1] Work From Home: Employees may work remotely on Wednesdays and Fridays...
# [2] Working Hours: The standard working day is from 9:30 AM to 5:30 PM...
# [3] Annual Leave: Every full-time employee receives 18 paid annual leave days...
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Question 3 - Information not present in the document
# Question: What is the company parental leave policy?
#
# Answer:
# The provided policy document does not contain information about this topic.
#
# Source Chunks:
# [1] Annual Leave: Every full-time employee receives 18 paid annual leave days...
# [2] TechNova Solutions - Employee Policy Document header...
# [3] Sick Leave: Employees receive 10 paid sick leave days...
# -----------------------------------------------------------------------------

