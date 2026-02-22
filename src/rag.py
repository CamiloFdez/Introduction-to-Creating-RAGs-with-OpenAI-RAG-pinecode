import os
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFacePipeline
from langchain_pinecone import PineconeVectorStore

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# 1. Embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# 2. Vector store (Pinecone)
vectorstore = PineconeVectorStore(
    index_name=os.getenv("PINECONE_INDEX_NAME"),
    embedding=embeddings
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# 3. LLM (FLAN-T5) — SIN transformers.pipeline()
llm = HuggingFacePipeline.from_model_id(
    model_id="gpt2",
    task="text-generation",
    pipeline_kwargs={
        "max_new_tokens": 200,
        "temperature": 0.1,
        "do_sample": False
    }
)

# 4. Prompt
prompt = ChatPromptTemplate.from_template("""
Context:
{context}

Question:
{input}

Answer:
""")

# 5. RAG chain
doc_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, doc_chain)

# 6. Run
response = rag_chain.invoke({
    "input": "What is this document about?"
})

print("\nAnswer:")
print(response["answer"])