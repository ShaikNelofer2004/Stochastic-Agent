import os
import google.generativeai as genai
import arxiv
from dotenv import load_dotenv
from .vector_store import SimpleVectorStore

load_dotenv()

# Configure Gemini
api_key = os.getenv("GOOGLE_API_KEY")
if api_key:
    genai.configure(api_key=api_key)

class DocumentAgent:
    def __init__(self, vector_store_path: str = "vector_store.pkl"):
        self.vector_store_path = vector_store_path
        self.store = SimpleVectorStore.load_from_disk(vector_store_path)
        self.model = genai.GenerativeModel('gemini-1.5-flash')
        self.chat_history = []

    def _get_embedding(self, text: str):
        try:
            result = genai.embed_content(
                model="models/text-embedding-004",
                content=text,
                task_type="retrieval_query"
            )
            return result['embedding']
        except Exception as e:
            print(f"Embedding error: {e}")
            return []

    def retrieve_context(self, query: str, k: int = 5) -> str:
        """Retrieves relevant context from local documents."""
        query_embedding = self._get_embedding(query)
        if not query_embedding:
            return ""
            
        results = self.store.search(query_embedding, k=k)
        context_parts = []
        for res in results:
             # Add citation info to the context
            source = res['metadata'].get('source', 'Unknown')
            context_parts.append(f"--- SOURCE: {source} ---\n{res['text']}\n")
            
        return "\n".join(context_parts)

    def search_arxiv(self, query: str, max_results: int = 3) -> str:
        """Searches Arxiv for papers."""
        try:
            search = arxiv.Search(
                query = query,
                max_results = max_results,
                sort_by = arxiv.SortCriterion.Relevance
            )
            
            results = []
            for result in search.results():
                results.append(
                    f"Title: {result.title}\n"
                    f"Authors: {', '.join(a.name for a in result.authors)}\n"
                    f"Summary: {result.summary}\n"
                    f"URL: {result.pdf_url}\n"
                )
            return "\n---\n".join(results)
        except Exception as e:
            return f"Error searching Arxiv: {e}"

    def ask(self, user_query: str) -> str:
        """
        Main entry point. Decides whether to use local documents, Arxiv, or general knowledge.
        """
        # 1. Routing Logic (Simple Keyword/LLM based)
        # We'll use a lightweight LLM call or heuristics to decide the tool.
        # For efficiency, let's try a heuristic first: 
        # If "find papers" or "search arxiv" in query -> Arxiv
        # Else -> RAG
        
        tool_use = "RAG"
        if "find papers" in user_query.lower() or "search arxiv" in user_query.lower() or "look up" in user_query.lower():
            tool_use = "ARXIV"
            
        context = ""
        system_instruction = ""
        
        if tool_use == "ARXIV":
            print("Creating Arxiv context...")
            arxiv_results = self.search_arxiv(user_query)
            context = f"Arxiv Search Results:\n{arxiv_results}"
            system_instruction = (
                "You are a helpful research assistant. "
                "The user is asking to find research papers. "
                "Use the provided Arxiv search results to answer the user's request. "
                "Cite the papers with their Titles and URLs."
            )
        else:
            # RAG Mode
            print("Retrieving context...")
            context = self.retrieve_context(user_query)
            if not context:
                # Fallback if no docs
                system_instruction = (
                    "You are a helpful AI assistant. "
                    "The user is asking a question, but no relevant local documents were found. "
                    "Answer to the best of your ability using your general knowledge, "
                    "but mention that you don't have access to specific documents about this."
                )
            else:
                system_instruction = (
                    "You are an expert research assistant. "
                    "Use the provided Context from uploaded PDF documents to answer the user's question. "
                    "If the answer is found in the context, cite the source document name. "
                    "If the answer is NOT in the context, say so.\n\n"
                    "FORMATTING:\n"
                    "- Use Markdown.\n"
                    "- If listing facts, use bullet points.\n"
                    "- When summarizing, keep it structured."
                )

        prompt = f"{system_instruction}\n\nUSER QUESTION: {user_query}\n\nCONTEXT:\n{context}"
        
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error generating answer: {e}"
