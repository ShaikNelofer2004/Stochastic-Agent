import os
import pymupdf4llm
import google.generativeai as genai
from typing import List
from dotenv import load_dotenv
from .vector_store import SimpleVectorStore

load_dotenv()

# Configure Gemini
api_key = os.getenv("GOOGLE_API_KEY")
if api_key:
    genai.configure(api_key=api_key)

def split_text_recursive(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
    """
    Simple recursive text splitter. 
    Splits by paragraphs, then newlines, then spaces if chunks are too large.
    """
    if len(text) <= chunk_size:
        return [text]
    
    # Try splitting by double newline (paragraph)
    splits = text.split('\n\n')
    chunks = []
    current_chunk = ""
    
    for split in splits:
        if len(current_chunk) + len(split) + 2 <= chunk_size:
            current_chunk += split + "\n\n"
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = split + "\n\n"
            
            # If a single split is too big, split by newline
            if len(current_chunk) > chunk_size:
                sub_splits = current_chunk.split('\n')
                current_chunk = ""
                for sub_split in sub_splits:
                    if len(current_chunk) + len(sub_split) + 1 <= chunk_size:
                        current_chunk += sub_split + "\n"
                    else:
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                        current_chunk = sub_split + "\n"
                        # Handle extremely long lines (force split)
                        while len(current_chunk) > chunk_size:
                            chunks.append(current_chunk[:chunk_size])
                            current_chunk = current_chunk[chunk_size:]
    
    if current_chunk:
        chunks.append(current_chunk.strip())
        
    return chunks

def batch_embed_contents(notebook_contents: List[str], model: str = "models/text-embedding-004"):
    """
    Generates embeddings for a batch of text chunks.
    """
    try:
        results = genai.embed_content(
            model=model,
            content=notebook_contents,
            task_type="retrieval_document"
        )
        return results['embedding']
    except Exception as e:
        print(f"Error generating embeddings: {e}")
        return []

def process_pdfs(pdf_paths: List[str], vector_store_path: str = "vector_store.pkl"):
    """
    Main function to process PDFs, chunk them, embed, and save to vector store.
    """
    store = SimpleVectorStore.load_from_disk(vector_store_path)
    
    for path in pdf_paths:
        print(f"Processing {path}...")
        try:
            # Extract text as Markdown (preserves structure/tables)
            md_text = pymupdf4llm.to_markdown(path)
            
            # Add basic source info to text (optional, but good for context)
            source_header = f"Source Document: {os.path.basename(path)}\n\n"
            md_text = source_header + md_text

            # Chunking
            chunks = split_text_recursive(md_text)
            print(f"  - Split into {len(chunks)} chunks.")
            
            # Embeddings
            # Gemini API has limits, so we might need to batch if too many chunks
            batch_size = 50 # Safe batch size
            all_embeddings = []
            
            for i in range(0, len(chunks), batch_size):
                batch_chunks = chunks[i : i + batch_size]
                embeddings = batch_embed_contents(batch_chunks)
                all_embeddings.extend(embeddings)
                
            # Create Metadata
            metadatas = [{"source": os.path.basename(path)} for _ in chunks]
            
            # Add to store
            if len(all_embeddings) == len(chunks):
                store.add_documents(chunks, all_embeddings, metadatas)
                print(f"  - Added {len(chunks)} chunks to store.")
            else:
                print(f"  - Error: Embedding count mismatch for {path}.")
                
        except Exception as e:
            print(f"Failed to process {path}: {e}")
            
    store.save_to_disk(vector_store_path)
    return store
