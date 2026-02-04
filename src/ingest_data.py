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

def process_pdfs(pdf_paths: List[str], vector_store_path: str = "vector_store.pkl", clear_existing: bool = True):
    """
    Main function to process PDFs, chunk them, embed, and save to vector store.
    """
    if clear_existing and os.path.exists(vector_store_path):
        print(f"Clearing existing vector store at {vector_store_path}...")
        if os.path.exists(vector_store_path):
            os.remove(vector_store_path)
        store = SimpleVectorStore()
    else:
        store = SimpleVectorStore.load_from_disk(vector_store_path)
    
    for path in pdf_paths:
        print(f"Processing {path}...")
        try:
            # Open the PDF file
            import pymupdf
            doc = pymupdf.open(path)
            
            all_chunks = []
            all_metadatas = []
            
            # Iterate through each page
            for page_num, page in enumerate(doc):
                # Extract text as Markdown for this specific page
                # pymupdf4llm can convert specific pages
                md_text = pymupdf4llm.to_markdown(path, pages=[page_num])
                
                # Add basic source info to text
                source_header = f"Source Document: {os.path.basename(path)} | Page: {page_num + 1}\n\n"
                md_text = source_header + md_text

                # Chunking
                page_chunks = split_text_recursive(md_text)
                
                # Extend lists
                all_chunks.extend(page_chunks)
                # Add metadata for each chunk
                for _ in page_chunks:
                    all_metadatas.append({
                        "source": os.path.basename(path),
                        "page": page_num + 1
                    })

            print(f"  - Split {os.path.basename(path)} into {len(all_chunks)} chunks.")
            
            # Embeddings
            batch_size = 50
            all_embeddings = []
            
            for i in range(0, len(all_chunks), batch_size):
                batch_chunks = all_chunks[i : i + batch_size]
                embeddings = batch_embed_contents(batch_chunks)
                all_embeddings.extend(embeddings)
            
            # Add to store
            if len(all_embeddings) == len(all_chunks):
                store.add_documents(all_chunks, all_embeddings, all_metadatas)
                print(f"  - Added {len(all_chunks)} chunks to store.")
            else:
                print(f"  - Error: Embedding count mismatch for {path}.")
                
        except Exception as e:
            print(f"Failed to process {path}: {e}")
            
    store.save_to_disk(vector_store_path)
    return store
