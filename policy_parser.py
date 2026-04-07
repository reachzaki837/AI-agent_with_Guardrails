import os
from typing import Iterable, Union
import requests
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

try:
    from langchain_ollama import OllamaEmbeddings
except ImportError:
    from langchain_community.embeddings import OllamaEmbeddings

# Global vector store instance
_vector_store = None


def _ollama_is_available(base_url: str) -> bool:
    """Checks whether Ollama is reachable before trying embeddings."""
    try:
        response = requests.get(f"{base_url}/api/tags", timeout=3)
        return response.status_code == 200
    except requests.RequestException:
        return False

def initialize_rag(file_paths: Union[str, Iterable[str]]):
    """Chunks one or more PDFs and loads them into a local vector database."""
    global _vector_store

    if isinstance(file_paths, str):
        normalized_paths = [file_paths]
    else:
        normalized_paths = [path for path in file_paths if path]

    if not normalized_paths:
        print("[red]No valid PDF paths were provided.[/red]")
        return

    existing_paths = [path for path in normalized_paths if os.path.exists(path)]
    missing_paths = [path for path in normalized_paths if not os.path.exists(path)]

    for missing_path in missing_paths:
        print(f"[red]File not found: {missing_path}[/red]")

    if not existing_paths:
        return

    print(f"\n[dim]1/3 Extracting text from {len(existing_paths)} PDF file(s)...[/dim]")
    text = ""
    for path in existing_paths:
        print(f"[dim]- {os.path.basename(path)}[/dim]")
        reader = PdfReader(path)
        for page in reader.pages:
            text += (page.extract_text() or "") + "\n"

    if not text.strip():
        print("[red]No readable text was extracted from the provided PDF files.[/red]")
        return

    print("[dim]2/3 Chunking text into readable segments...[/dim]")
    # Split the text into 1000-character chunks with a 200-character overlap so sentences aren't cut in half
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_text(text)

    if not chunks:
        print("[red]No text chunks were created from the provided PDF files.[/red]")
        return

    print("[dim]3/3 Creating local vector database (this takes a few seconds)...[/dim]")
    ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    if not _ollama_is_available(ollama_base_url):
        print(
            "[red]Ollama is not reachable at "
            f"{ollama_base_url}[/red]\n"
            "[yellow]Start Ollama first (for example: 'ollama serve') and retry.[/yellow]"
        )
        return

    # Use Ollama to generate embeddings locally
    embeddings = OllamaEmbeddings(model="nomic-embed-text", base_url=ollama_base_url)
    
    try:
        # Load the chunks into Chroma
        _vector_store = Chroma.from_texts(texts=chunks, embedding=embeddings)
        print("[bold green]Vector database ready![/bold green]\n")
    except Exception as exc:
        print(f"[red]Failed to build vector database: {exc}[/red]")
        _vector_store = None

def get_relevant_policy(query: str, k: int = 2) -> str:
    """Searches the vector DB and returns the top 'k' most relevant paragraphs."""
    if not _vector_store:
        return ""
        
    # Search the database for the chunks that best match the user's query
    docs = _vector_store.similarity_search(query, k=k)
    
    # Combine the top results into a single string
    return "\n\n...\n\n".join([doc.page_content for doc in docs])