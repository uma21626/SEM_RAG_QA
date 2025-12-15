# ===============================
# Imports & Model Setup
# ===============================
import subprocess
import nltk
import spacy
import networkx as nx
import community as community_louvain
import numpy as np
from collections import defaultdict

from pypdf import PdfReader
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer
import pdfplumber
import warnings
warnings.filterwarnings('ignore')

# nltk.download("punkt")
nlp = spacy.load("en_core_web_sm")
embedder = SentenceTransformer("all-MiniLM-L6-v2")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

TAU_E = 0.40   # entity threshold
TAU_D = 0.35   # chunk threshold

class Chunk:
    def __init__(self, text, page):
        self.text = text
        self.page = page
        self.entities = []
        self.embedding = None
        self.community = None
        self.entity_relations = []

# ===============================
# Load PDF & Sentence Extraction
# ===============================
def load_pdf(pdf_path: str):
    """Load PDF with pdfplumber"""
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text:
                pages.append((i + 1, text))
    return pages

def extract_sentences(pages):
    """Extract sentences with page numbers"""
    sentences = []
    for page_num, text in pages:
        for sent in sent_tokenize(text):
            sentences.append({
                "text": sent,
                "page": page_num
            })
    return sentences

# ===============================
# Semantic Chunking
# ===============================

def semantic_chunking(sentences, sim_threshold=0.75):
    """Semantic chunking via cosine similarity"""
    texts = [s["text"] for s in sentences]
    embeddings = embedder.encode(texts, normalize_embeddings=True)

    chunks = []
    buffer = [sentences[0]]
    buffer_embedding = embeddings[0]

    for i in range(1, len(sentences)):
        sim = cosine_similarity(
            [buffer_embedding],
            [embeddings[i]]
        )[0][0]

        if sim >= sim_threshold:
            buffer.append(sentences[i])
            # Update buffer embedding as average
            buffer_indices = [j for j in range(i - len(buffer) + 1, i + 1)]
            buffer_embedding = np.mean(embeddings[buffer_indices], axis=0)
        else:
            # Create chunk from buffer
            chunks.append(buffer)
            buffer = [sentences[i]]
            buffer_embedding = embeddings[i]

    chunks.append(buffer)
    return chunks

# ===============================
# Buffer Merge & Token Control
# ===============================
def buffer_merge(chunks, max_tokens=1024, subchunk_size=128):
    """Merge chunks respecting token limits"""
    merged = []

    for chunk in chunks:
        text = " ".join(s["text"] for s in chunk)
        page = chunk[0]["page"] if chunk else 1

        tokens = tokenizer.encode(text, add_special_tokens=False)

        if len(tokens) <= max_tokens:
            merged.append(Chunk(text, page))
        else:
            # Create sub-chunks with overlap
            for i in range(0, len(tokens), subchunk_size):
                end_idx = min(i + subchunk_size, len(tokens))
                sub_tokens = tokens[i:end_idx]

                # Add overlap
                if i > 0:
                    overlap_start = max(0, i - 32)
                    overlap_tokens = tokens[overlap_start:i]
                    sub_tokens = overlap_tokens + sub_tokens

                sub_text = tokenizer.decode(sub_tokens)
                merged.append(Chunk(sub_text, page))

    return merged

# ===============================
# Entity & Embeddings
# ===============================
def embed_chunks(chunks):
    """Embed chunks using sentence transformer"""
    for chunk in chunks:
        chunk.embedding = embedder.encode(chunk.text, normalize_embeddings=True)
    return chunks
def enrich_chunks(chunks):
    """Extract entities and relations from chunks"""
    for chunk in chunks:
        doc = nlp(chunk.text)

        # Extract entities
        entities = []
        for ent in doc.ents:
            if ent.label_ in ['PERSON', 'ORG', 'GPE', 'NORP', 'EVENT', 'WORK_OF_ART']:
                entities.append({
                    'text': ent.text,
                    'label': ent.label_,
                    'start': ent.start_char,
                    'end': ent.end_char
                })
        chunk.entities = [e['text'] for e in entities]

        # Extract simple relations (subject-verb-object)
        relations = []
        for sent in doc.sents:
            # Find subject, verb, object patterns
            for token in sent:
                if token.dep_ in ['nsubj', 'nsubjpass']:
                    subj = token.text
                    # Find the root verb
                    root = token.head
                    while root.head != root:
                        root = root.head

                    # Find objects
                    objs = [child.text for child in root.children
                           if child.dep_ in ['dobj', 'pobj', 'attr']]

                    for obj in objs:
                        if subj in chunk.entities and obj in chunk.entities:
                            relations.append({
                                'subject': subj,
                                'relation': root.lemma_,
                                'object': obj
                            })

        chunk.entity_relations = relations

    return chunks

# ===============================
# Knowledge Graph
# ===============================
def build_knowledge_graph(chunks):
    """Build knowledge graph with entities and relationships"""
    G = nx.Graph()

    # Add nodes (entities)
    for idx, chunk in enumerate(chunks):
        for entity in chunk.entities:
            if not G.has_node(entity):
                G.add_node(entity,
                          chunks=[idx],
                          frequency=1,
                          type='entity')
            else:
                G.nodes[entity]['chunks'].append(idx)
                G.nodes[entity]['frequency'] += 1

    # Add edges based on co-occurrence and relations
    for idx, chunk in enumerate(chunks):
        # Co-occurrence edges
        entities_in_chunk = list(set(chunk.entities))
        for i in range(len(entities_in_chunk)):
            for j in range(i + 1, len(entities_in_chunk)):
                e1, e2 = entities_in_chunk[i], entities_in_chunk[j]

                if not G.has_edge(e1, e2):
                    G.add_edge(e1, e2,
                              weight=1,
                              relations=[],
                              chunks=[idx])
                else:
                    G[e1][e2]['weight'] += 1
                    if idx not in G[e1][e2]['chunks']:
                        G[e1][e2]['chunks'].append(idx)

        # Relation-based edges
        for rel in chunk.entity_relations:
            if G.has_node(rel['subject']) and G.has_node(rel['object']):
                if not G.has_edge(rel['subject'], rel['object']):
                    G.add_edge(rel['subject'], rel['object'],
                              weight=2,  # Higher weight for explicit relations
                              relations=[rel['relation']],
                              chunks=[idx])
                else:
                    G[rel['subject']][rel['object']]['weight'] += 2
                    if rel['relation'] not in G[rel['subject']][rel['object']]['relations']:
                        G[rel['subject']][rel['object']]['relations'].append(rel['relation'])

    print(f"Graph built with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    return G

# ===============================
# Community Detection
# ===============================
def detect_communities(G, chunks):
    """Detect communities using Louvain algorithm"""
    if G.number_of_nodes() == 0:
        print("Empty graph, no communities to detect")
        return chunks, {}

    # Create weighted graph for community detection
    weighted_graph = G.copy()
    for u, v, data in weighted_graph.edges(data=True):
        if 'weight' not in data:
            data['weight'] = 1

    try:
        partition = community_louvain.best_partition(weighted_graph, weight='weight')

        # Map communities to chunks
        community_chunks = defaultdict(set)
        for entity, comm in partition.items():
            if entity in G.nodes:
                for chunk_id in G.nodes[entity]['chunks']:
                    community_chunks[chunk_id].add(comm)

        # Assign dominant community to each chunk
        for chunk_id, communities in community_chunks.items():
            if chunk_id < len(chunks):
                # Get the most frequent community
                if communities:
                    from collections import Counter
                    comm_counter = Counter(communities)
                    dominant_comm = comm_counter.most_common(1)[0][0]
                    chunks[chunk_id].community = dominant_comm

        print(f"Detected {len(set(partition.values()))} communities")
        return chunks, partition

    except Exception as e:
        print(f"Community detection failed: {e}")
        return chunks, {}

def local_graph_rag_search(query, chunks, top_k=5):
    """Equation 4: Local Graph RAG Search"""
    q_emb = embedder.encode(query, normalize_embeddings=True)
    results = []

    for chunk in chunks:
        if chunk.embedding is None:
            continue

        # Chunk similarity
        chunk_sim = cosine_similarity([q_emb], [chunk.embedding])[0][0]

        # Entity relevance
        entity_relevance = 0
        if chunk.entities:
            entity_embeddings = embedder.encode(chunk.entities, normalize_embeddings=True)
            entity_sims = cosine_similarity([q_emb], entity_embeddings)[0]
            entity_relevance = np.max(entity_sims) if len(entity_sims) > 0 else 0

        # Combined score: α * chunk_sim + (1-α) * entity_relevance
        alpha = 0.7
        combined_score = alpha * chunk_sim + (1 - alpha) * entity_relevance

        if combined_score > TAU_D:
            results.append((chunk, combined_score, chunk_sim, entity_relevance))

    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_k]

def global_graph_rag_search(query, chunks, top_k=3):
    """Equation 5: Global Graph RAG Search"""
    q_emb = embedder.encode(query, normalize_embeddings=True)

    # Group chunks by community
    comm_chunks = defaultdict(list)
    for chunk in chunks:
        if chunk.community is not None:
            comm_chunks[chunk.community].append(chunk)

    # Calculate community embeddings and scores
    comm_scores = {}
    for comm, comm_chunks_list in comm_chunks.items():
        if not comm_chunks_list:
            continue

        # Community embedding as average of chunk embeddings
        chunk_embeddings = [c.embedding for c in comm_chunks_list if c.embedding is not None]
        if not chunk_embeddings:
            continue

        comm_embedding = np.mean(chunk_embeddings, axis=0)

        # Community similarity
        comm_sim = cosine_similarity([q_emb], [comm_embedding])[0][0]

        # Entity relevance in community
        all_entities = []
        for chunk in comm_chunks_list:
            all_entities.extend(chunk.entities)

        entity_relevance = 0
        if all_entities:
            unique_entities = list(set(all_entities))
            entity_embeddings = embedder.encode(unique_entities[:10], normalize_embeddings=True)
            entity_sims = cosine_similarity([q_emb], entity_embeddings)[0]
            entity_relevance = np.mean(entity_sims) if len(entity_sims) > 0 else 0

        # Global score: β * comm_sim + γ * entity_relevance
        beta, gamma = 0.6, 0.3
        global_score = beta * comm_sim + gamma * entity_relevance
        comm_scores[comm] = (global_score, comm_sim, entity_relevance)

    # Sort communities by score
    ranked_comms = sorted(comm_scores.items(), key=lambda x: x[1][0], reverse=True)[:top_k]

    # Get chunks from top communities
    top_chunks = []
    for comm, (score, comm_sim, entity_rel) in ranked_comms:
        for chunk in comm_chunks[comm]:
            top_chunks.append((chunk, score, comm_sim, entity_rel))

    # Sort chunks by score
    top_chunks.sort(key=lambda x: x[1], reverse=True)
    return top_chunks[:top_k]

def ollama_llm(prompt, model="llama3.2:1b"):
    """Generate response using Ollama"""
    try:
        result = subprocess.run(
            ["ollama", "run", model],
            input=prompt.encode(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30
        )
        return result.stdout.decode().strip()
    except subprocess.TimeoutExpired:
        return "Error: Ollama request timed out."
    except Exception as e:
        return f"Error: {str(e)}"

# ============================================================
# LLM Integration & Answer Generation
# ============================================================
def generate_answer(query, local_docs, global_docs):
    """Generate answer using retrieved context"""

    # Combine local and global contexts
    all_docs = []
    if local_docs:
        all_docs.extend([(c, s) for c, s, _, _ in local_docs])
    if global_docs:
        all_docs.extend([(c, s) for c, s, _, _ in global_docs])

    # Remove duplicates
    unique_docs = {}
    for chunk, score in all_docs:
        if chunk.text not in unique_docs:
            unique_docs[chunk.text] = (chunk, score)

    # Sort by score
    sorted_docs = sorted(unique_docs.values(), key=lambda x: x[1], reverse=True)[:5]

    # Prepare context
    context_parts = []
    for i, (chunk, score) in enumerate(sorted_docs, 1):
        context_parts.append(f"[Context {i}, Score: {score:.3f}, Page: {chunk.page}]\n{chunk.text[:800]}")

    context = "\n\n".join(context_parts)

    prompt = f"""You are an expert assistant answering questions based on Dr. B.R. Ambedkar's writings.

CONTEXT FROM AMBEDKAR'S WRITINGS:
{context}

QUESTION: {query}

INSTRUCTIONS:
1. Answer using ONLY the provided context above
2. If the context doesn't contain relevant information, say so
3. Be accurate and concise
4. Reference specific parts of the context when possible

ANSWER:"""

    return ollama_llm(prompt)

# ============================================================
# System Initialization Pipeline
# ============================================================
def initialize_semrag():
    """Main execution pipeline"""
    print("="*60)
    print("SEMRAG System ")
    print("="*60)

    # 1. Load and process PDF
    print("\n1. Loading PDF...")
    pages = load_pdf("Ambedkar_book.pdf")
    print(f"   Loaded {len(pages)} pages")

    # 2. Extract sentences
    print("\n2. Extracting sentences...")
    sentences = extract_sentences(pages)
    print(f"   Extracted {len(sentences)} sentences")

    # 3. Semantic chunking
    print("\n3. Performing semantic chunking...")
    sem_chunks = semantic_chunking(sentences[:2000])  # Limit for testing
    print(f"   Created {len(sem_chunks)} semantic groups")

    # 4. Buffer merging
    print("\n4. Merging chunks with token limits...")
    chunks = buffer_merge(sem_chunks, max_tokens=1024, subchunk_size=128)
    print(f"   Created {len(chunks)} final chunks")

    # 5. Enrich chunks with entities and embeddings
    print("\n5. Enriching chunks with entities and embeddings...")
    chunks = enrich_chunks(chunks)
    chunks = embed_chunks(chunks)

    # 6. Build knowledge graph
    print("\n6. Building knowledge graph...")
    G = build_knowledge_graph(chunks)

    # 7. Detect communities
    print("\n7. Detecting communities...")
    chunks, partition = detect_communities(G, chunks)

    return chunks, G

chunks, G = initialize_semrag()

# ============================================================
# Question Answering Interface
# ============================================================
def semrag_qa(query, chunks):
        # Local search
        local_docs = local_graph_rag_search(query, chunks)
        print(f"Local chunks retrieved: {len(local_docs)}")
        # Global search
        global_docs = global_graph_rag_search(query, chunks)
        print(f"Global communities retrieved: {len(global_docs)}")
        # Fallback if local search fails
        if not local_docs and global_docs:
            print(" Falling back to global retrieval")
            local_docs = [(c, 0.0) for c in global_docs[:5]]

        if not local_docs:
            return "No relevant information found in the document.", [], []

        # Generate answer
        answer = generate_answer(query, local_docs, global_docs)

        return answer, local_docs, global_docs

# ===============================
# SEMRAG Question Answering - 1
# ===============================
query = "What was Ambedkar's view on caste discrimination?"
print('Question:' , query)

answer, local_docs, global_docs = semrag_qa(query, chunks)

print("Answer:", answer)

# ===============================
# SEMRAG Question Answering - 2
# ===============================
query = "What are Ambedkar's views on quantum mechanics?"
print('Question:' , query)

answer, local_docs, global_docs = semrag_qa(query, chunks)

print("Answer:", answer)

# ===============================
# SEMRAG Question Answering - 3
# ===============================
query = "What were Ambedkar's views on Mahatma Gandhi's assassination?"
print('Question:' , query)

answer, local_docs, global_docs = semrag_qa(query, chunks)

print("Answer:", answer)

# ===============================
# SEMRAG Question Answering - 4
# ===============================
query = "What did Ambedkar say about time travel?"
print('Question:' , query)

answer, local_docs, global_docs = semrag_qa(query, chunks)

print("Answer:", answer)

# ===============================
# SEMRAG Question Answering - 5
# ===============================
query ="What did Ambedkar say about social justice and equality?"
print('Question:' , query)

answer, local_docs, global_docs = semrag_qa(query, chunks)

print("Answer:", answer)

# ===============================
# SEMRAG Question Answering - 6
# ===============================
query ="What did Ambedkar say about labor and workers' rights?"
print('Question:' , query)

answer, local_docs, global_docs = semrag_qa(query, chunks)

print("Answer:", answer)


# ---------------------------------
def main():
    chunks, G = initialize_semrag()
    
    while True:
        query = input("\nEnter your question (or type 'exit'): ")
        if query.lower() == 'exit':
            break
        
        answer, local_docs, global_docs = semrag_qa(query, chunks)
        print("\nAnswer:\n", answer)
        print(f"\nLocal chunks retrieved: {len(local_docs)}")
        print(f"Global communities retrieved: {len(global_docs)}")

if __name__ == "__main__":
    main()


