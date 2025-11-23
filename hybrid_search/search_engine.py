from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Dict, Tuple
from database import PaperDatabase

class HybridSearchEngine:
    def __init__(self, db: PaperDatabase, model_name: str = 'all-MiniLM-L6-v2'):        
        # Initialize hybrid search engine with semantic and keyword search
        self.db = db
        self.model = SentenceTransformer(model_name)
        self.paper_embeddings = None
        self.papers = None
        
    def build_index(self):
        print("Building semantic search index...")
        self.papers = self.db.get_all_papers()
        
        if not self.papers:
            print("No papers found in database")
            return
        
        # Create text representation for each paper
        paper_texts = []
        for paper in self.papers:
            text = f"{paper['title']} {' '.join(paper.get('authors', []))}"
            paper_texts.append(text)
        
        # Compute embeddings
        self.paper_embeddings = self.model.encode(paper_texts, 
                                                   convert_to_numpy=True,
                                                   show_progress_bar=True)
        
        print(f"Index built with {len(self.papers)} papers")
    
    def semantic_search(self, query: str, top_k: int = 10) -> List[Tuple[Dict, float]]:
        """
        Perform semantic search using embeddings
        Returns list of (paper, similarity_score) tuples
        """
        if self.paper_embeddings is None or self.papers is None:
            print("Index not built. Call build_index() first.")
            return []
        
        # Encode query
        query_embedding = self.model.encode(query, convert_to_numpy=True)
        
        # Calculate cosine similarity
        similarities = np.dot(self.paper_embeddings, query_embedding) / (
            np.linalg.norm(self.paper_embeddings, axis=1) * np.linalg.norm(query_embedding)
        )
        
        # Get top-k results
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            results.append((self.papers[idx], float(similarities[idx])))
        
        return results
    
    def keyword_search(self, query: str, top_k: int = 10) -> List[Dict]:
        """
        Perform keyword-based search using MongoDB
        """
        return self.db.keyword_search(query, limit=top_k)
    
    def hybrid_search(self, query: str, top_k: int = 10, 
                     semantic_weight: float = 0.7) -> List[Tuple[Dict, float]]:
        """
        Combine semantic and keyword search
        semantic_weight: weight for semantic results (0-1)
        """
        # Get semantic results
        semantic_results = self.semantic_search(query, top_k * 2)
        
        # Get keyword results
        keyword_results = self.keyword_search(query, top_k * 2)
        
        # Combine and re-rank
        paper_scores = {}
        
        # Add semantic scores
        for paper, score in semantic_results:
            paper_id = str(paper.get('_id', ''))
            paper_scores[paper_id] = {
                'paper': paper,
                'score': score * semantic_weight
            }
        
        # Add keyword scores (normalized)
        keyword_weight = 1 - semantic_weight
        for i, paper in enumerate(keyword_results):
            paper_id = str(paper.get('_id', ''))
            keyword_score = (len(keyword_results) - i) / len(keyword_results)
            
            if paper_id in paper_scores:
                paper_scores[paper_id]['score'] += keyword_score * keyword_weight
            else:
                paper_scores[paper_id] = {
                    'paper': paper,
                    'score': keyword_score * keyword_weight
                }
        
        # Sort by combined score
        sorted_results = sorted(paper_scores.values(), 
                               key=lambda x: x['score'], 
                               reverse=True)
        
        return [(item['paper'], item['score']) for item in sorted_results[:top_k]]
    
    def search_by_author(self, author_name: str, top_k: int = 10) -> List[Dict]:
        """
        Search papers by author name
        """
        return self.db.author_search(author_name, limit=top_k)
    
    def display_results(self, results: List[Tuple[Dict, float]], 
                       show_scores: bool = True):
        """
        Display search results in a readable format
        """
        if not results:
            print("No results found.")
            return
        
        print(f"\n{'='*80}")
        print(f"Found {len(results)} results:")
        print(f"{'='*80}\n")
        
        for i, (paper, score) in enumerate(results, 1):
            print(f"{i}. {paper['title']}")
            print(f"   Authors: {', '.join(paper.get('authors', ['N/A']))}")
            print(f"   Link: {paper.get('link', 'N/A')}")
            if show_scores:
                print(f"   Score: {score:.4f}")
            print()


if __name__ == "__main__":
    # Test search engine
    db = PaperDatabase()
    engine = HybridSearchEngine(db)
    
    if db.get_paper_count() > 0:
        engine.build_index()
        
        # Test search
        results = engine.hybrid_search("transformer attention mechanism", top_k=5)
        engine.display_results(results)
    else:
        print("No papers in database. Run main.py first.")
    
    db.close()