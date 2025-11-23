from image_embeddings import ImageEmbeddingGenerator
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
from typing import List, Tuple, Dict
import numpy as np

class ReverseImageSearch:
    def __init__(self, embeddings_path: str = None):
        """
        Initialize reverse image search engine
        """
        self.generator = ImageEmbeddingGenerator()
        self.database_embeddings = {}
        
        if embeddings_path and os.path.exists(embeddings_path):
            self.database_embeddings = self.generator.load_embeddings(embeddings_path)
    
    def index_images(self, folder_path: str, save_path: str = "image_embeddings.pkl"):
        """
        Index all images in a folder
        """
        print(f"Indexing images from: {folder_path}")
        self.database_embeddings = self.generator.generate_embeddings_for_folder(folder_path)
        
        if self.database_embeddings:
            self.generator.save_embeddings(self.database_embeddings, save_path)
            print(f"Indexed {len(self.database_embeddings)} images")
        else:
            print("No images indexed")
    
    def search(self, query_image_path: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Search for similar images given a query image
        Returns list of (image_path, similarity_score) tuples
        """
        if not self.database_embeddings:
            print("No images in database. Please index images first.")
            return []
        
        print(f"Searching for similar images to: {query_image_path}")
        
        # Generate embedding for query image
        query_embedding = self.generator.get_embedding(query_image_path)
        
        if query_embedding is None:
            print("Failed to process query image")
            return []
        
        # Find similar images
        results = self.generator.find_similar_images(
            query_embedding, 
            self.database_embeddings, 
            top_k=top_k
        )
        
        return results
    
    def display_results(self, query_image_path: str, results: List[Tuple[str, float]]):
        """
        Display query image and search results using matplotlib
        """
        if not results:
            print("No results to display")
            return
        
        # Calculate grid layout
        n_results = len(results)
        n_cols = min(3, n_results + 1)  # Max 3 columns
        n_rows = (n_results + 1 + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows))
        
        # Flatten axes for easier indexing
        if n_rows == 1:
            axes = [axes] if n_cols == 1 else axes
        else:
            axes = axes.flatten()
        
        # Display query image
        try:
            query_img = mpimg.imread(query_image_path)
            axes[0].imshow(query_img)
            axes[0].set_title("Query Image", fontsize=12, fontweight='bold')
            axes[0].axis('off')
        except Exception as e:
            print(f"Error displaying query image: {e}")
        
        # Display results
        for i, (image_path, similarity) in enumerate(results, 1):
            if i >= len(axes):
                break
            
            try:
                img = mpimg.imread(image_path)
                axes[i].imshow(img)
                filename = os.path.basename(image_path)
                axes[i].set_title(f"{filename}\nSimilarity: {similarity:.4f}", 
                                fontsize=10)
                axes[i].axis('off')
            except Exception as e:
                print(f"Error displaying image {image_path}: {e}")
        
        # Hide empty subplots
        for i in range(n_results + 1, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def print_results(self, results: List[Tuple[str, float]]):
        """
        Print search results in text format
        """
        print("\n" + "="*80)
        print(f"Top {len(results)} Similar Images:")
        print("="*80 + "\n")
        
        for i, (image_path, similarity) in enumerate(results, 1):
            filename = os.path.basename(image_path)
            print(f"{i}. {filename}")
            print(f"   Path: {image_path}")
            print(f"   Similarity: {similarity:.4f}")
            print()
    
    def get_database_stats(self):
        """Get statistics about the indexed database"""
        if not self.database_embeddings:
            print("Database is empty")
            return
        
        print(f"Total indexed images: {len(self.database_embeddings)}")
        
        if self.database_embeddings:
            sample_embedding = list(self.database_embeddings.values())[0]
            print(f"Embedding dimension: {sample_embedding.shape[0]}")
        
        print("\nIndexed images:")
        for i, path in enumerate(self.database_embeddings.keys(), 1):
            print(f"  {i}. {os.path.basename(path)}")


if __name__ == "__main__":
    # Test reverse image search
    search_engine = ReverseImageSearch()
    
    # Index images
    folder_path = "sample_images"
    if os.path.exists(folder_path):
        search_engine.index_images(folder_path)
        
        # Get stats
        search_engine.get_database_stats()
        
        # Test search with first image as query
        images = list(search_engine.database_embeddings.keys())
        if images:
            query_image = images[0]
            print(f"\nUsing {os.path.basename(query_image)} as query")
            results = search_engine.search(query_image, top_k=5)
            search_engine.print_results(results)
            search_engine.display_results(query_image, results)
    else:
        print(f"Please create '{folder_path}' folder and add 20-30 images")