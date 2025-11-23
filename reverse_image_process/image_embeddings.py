import torch
from transformers import ViTImageProcessor, ViTModel
from PIL import Image
import numpy as np
import os
from typing import List, Dict, Tuple
import pickle

class ImageEmbeddingGenerator:
    def __init__(self, model_name: str = 'google/vit-base-patch32-224-in21k'):
        """
        Initialize Vision Transformer model for image embeddings
        ViT-B/32 is a good balance of speed and accuracy
        """
        print(f"Loading model: {model_name}")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        self.processor = ViTImageProcessor.from_pretrained(model_name)
        self.model = ViTModel.from_pretrained(model_name).to(self.device)
        self.model.eval()  # Set to evaluation mode
        
        print("Model loaded successfully")
    
    def load_image(self, image_path: str) -> Image.Image:
        """Load and convert image to RGB"""
        try:
            img = Image.open(image_path).convert('RGB')
            return img
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return None
    
    def get_embedding(self, image_path: str) -> np.ndarray:
        """
        Generate embedding for a single image
        Returns normalized embedding vector
        """
        img = self.load_image(image_path)
        if img is None:
            return None
        
        # Process image
        inputs = self.processor(images=img, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate embedding
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Use [CLS] token embedding (first token)
            embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        
        # Normalize embedding
        embedding = embedding / np.linalg.norm(embedding)
        
        return embedding.flatten()
    
    def generate_embeddings_for_folder(self, folder_path: str) -> Dict[str, np.ndarray]:
        """
        Generate embeddings for all images in a folder
        Returns dictionary mapping image paths to embeddings
        """
        if not os.path.exists(folder_path):
            print(f"Folder not found: {folder_path}")
            return {}
        
        embeddings = {}
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff'}
        
        image_files = [f for f in os.listdir(folder_path) 
                      if os.path.splitext(f.lower())[1] in image_extensions]
        
        if not image_files:
            print(f"No images found in {folder_path}")
            return {}
        
        print(f"Processing {len(image_files)} images...")
        
        for i, filename in enumerate(image_files, 1):
            image_path = os.path.join(folder_path, filename)
            print(f"[{i}/{len(image_files)}] Processing: {filename}")
            
            embedding = self.get_embedding(image_path)
            if embedding is not None:
                embeddings[image_path] = embedding
        
        print(f"\nSuccessfully generated {len(embeddings)} embeddings")
        return embeddings
    
    def save_embeddings(self, embeddings: Dict[str, np.ndarray], save_path: str):
        """Save embeddings to disk"""
        with open(save_path, 'wb') as f:
            pickle.dump(embeddings, f)
        print(f"Embeddings saved to {save_path}")
    
    def load_embeddings(self, load_path: str) -> Dict[str, np.ndarray]:
        """Load embeddings from disk"""
        if not os.path.exists(load_path):
            print(f"File not found: {load_path}")
            return {}
        
        with open(load_path, 'rb') as f:
            embeddings = pickle.load(f)
        print(f"Loaded {len(embeddings)} embeddings from {load_path}")
        return embeddings
    
    def calculate_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two embeddings
        Returns similarity score (0-1, higher is more similar)
        """
        # Embeddings are already normalized, so dot product = cosine similarity
        similarity = np.dot(emb1, emb2)
        return float(similarity)
    
    def find_similar_images(self, 
                           query_embedding: np.ndarray, 
                           database_embeddings: Dict[str, np.ndarray],
                           top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Find most similar images to query
        Returns list of (image_path, similarity_score) tuples
        """
        similarities = []
        
        for image_path, embedding in database_embeddings.items():
            similarity = self.calculate_similarity(query_embedding, embedding)
            similarities.append((image_path, similarity))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]


if __name__ == "__main__":
    # Test embedding generation
    generator = ImageEmbeddingGenerator()
    
    # Test with sample images folder
    folder_path = "sample_images"
    if os.path.exists(folder_path):
        embeddings = generator.generate_embeddings_for_folder(folder_path)
        
        if embeddings:
            # Save embeddings
            generator.save_embeddings(embeddings, "image_embeddings.pkl")
            
            print(f"\nEmbedding shape: {list(embeddings.values())[0].shape}")
    else:
        print(f"Please create '{folder_path}' folder and add images")