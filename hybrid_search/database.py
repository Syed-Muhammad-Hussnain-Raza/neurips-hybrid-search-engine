from pymongo import MongoClient, TEXT
from typing import List, Dict, Optional
import os

class PaperDatabase:
    def __init__(self, 
                 connection_string: str = "mongodb://localhost:27017/",
                 db_name: str = "nips_papers",
                 collection_name: str = "papers"):
        """
        Initialize MongoDB connection
        """
        self.client = MongoClient(connection_string)
        self.db = self.client[db_name]
        self.collection = self.db[collection_name]
        
        # Create text index for full-text search
        self._create_indexes()
    
    def _create_indexes(self):
        """Create necessary indexes for efficient searching"""
        try:
            # Text index on title and authors for full-text search
            self.collection.create_index([
                ("title", TEXT),
                ("authors", TEXT)
            ])
            print("Indexes created successfully")
        except Exception as e:
            print(f"Index creation note: {e}")
    
    def insert_papers(self, papers: List[Dict]) -> int:
        """
        Insert multiple papers into database
        Returns count of inserted documents
        """
        if not papers:
            return 0
        
        try:
            # Clear existing data (optional)
            # self.collection.delete_many({})
            
            result = self.collection.insert_many(papers)
            count = len(result.inserted_ids)
            print(f"Inserted {count} papers into database")
            return count
        except Exception as e:
            print(f"Error inserting papers: {e}")
            return 0
    
    def clear_collection(self):
        """Clear all documents from collection"""
        result = self.collection.delete_many({})
        print(f"Deleted {result.deleted_count} documents")
    
    def get_paper_count(self) -> int:
        """Get total count of papers in database"""
        return self.collection.count_documents({})
    
    def get_all_papers(self, limit: int = 0) -> List[Dict]:
        """Get all papers from database"""
        cursor = self.collection.find({})
        if limit > 0:
            cursor = cursor.limit(limit)
        return list(cursor)
    
    def text_search(self, query: str, limit: int = 10) -> List[Dict]:
        """
        Perform full-text search on title and authors
        """
        try:
            results = self.collection.find(
                {"$text": {"$search": query}},
                {"score": {"$meta": "textScore"}}
            ).sort([("score", {"$meta": "textScore"})]).limit(limit)
            
            return list(results)
        except Exception as e:
            print(f"Text search error: {e}")
            return []
    
    def keyword_search(self, keyword: str, limit: int = 10) -> List[Dict]:
        """
        Search using regex pattern matching (case-insensitive)
        Searches in title and authors
        """
        try:
            regex_pattern = {"$regex": keyword, "$options": "i"}
            
            results = self.collection.find({
                "$or": [
                    {"title": regex_pattern},
                    {"authors": regex_pattern}
                ]
            }).limit(limit)
            
            return list(results)
        except Exception as e:
            print(f"Keyword search error: {e}")
            return []
    
    def author_search(self, author_name: str, limit: int = 10) -> List[Dict]:
        """
        Search papers by author name
        """
        try:
            results = self.collection.find({
                "authors": {"$regex": author_name, "$options": "i"}
            }).limit(limit)
            
            return list(results)
        except Exception as e:
            print(f"Author search error: {e}")
            return []
    
    def close(self):
        """Close database connection"""
        self.client.close()


if __name__ == "__main__":
    # Test database connection
    db = PaperDatabase()
    count = db.get_paper_count()
    print(f"Database contains {count} papers")
    db.close()