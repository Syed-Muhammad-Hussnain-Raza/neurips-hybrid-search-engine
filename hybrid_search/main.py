from scraper import NIPSScraper
from database import PaperDatabase
from search_engine import HybridSearchEngine
import sys

def scrape_and_store():
    print("Scraping NIPS papers")
    scraper = NIPSScraper(2024)
    papers = scraper.scrape_with_retry()
    
    if not papers:
        print("Failed to scrape papers. Exiting.")
        return False
    
    print(f"\nStoring {len(papers)} papers in MongoDB...")
    db = PaperDatabase()
    
    # Clear existing data
    response = input("Clear existing data? (y/n): ")
    if response.lower() == 'y':
        db.clear_collection()
    
    db.insert_papers(papers)
    print(f"Total papers in database: {db.get_paper_count()}")
    db.close()
    
    return True

def search_interface():
    db = PaperDatabase()
    
    if db.get_paper_count() == 0:
        print("No papers in database. Please scrape first.")
        db.close()
        return
    
    engine = HybridSearchEngine(db)
    print("\nBuilding search index (this may take a minute)...")
    engine.build_index()
    
    print("\n" + "="*40)
    print("NIPS Paper Search Engine")
    print("="*40)
    print("\nSearch Types:")
    print("1. Hybrid Search")
    print("2. Semantic Search")
    print("3. Keyword Search")
    print("4. Author Search")
    print("5. Exit")
    
    while True:
        print("\n" + "-"*40)
        choice = input("\nSelect search type (1-5): ").strip()
        
        if choice == '5':
            print("Goodbye!")
            break
        
        if choice not in ['1', '2', '3', '4']:
            print("Invalid choice. Please try again.")
            continue
        
        query = input("Enter your search query: ").strip()
        if not query:
            print("Empty query. Please try again.")
            continue
        
        top_k = input("Number of results (default 10): ").strip()
        top_k = int(top_k) if top_k.isdigit() else 10
        
        print(f"\nSearching for: '{query}'...")
        
        if choice == '1':
            results = engine.hybrid_search(query, top_k=top_k)
            engine.display_results(results, show_scores=True)
        elif choice == '2':
            results = engine.semantic_search(query, top_k=top_k)
            engine.display_results(results, show_scores=True)
        elif choice == '3':
            results = [(paper, 0.0) for paper in engine.keyword_search(query, top_k=top_k)]
            engine.display_results(results, show_scores=False)
        elif choice == '4':
            results = [(paper, 0.0) for paper in engine.search_by_author(query, top_k=top_k)]
            engine.display_results(results, show_scores=False)
    
    db.close()

def main():
    print("="*40)
    print("NIPS Papers Hybrid Search System")
    print("="*40)
    print("\nOptions:")
    print("1. Scrape and store papers")
    print("2. Search papers")
    print("3. Both (scrape then search)")
    print("4. Exit")
    
    choice = input("\nSelect option (1-4): ").strip()
    
    if choice == '1':
        scrape_and_store()
    elif choice == '2':
        search_interface()
    elif choice == '3':
        if scrape_and_store():
            search_interface()
    elif choice == '4':
        print("Goodbye!")
    else:
        print("Invalid choice.")

if __name__ == "__main__":
    main()