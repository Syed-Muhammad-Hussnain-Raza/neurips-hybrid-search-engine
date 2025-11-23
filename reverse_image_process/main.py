from image_search import ReverseImageSearch
import os
import sys

if __name__ == "__main__":
    print("="*40)
    print("Reverse Image Search Engine")
    print("="*40)
    
    # Configuration
    images_folder = "sample_images"
    embeddings_file = "image_embeddings.pkl"
    
    # Initialize search engine
    search_engine = ReverseImageSearch(embeddings_file)
    
    while True:
        print("\n" + "-"*40)
        print("\nOptions:")
        print("1. Index images from folder")
        print("2. Search for similar images")
        print("3. View database statistics")
        print("4. Exit")
        
        choice = input("\nSelect option (1-4): ").strip()
        
        if choice == '1':
            # Index images
            folder = input(f"Enter folder path (default: {images_folder}): ").strip()
            if not folder:
                folder = images_folder
            
            if not os.path.exists(folder):
                print(f"Folder not found: {folder}")
                print("Please create the folder and add 20-30 images")
                continue
            
            save_path = input(f"Save embeddings to (default: {embeddings_file}): ").strip()
            if not save_path:
                save_path = embeddings_file
            
            search_engine.index_images(folder, save_path)
        
        elif choice == '2':
            # Search
            if not search_engine.database_embeddings:
                print("\nNo images indexed yet. Please index images first (Option 1)")
                continue
            
            query_path = input("\nEnter path to query image: ").strip()
            
            if not os.path.exists(query_path):
                print(f"Image not found: {query_path}")
                continue
            
            top_k = input("Number of similar images to find (default: 5): ").strip()
            top_k = int(top_k) if top_k.isdigit() else 5
            
            results = search_engine.search(query_path, top_k=top_k)
            
            if results:
                search_engine.print_results(results)
                
                show_visual = input("\nDisplay visual results? (y/n): ").strip().lower()
                if show_visual == 'y':
                    search_engine.display_results(query_path, results)
        
        elif choice == '3':
            # Show stats
            print()
            search_engine.get_database_stats()
        
        elif choice == '4':
            print("\nGoodbye!")
            break
        
        else:
            print("Invalid choice. Please try again.")