# Search Engine Project

This project implements two types of search engines:
1. **Hybrid Search** - For searching NIPS conference papers
2. **Reverse Image Search** - For finding similar images using Vision Transformers

## Project Structure

```
search-engine-project/
├── hybrid_search/           # NIPS papers search engine
│   ├── scraper.py          # Web scraper for NIPS papers
│   ├── database.py         # MongoDB database handler
│   ├── search_engine.py    # Hybrid search implementation
│   ├── main.py             # Main application
│   └── requirements.txt    # Python dependencies
├── reverse_image_search/    # Image similarity search
│   ├── image_embeddings.py # ViT embedding generator
│   ├── image_search.py     # Search engine implementation
│   ├── main.py             # Main application
│   ├── requirements.txt    # Python dependencies
│   └── sample_images/      # Folder for image dataset
└── README.md               # This file
```

## Part 1: Hybrid Search on NIPS Papers

### Features
- Scrapes papers from NIPS 2024 conference website
- Stores papers in MongoDB with hierarchical structure
- Implements three search methods:
  - **Semantic Search**: Using sentence embeddings
  - **Keyword Search**: Traditional text matching
  - **Hybrid Search**: Combines both approaches
- Search by title, authors, or keywords

### Setup

1. **Install MongoDB** (if not already installed)
   ```bash
   # macOS
   brew install mongodb-community
   brew services start mongodb-community
   
   # Ubuntu
   sudo apt install mongodb
   sudo systemctl start mongodb
   
   # Windows: Download from mongodb.com
   ```

2. **Install Python dependencies**
   ```bash
   cd hybrid_search
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   python main.py
   ```

### Usage

The main application offers three options:
1. **Scrape and store papers** - Fetches papers from NIPS website
2. **Search papers** - Interactive search interface
3. **Both** - Scrape then search

Search types available:
- Hybrid Search (combines semantic + keyword)
- Semantic Search (using embeddings)
- Keyword Search (traditional)
- Author Search (by author name)

### Example Searches
```
Query: "transformer attention mechanism"
Query: "reinforcement learning"
Query: "Geoffrey Hinton" (author search)
```

## Part 2: Reverse Image Search

### Features
- Uses Vision Transformer (ViT-B/32) for image embeddings
- Generates embeddings for image dataset
- Finds similar images using cosine similarity
- Visual display of search results

### Setup

1. **Install Python dependencies**
   ```bash
   cd reverse_image_search
   pip install -r requirements.txt
   ```

2. **Prepare image dataset**
   ```bash
   mkdir sample_images
   # Add 20-30 images to this folder
   ```

3. **Run the application**
   ```bash
   python main.py
   ```

### Usage

The main application offers:
1. **Index images** - Generate embeddings for all images
2. **Search** - Find similar images to a query
3. **View statistics** - Database information

### Example Workflow
1. Add 20-30 images to `sample_images/` folder
2. Run `python main.py`
3. Choose option 1 to index images
4. Choose option 2 to search with a query image
5. View results visually with matplotlib

## Technical Details

### Hybrid Search
- **Web Scraping**: BeautifulSoup for parsing HTML
- **Database**: MongoDB for hierarchical data storage
- **Embeddings**: sentence-transformers (all-MiniLM-L6-v2)
- **Search**: Combines semantic and keyword matching

### Reverse Image Search
- **Model**: Vision Transformer (ViT-B/32)
- **Embeddings**: 768-dimensional vectors
- **Similarity**: Cosine similarity on normalized embeddings
- **Visualization**: matplotlib for displaying results

## Requirements

### System Requirements
- Python 3.8+
- MongoDB 4.0+
- 4GB RAM minimum
- GPU recommended for faster image processing (optional)

### Python Packages
See individual `requirements.txt` files in each subdirectory.

## Notes

### Hybrid Search
- First run will take time to scrape papers
- Embeddings are computed when building search index
- MongoDB must be running before starting

### Reverse Image Search
- First run downloads ViT model (~300MB)
- GPU significantly speeds up embedding generation
- Supported image formats: JPG, PNG, BMP, GIF, TIFF

## Troubleshooting

**MongoDB Connection Error**
```bash
# Check if MongoDB is running
# macOS/Linux
brew services list  # or: systemctl status mongodb

# Start MongoDB if needed
brew services start mongodb-community
```

**Model Download Issues**
```bash
# Set HuggingFace cache directory
export TRANSFORMERS_CACHE=/path/to/cache
```

**Out of Memory**
```bash
# Reduce batch size or use CPU instead of GPU
# Edit image_embeddings.py: device = 'cpu'
```

## Future Enhancements

- Add vector database (Pinecone, Milvus) for scaling
- Implement advanced filtering options
- Add web interface using Flask/FastAPI
- Support for more image formats and preprocessing
- Multi-modal search (text + image)