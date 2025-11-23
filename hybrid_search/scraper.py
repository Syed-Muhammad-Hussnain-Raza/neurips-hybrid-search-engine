import requests
from bs4 import BeautifulSoup
import time
from typing import List, Dict

class NIPSScraper:
    def __init__(self, year: int = 2024):
        self.year = year
        self.base_url = f"https://papers.nips.cc/paper_files/paper/{year}"
        
    def scrape_papers(self) -> List[Dict]:
        """
        Scrape all papers from NIPS conference page
        Returns list of dictionaries with title, authors, and link
        """
        print(f"Scraping NIPS {self.year} papers...")
        
        try:
            response = requests.get(self.base_url, timeout=30)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            
            papers = []
            
            # Find all paper entries
            paper_items = soup.find_all('li')
            
            for item in paper_items:
                try:
                    # Find title and link
                    title_link = item.find('a')
                    if not title_link:
                        continue
                    
                    title = title_link.text.strip()
                    link = title_link.get('href', '')
                    
                    # Make absolute URL if relative
                    if link and not link.startswith('http'):
                        link = f"https://papers.nips.cc{link}"
                    
                    # Find authors
                    authors = []
                    author_tag = item.find('i')
                    if author_tag:
                        author_text = author_tag.text.strip()
                        # Split by comma and clean
                        authors = [a.strip() for a in author_text.split(',')]
                    
                    if title and link:
                        paper = {
                            'title': title,
                            'authors': authors,
                            'link': link,
                            'year': self.year
                        }
                        papers.append(paper)
                        print(f"Scraped: {title[:50]}...")
                        
                except Exception as e:
                    print(f"Error parsing paper item: {e}")
                    continue
            
            print(f"\nTotal papers scraped: {len(papers)}")
            return papers
            
        except requests.RequestException as e:
            print(f"Error fetching page: {e}")
            return []
    
    def scrape_with_retry(self, max_retries: int = 3) -> List[Dict]:
        """Scrape with retry logic"""
        for attempt in range(max_retries):
            papers = self.scrape_papers()
            if papers:
                return papers
            print(f"Retry {attempt + 1}/{max_retries}...")
            time.sleep(2)
        return []


if __name__ == "__main__":
    scraper = NIPSScraper(2024)
    papers = scraper.scrape_with_retry()
    print(f"\nSuccessfully scraped {len(papers)} papers")