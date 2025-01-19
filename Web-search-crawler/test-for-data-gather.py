import asyncio
import aiohttp
from bs4 import BeautifulSoup
from typing import List, Dict, Any
import requests

class FastCrawler:
    def __init__(self, api_key: str, cse_id: str):
        self.api_key = api_key
        self.cse_id = cse_id
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }

    def get_search_links(self, query: str, num_results: int = 4) -> List[str]:
        """Get links from Google CSE."""
        search_url = "https://www.googleapis.com/customsearch/v1"
        params = {
            'q': query,
            'cx': self.cse_id,
            'key': self.api_key,
            'num': num_results
        }

        try:
            response = requests.get(search_url, params=params)
            results = response.json()
            return [item['link'] for item in results.get('items', [])]
        except Exception as e:
            print(f"Search error: {e}")
            return []

    async def fetch_content(self, session: aiohttp.ClientSession, url: str) -> Dict[str, Any]:
        """Quickly fetch content from URL with 5s timeout."""
        try:
            async with session.get(url, timeout=5) as response:
                if response.status == 200:
                    html = await response.text()
                    soup = BeautifulSoup(html, 'html.parser')

                    # Remove non-content elements
                    for tag in soup(['script', 'style', 'nav', 'header', 'footer']):
                        tag.decompose()

                    text = soup.get_text(separator='\n', strip=True)
                    return {'url': url, 'content': text, 'status': 'success'}

                return {'url': url, 'content': None, 'status': f'error: {response.status}'}

        except asyncio.TimeoutError:
            return {'url': url, 'content': None, 'status': 'timeout'}
        except Exception as e:
            return {'url': url, 'content': None, 'status': f'error: {str(e)}'}

    async def process_query(self, query: str) -> List[Dict[str, Any]]:
        """Process search query and fetch results."""
        urls = self.get_search_links(query)
        if not urls:
            return []

        async with aiohttp.ClientSession(headers=self.headers) as session:
            tasks = [self.fetch_content(session, url) for url in urls]
            return await asyncio.gather(*tasks)

# Example usage
async def main():
    API_KEY = "AIzaSyBxUlm_z4rOjdvJTa_JVOrdMym97eo4oQk"
    CSE_ID = "01eb2bd62cc744962"

    crawler = FastCrawler(API_KEY, CSE_ID)
    results = await crawler.process_query("Why did Mrs. Packletide want to kill a tiger?")

    for result in results:
        if result['status'] == 'success':
            print(f"\nURL: {result['url']}")
            print(f"Content preview: {result['content']}...")
        else:
            print(f"\nFailed: {result['url']} - {result['status']}")

if __name__ == "__main__":
    asyncio.run(main())
