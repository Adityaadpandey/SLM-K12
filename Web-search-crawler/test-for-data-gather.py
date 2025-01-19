from __future__ import annotations
from typing import List, Optional
import asyncio
import logging
from bs4 import BeautifulSoup
import httpx
from crawler import WebCrawler, WebContent
from urllib.parse import urlparse
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FastWebSearchCrawler(WebCrawler):
    """An optimized web crawler for quick search and content extraction."""

    def __init__(self, cache_dir: str = ".cache"):
        super().__init__(cache_dir=cache_dir)
        # Initialize httpx client with optimized settings
        self.http_client = httpx.AsyncClient(
            timeout=30.0,
            follow_redirects=True,
            verify=False,
            limits=httpx.Limits(max_keepalive_connections=10, max_connections=20),
            headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
        )

    async def _extract_markdown(self, html: str, url: str) -> str:
        """Extract and format content as markdown with improved parsing."""
        try:
            soup = BeautifulSoup(html, 'html.parser')

            # Remove unwanted elements
            for elem in soup.find_all(['script', 'style', 'nav', 'footer', 'header', 'aside', 'iframe']):
                elem.decompose()

            content_parts = []
            title = soup.title.string if soup.title else ''
            content_parts.append(f"# {title.strip()}\n")

            # Process main content elements
            main_tags = ['article', 'main', 'div[role="main"]', '.main-content', '#content']
            main_content = None
            for selector in main_tags:
                main_content = soup.select_one(selector)
                if main_content:
                    break

            if not main_content:
                main_content = soup

            # Extract structured content
            for tag in main_content.find_all(['h1', 'h2', 'h3', 'h4', 'p', 'ul', 'ol']):
                text = tag.get_text(strip=True)
                if not text:
                    continue

                if tag.name.startswith('h'):
                    level = int(tag.name[1])
                    content_parts.append(f"\n{'#' * level} {text}\n")
                elif tag.name == 'p' and len(text) > 30:
                    content_parts.append(f"\n{text}\n")
                elif tag.name in ['ul', 'ol']:
                    items = []
                    for li in tag.find_all('li', recursive=False):
                        li_text = li.get_text(strip=True)
                        if li_text:
                            items.append(f"- {li_text}")
                    if items:
                        content_parts.append("\n" + "\n".join(items) + "\n")

            formatted_content = "\n".join(content_parts)
            return formatted_content.strip()

        except Exception as e:
            logger.error(f"Markdown extraction error for {url}: {e}")
            return ""

    async def fetch_url(self, url: str, retries: int = 2) -> Optional[WebContent]:
        """Enhanced fetch_url with better error handling and retry logic."""
        try:
            # Check cache first
            if cached := await self._get_from_cache(url):
                logger.info(f"Cache hit for {url}")
                return cached

            for attempt in range(retries):
                try:
                    response = await self.http_client.get(url)
                    response.raise_for_status()

                    soup = BeautifulSoup(response.text, 'html.parser')
                    markdown_content = await self._extract_markdown(response.text, url)

                    if not markdown_content:
                        continue

                    content = WebContent(
                        title=soup.title.string if soup.title else None,
                        content=markdown_content,
                        url=url,
                        source_quality=self.get_domain_score(url),
                        timestamp=datetime.now().isoformat()
                    )

                    await self._save_to_cache(content)
                    return content

                except httpx.TimeoutException:
                    if attempt == retries - 1:
                        logger.error(f"Timeout fetching {url} after {retries} retries")
                        break
                    await asyncio.sleep(1)
                except Exception as e:
                    if attempt == retries - 1:
                        logger.error(f"Error fetching {url}: {e}")
                        break
                    await asyncio.sleep(1)

        except Exception as e:
            logger.error(f"Error processing {url}: {e}")

        return None

    async def search_and_extract(self, query: str, num_results: int = 4) -> str:
        """Search and extract content with parallel processing."""
        results = await self.crawl(query, num_results)
        if not results:
            return ""

        context_parts = []
        for result in results:
            if result and result.content:
                context_parts.append(
                    f"Source: {result.url}\n"
                    f"Quality Score: {result.source_quality}\n"
                    f"---\n"
                    f"{result.content}\n"
                    f"---\n"
                )

        return "\n\n".join(context_parts)

    async def close(self):
        """Clean up resources."""
        await self.http_client.aclose()

async def main():
    """Example usage of FastWebSearchCrawler."""
    crawler = FastWebSearchCrawler()
    try:
        query = "What is photosynthesis?"
        print(f"Searching for: {query}")

        context = await crawler.search_and_extract(query)
        if context:
            print("\nExtracted content:")
            print(context)
        else:
            print("No results found")

    except Exception as e:
        print(f"Error: {e}")
    finally:
        await crawler.close()

if __name__ == "__main__":
    asyncio.run(main())
