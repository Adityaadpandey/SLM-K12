from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, AsyncIterator, Literal
import asyncio
import httpx
import cohere
import google.generativeai as genai
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from urllib.parse import urlparse
import streamlit as st
from streamlit_chat import message
import time
import logging
from functools import lru_cache
import os
from datetime import datetime
from googlesearch import search
from pathlib import Path
import json
import aiofiles
import hashlib
import urllib3

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

@dataclass(frozen=True)
class WebContent:
    """Represents crawled web content with immutable fields."""
    title: str | None
    content: str | None
    url: str | None
    source_quality: float | None
    timestamp: str | None

@dataclass(frozen=True)
class EducationLevel:
    """Defines grade-specific parameters for content adaptation."""
    grade: int
    reading_level: Literal["basic", "intermediate", "advanced"]
    vocabulary_level: Literal["simple", "moderate", "advanced"]
    explanation_style: Literal["storytelling", "conceptual", "technical"]

    @classmethod
    @lru_cache(maxsize=12)
    def from_grade(cls, grade: int) -> "EducationLevel":
        if grade <= 4:
            return cls(grade, "basic", "simple", "storytelling")
        elif grade <= 8:
            return cls(grade, "intermediate", "moderate", "conceptual")
        else:
            return cls(grade, "advanced", "advanced", "technical")

class WebCrawler:
    """Enhanced web crawler with Google Custom Search Engine integration."""

    def __init__(self, cache_dir: str = ".cache"):
        self.api_key = os.getenv("GOOGLE_API_KEY")
        self.cse_id = os.getenv("GOOGLE_CSE_ID")
        self.trusted_domains = {
            'wikipedia.org': 0.9,
            'britannica.com': 0.9,
            'khanacademy.org': 0.95,
            'nasa.gov': 0.9,
            'nationalgeographic.com': 0.85,
            'sciencedaily.com': 0.8,
            'education.com': 0.75,
            'scholastic.com': 0.85,
            'nature.com': 0.95,
            'science.org': 0.95
        }
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        # Initialize HTTP client with optimized settings
        self.http_client = httpx.AsyncClient(
            timeout=30.0,
            follow_redirects=True,
            verify=False,
            limits=httpx.Limits(max_keepalive_connections=10, max_connections=20),
            headers=self.headers
        )

    def search_urls(self, query: str, num_results: int = 4) -> List[str]:
        """Get links from Google Custom Search Engine or fallback to googlesearch."""
        try:
            # First try using Google Custom Search API if credentials are available
            if self.api_key and self.cse_id:
                search_url = "https://www.googleapis.com/customsearch/v1"
                params = {
                    'q': query,
                    'cx': self.cse_id,
                    'key': self.api_key,
                    'num': num_results
                }
                response = httpx.get(search_url, params=params)
                results = response.json()
                if 'items' in results:
                    return [item['link'] for item in results['items']]

            # Fallback to googlesearch library
            logger.info("Falling back to googlesearch library")
            search_results = []
            for url in search(query, stop=num_results, pause=2.0):
                domain = urlparse(url).netloc.lower()
                if any(trusted in domain for trusted in self.trusted_domains):
                    search_results.append(url)
            return search_results[:num_results]

        except Exception as e:
            logger.error(f"Search error: {e}")
            return []

    async def _extract_markdown(self, html: str, url: str) -> str:
        """Extract and format content as markdown."""
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
        """Fetch and parse content from a URL with improved error handling."""
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

    async def crawl(self, query: str, num_results: int = 4) -> List[WebContent]:
        """Perform parallel crawling of search results with improved handling."""
        urls = self.search_urls(query, num_results)
        if not urls:
            return []

        tasks = [self.fetch_url(url) for url in urls]
        results = await asyncio.gather(*tasks)

        # Filter out None results and sort by source quality
        valid_results = [r for r in results if r is not None and r.content]
        valid_results.sort(key=lambda x: x.source_quality or 0, reverse=True)

        return valid_results[:num_results]

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

    # Helper methods
    async def _cache_key(self, url: str) -> str:
        return hashlib.md5(url.encode()).hexdigest()

    async def _get_from_cache(self, url: str) -> Optional[WebContent]:
        try:
            cache_key = await self._cache_key(url)
            cache_file = self.cache_dir / f"{cache_key}.json"
            if cache_file.exists():
                async with aiofiles.open(cache_file, 'r') as f:
                    data = json.loads(await f.read())
                    return WebContent(**data)
        except Exception as e:
            logger.error(f"Cache read error for {url}: {e}")
        return None

    async def _save_to_cache(self, content: WebContent) -> None:
        try:
            if content.url:
                cache_key = await self._cache_key(content.url)
                cache_file = self.cache_dir / f"{cache_key}.json"
                async with aiofiles.open(cache_file, 'w') as f:
                    await f.write(json.dumps(content.__dict__))
        except Exception as e:
            logger.error(f"Cache write error: {e}")

    def get_domain_score(self, url: str) -> float:
        domain = urlparse(url).netloc.lower()
        return next((score for trusted_domain, score in self.trusted_domains.items()
                    if trusted_domain in domain), 0.5)

    async def close(self):
        """Clean up resources."""
        await self.http_client.aclose()


class EducationalLLMWrapper:
    """Enhanced LLM wrapper with improved error handling and streaming."""

    def __init__(self):
        logger.info("Initializing EducationalLLMWrapper")
        self._initialize_llms()
        self.web_crawler = WebCrawler()

    def _initialize_llms(self):
        """Initialize LLM clients with error handling."""
        try:
            genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
            self.model = genai.GenerativeModel('gemini-pro')
            self.cohere_client = cohere.Client(os.getenv("COHERE_API_KEY"))
        except Exception as e:
            logger.error(f"Failed to initialize LLMs: {e}")
            raise

    async def get_response(self, query: str, grade: int) -> AsyncIterator[str]:
        """Get educational response with streaming support."""
        try:
            context = await self.web_crawler.search_and_extract(query)
            education_level = EducationLevel.from_grade(grade)
            prompt = self._create_educational_prompt(query, grade, education_level, context)

            try:
                response = self.model.generate_content(prompt, stream=True)
                for chunk in response:
                    if chunk.text:
                        yield chunk.text
            except Exception as e:
                logger.info(f"Falling back to Cohere due to: {e}")
                try:
                    response = self.cohere_client.generate(
                        prompt=prompt,
                        max_tokens=300,
                        temperature=0.3
                    )
                    yield response.generations[0].text
                except Exception as e:
                    logger.error(f"Cohere error: {e}")
                    yield "I apologize, but I'm having trouble generating a response."
        except Exception as e:
            logger.error(f"Error getting response: {e}")
            yield "I apologize, but I encountered an error. Please try again."

    def _create_educational_prompt(self, query: str, grade: int, education_level: EducationLevel, context: str) -> str:
        print("Creating educational prompt",context)
        format_guidelines = (
            '- Use simple words and short sentences\n- Include storytelling elements\n- Word limit: 150-200 words'
            if grade <= 5 else
            '- Use moderate vocabulary\n- Include real-world applications\n- Word limit: 200-300 words'
            if grade <= 8 else
            '- Use subject-specific terminology\n- Focus on analytical thinking\n- Word limit: 300-400 words'
            )

        return f'''
                            You are an experienced CBSE teacher helping a Grade {grade} student understand a concept.
                            Remember to maintain strict focus on CBSE curriculum guidelines.

                            STUDENT CONTEXT:
                            - Grade Level: {grade}
                            - Subject Area: {context}
                            - Question: {query}

                            RESPONSE STRUCTURE:
                            1. Title: Start with a clear, engaging title for the topic

                            2. Introduction (2-3 sentences):
                            - Hook the student's interest
                            - Connect to prior knowledge
                            - State what they will learn

                            3. Main Explanation:
                            - Break down complex concepts into simple parts
                            - Use bullet points for clarity
                            - Include visual descriptions where helpful
                            - Stay within grade-appropriate vocabulary

                            4. Real-World Application(if required then only):
                            - Provide exactly 2 relatable examples
                            - Use scenarios from daily life
                            - Connect to student's experiences

                            5. Key Points to Remember (exactly 3 points):
                            - Use simple, memorable language
                            - Focus on essential concepts
                            - Make it exam-relevant

                            6. Practice Question(if required then only):
                            - One grade-appropriate question
                            - Include step-by-step solution approach
                            - Match CBSE exam pattern

                            FORMAT GUIDELINES:
                            Grade {grade} Specific Approach:
                            {format_guidelines}

                            IMPORTANT RULES:
                            1. Never exceed the word limit
                            2. Always maintain CBSE curriculum alignment
                            3. Use clear section headings
                            4. Keep language consistent with grade level
                            5. Focus only on the asked topic
                            6. Avoid tangential information

                            Remember to present information in a visually structured format using proper spacing and bullet points for better readability.
                            '''



    async def close(self):
        """Clean up resources."""
        await self.web_crawler.close()

class StreamlitEducationalApp:
    """Streamlit interface with improved async handling."""

    def __init__(self):
        self.llm = EducationalLLMWrapper()
        self.setup_page_config()
        self.initialize_session_state()
        self.create_custom_theme()

    def setup_page_config(self):
        st.set_page_config(
            page_title="Educational AI Assistant",
            page_icon="📚",
            layout="wide"
        )

    def initialize_session_state(self):
        if "messages" not in st.session_state:
            st.session_state.messages = []
        if "current_grade" not in st.session_state:
            st.session_state.current_grade = 6

    def create_custom_theme(self):
        st.markdown("""
            <style>
            .stTextInput > div > div > input {
                border-radius: 10px;
            }
            .stButton > button {
                border-radius: 10px;
                background-color: #4CAF50;
                color: white;
                transition: all 0.3s ease;
            }
            .stButton > button:hover {
                background-color: #45a049;
                transform: translateY(-2px);
            }
            .stMarkdown {
                font-size: 16px;
                line-height: 1.6;
            }
            </style>
        """, unsafe_allow_html=True)

    def setup_sidebar(self):
        with st.sidebar:
            st.title("Settings")
            st.session_state.current_grade = st.slider(
                "Select Grade Level",
                min_value=1,
                max_value=12,
                value=st.session_state.current_grade,
                help="Adjust the grade level to get age-appropriate responses"
            )

            education_level = EducationLevel.from_grade(st.session_state.current_grade)
            st.write("Current Education Level:")
            st.write(f"- Reading: {education_level.reading_level}")
            st.write(f"- Vocabulary: {education_level.vocabulary_level}")
            st.write(f"- Style: {education_level.explanation_style}")

            if st.button("Clear Chat History"):
                st.session_state.messages = []
                st.rerun()

    def display_chat_interface(self):
        st.title("Educational AI Assistant 📚")
        st.write(f"Currently helping Grade {st.session_state.current_grade} students")

        for msg in st.session_state.messages:
            message(
                msg["content"],
                is_user=msg["role"] == "user",
                key=str(msg["timestamp"])
            )

    async def process_user_input(self, prompt: str):
        st.session_state.messages.append({
            "role": "user",
            "content": prompt,
            "timestamp": time.time()
        })

        message_placeholder = st.empty()
        full_response = []

        async for chunk in self.llm.get_response(prompt, st.session_state.current_grade):
            full_response.append(chunk)
            message_placeholder.markdown(''.join(full_response) + "▌")

        full_response_text = ''.join(full_response)
        message_placeholder.markdown(full_response_text)

        st.session_state.messages.append({
            "role": "assistant",
            "content": full_response_text,
            "timestamp": time.time()
        })

    def run(self):
        try:
            self.setup_sidebar()
            self.display_chat_interface()

            if prompt := st.chat_input("Ask me anything! I'm here to help you learn 🌟"):
                with st.chat_message("user"):
                    st.markdown(prompt)
                with st.chat_message("assistant"):
                    asyncio.run(self.process_user_input(prompt))

        except Exception as e:
            logger.error(f"Application error: {e}")
            st.error("An error occurred. Please try again.")

        finally:
            if hasattr(self, 'llm'):
                asyncio.run(self.llm.close())

def main():
    """Main entry point with proper error handling."""
    try:
        if os.name == 'nt':
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

        app = StreamlitEducationalApp()
        app.run()
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        st.error("A fatal error occurred. Please refresh the page and try again.")

if __name__ == "__main__":
    main()
