from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, AsyncIterator, Literal
import asyncio
import os
import json
from datetime import datetime
import httpx
import cohere
import google.generativeai as genai
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from urllib.parse import urljoin, urlparse
from numpy import conj
import streamlit as st
from streamlit_chat import message
import time
import nest_asyncio
import ssl


nest_asyncio.apply()
load_dotenv()

@dataclass
class WebContent:
    """Represents crawled web content."""
    title: str | None
    content: str | None
    url: str | None
    source_quality: float | None

class WebCrawler:
    """Handles web crawling functionality with educational focus."""

    def __init__(self):
        print("Initializing WebCrawler")
        # Educational domains to prioritize
        self.trusted_domains = {
            'wikipedia.org': 0.8,
            'britannica.com': 0.9,
            'khanacademy.org': 0.95,
            'nasa.gov': 0.9,
            'nationalgeographic.com': 0.85,
            'sciencedaily.com': 0.8,
            'education.com': 0.75,
            'scholastic.com': 0.85,
        }
        # Create a custom SSL context
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE

        # Increased timeout and added retry configuration
        self.session = httpx.AsyncClient(
            timeout=30.0,  # Increased timeout
            follow_redirects=True,
            verify=ssl_context,
            headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
            },
            limits=httpx.Limits(max_keepalive_connections=5, max_connections=10)
        )
        print(f"Initialized with {len(self.trusted_domains)} trusted domains")

    def get_domain_score(self, url: str) -> float:
        """Calculate domain reliability score."""
        domain = urlparse(url).netloc.lower()
        base_domain = '.'.join(domain.split('.')[-2:])
        return self.trusted_domains.get(base_domain, 0.5)

    async def clean_text(self, soup: BeautifulSoup) -> str:
        """Clean and extract relevant text from HTML."""
        # Remove unwanted elements
        for tag in soup(['script', 'style', 'nav', 'footer', 'header', 'aside']):
            tag.decompose()

        # Extract text from paragraphs and headers
        content = []
        for elem in soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'article']):
            text = elem.get_text().strip()
            if text and len(text) > 20:  # Filter out short snippets
                content.append(text)

        return '\n\n'.join(content)

    async def crawl_url(self, url: str) -> Optional[WebContent]:
        """Crawl a single URL and extract content."""
        print(f"\nCrawling URL: {url}")
        retries = 3

        for attempt in range(retries):
            try:
                response = await asyncio.wait_for(
                    self.session.get(url),
                    timeout=30
                )
                response.raise_for_status()
                print(f"Successfully fetched {url}")

                soup = BeautifulSoup(response.text, 'html.parser')
                title = soup.title.string if soup.title else url
                content = await self.clean_text(soup)

                if not content:
                    print(f"No content found for {url}")
                    return None

                result = WebContent(
                    title=title,
                    content=content[:2000],  # Limit content length
                    url=url,
                    source_quality=self.get_domain_score(url)
                )
                print(f"Successfully extracted content from {url} (quality: {result.source_quality})")
                return result

            except asyncio.TimeoutError:
                print(f"Timeout on attempt {attempt + 1} for {url}")
                if attempt == retries - 1:
                    print(f"All retries failed for {url}")
                    return None
                await asyncio.sleep(1)  # Wait before retrying

            except Exception as e:
                print(f"Error crawling {url} on attempt {attempt + 1}: {str(e)}")
                if attempt == retries - 1:
                    print(f"All retries failed for {url}")
                    return None
                await asyncio.sleep(1)

    async def search(self, query: str, num_results: int = 3) -> List[WebContent]:
        """Perform web search using DuckDuckGo and crawl results."""
        print(f"\nStarting web search for query: {query}")

        # Using a more reliable search URL format
        search_url = (
            "https://duckduckgo.com/html/?" +
            f"q={query}+site:({'+OR+'.join(self.trusted_domains.keys())})" +
            "&kl=us-en&kt=n"
        )

        retries = 3
        for attempt in range(retries):
            try:
                print(f"Sending request to DuckDuckGo (attempt {attempt + 1})")
                response = await asyncio.wait_for(
                    self.session.get(
                        search_url,
                        headers={'Cache-Control': 'no-cache'}
                    ),
                    timeout=30
                )
                response.raise_for_status()
                print("Successfully received search results")

                soup = BeautifulSoup(response.text, 'html.parser')
                result_links = []

                # Updated selector for DuckDuckGo results
                for result in soup.select('.result__url, .result__a'):
                    url = result.get('href')
                    if url:
                        full_url = urljoin('https://', url)
                        if any(domain in full_url for domain in self.trusted_domains):
                            print(f"Found result URL: {full_url}")
                            result_links.append(full_url)
                    if len(result_links) >= num_results:
                        break

                print(f"Found {len(result_links)} result links")

                if not result_links:
                    if attempt < retries - 1:
                        print("No results found, retrying...")
                        await asyncio.sleep(2)
                        continue
                    return []

                tasks = [self.crawl_url(url) for url in result_links]
                results = await asyncio.gather(*tasks)

                valid_results = [r for r in results if r is not None]
                valid_results.sort(key=lambda x: x.source_quality, reverse=True)

                print(f"Successfully processed {len(valid_results)} valid results")
                return valid_results[:num_results]

            except asyncio.TimeoutError:
                print(f"Search timeout on attempt {attempt + 1}")
                if attempt < retries - 1:
                    await asyncio.sleep(2)
                    continue
                return []

            except Exception as e:
                print(f"Search error on attempt {attempt + 1}: {str(e)}")
                if attempt < retries - 1:
                    await asyncio.sleep(2)
                    continue
                return []

        return []

    async def close(self):
        """Close the HTTP session."""
        await self.session.aclose()

@dataclass
class EducationLevel:
    """Defines grade-specific parameters for content adaptation."""
    grade: int
    reading_level: Literal["basic", "intermediate", "advanced"]
    vocabulary_level: Literal["simple", "moderate", "advanced"]
    explanation_style: Literal["storytelling", "conceptual", "technical"]

    @classmethod
    def from_grade(cls, grade: int) -> "EducationLevel":
        if grade <= 4:
            return cls(grade, "basic", "simple", "storytelling")
        elif grade <= 8:
            return cls(grade, "intermediate", "moderate", "conceptual")
        else:
            return cls(grade, "advanced", "advanced", "technical")

class EducationalLLMWrapper:
    """Enhanced LLM wrapper with improved error handling."""

    def __init__(self):
        print("\nInitializing EducationalLLMWrapper")
        try:
            genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
            self.model = genai.GenerativeModel('gemini-pro')
            print("Successfully initialized Gemini model")

            self.cohere_client = cohere.Client(os.getenv("COHERE_API_KEY"))
            print("Successfully initialized Cohere model")

            self.web_crawler = WebCrawler()
            print("Successfully initialized WebCrawler")

        except Exception as e:
            print(f"Error initializing EducationalLLMWrapper: {str(e)}")
            raise

    async def get_web_content(self, query: str) -> str:
        """Get content from web crawling."""
        results = await self.web_crawler.search(query)
        if not results:
            return ""

        content = "\n\n".join([
            f"Source: {result.url}\nTitle: {result.title}\nContent: {result.content}"
            for result in results
        ])
        print("content of the web search:",content)
        return content

    async def _try_gemini_response(self, prompt: str) -> AsyncIterator[str]:
        """Attempt to get a response from Gemini."""
        chat = self.model.start_chat(history=[])

        try:
            response = chat.send_message(prompt)
            if response.text:
                yield response.text
        except Exception as e:
            print(f"Gemini error: {str(e)}")
            raise

    async def _try_cohere_response(self, prompt: str) -> str:
        """Get a response from Cohere as backup."""
        try:
            response = await asyncio.to_thread(
                self.cohere_client.generate,
                prompt=prompt,
                max_tokens=500,
                temperature=0.7,
                k=0,
                stop_sequences=[],
                return_likelihoods='NONE'
            )
            return response.generations[0].text
        except Exception as e:
            print(f"Cohere error: {str(e)}")
            raise

    async def generate_response(self, query: str, grade: int) -> AsyncIterator[str]:
        """Generate educational response with fallback options."""
        try:
            context = await self.get_web_content(query)
            education_level = EducationLevel.from_grade(grade)

            prompt = f"""
            You are a knowledgeable and engaging teacher, responsible for explaining concepts to grade {grade} students.
            Your explanation should be tailored to meet the specific educational needs of these students by considering the following:

            Reading Level: {education_level.reading_level}
            Vocabulary Level: {education_level.vocabulary_level}
            Explanation Style: {education_level.explanation_style}
            Context:
            {context}

            Question:
            {query}

            Your Task:
            Provide a detailed, accurate, and engaging explanation of the topic that is clear and age-appropriate for grade {grade} students.
            Make sure your response is long enough to thoroughly cover the topic, using simple language and relatable examples.
            Incorporate real-life scenarios where possible to help students easily grasp the idea.
            Focus on making the content engaging, clear, and easy to understand, ensuring that the students can fully comprehend the concept.


            """

            try:
                async for chunk in self._try_gemini_response(prompt):
                    yield chunk
            except Exception as e:
                print("Falling back to Cohere")
                try:
                    backup_response = await self._try_cohere_response(prompt)
                    yield backup_response
                except Exception as e:
                    yield "I apologize, but I'm having trouble generating a response. Please try again."

        except Exception as e:
            print(f"Error in generate_response: {str(e)}")
            yield "I apologize, but I encountered an error. Please try again."

    async def close(self):
        """Clean up resources."""
        await self.web_crawler.close()

class StreamlitEducationalApp:
    """Streamlit interface for the Educational LLM Wrapper."""

    def __init__(self):
        self.llm = EducationalLLMWrapper()
        self.setup_page_config()
        self.initialize_session_state()
        self.create_custom_theme()

    def setup_page_config(self):
        try:
            st.set_page_config(
                page_title="Educational AI Assistant",
                page_icon="📚",
                layout="wide"
            )
        except:
            pass

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
            }
            .stButton > button:hover {
                background-color: #45a049;
            }
            .stMarkdown {
                font-size: 16px;
            }
            </style>
        """, unsafe_allow_html=True)

    def initialize_session_state(self):
        if "messages" not in st.session_state:
            st.session_state.messages = []
        if "current_grade" not in st.session_state:
            st.session_state.current_grade = 6

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

        placeholder = st.empty()
        full_response = ""

        try:
            async for response in self.llm.generate_response(prompt, st.session_state.current_grade):
                if response:
                    full_response = response
                    placeholder.markdown(full_response + "▌")

            placeholder.markdown(full_response)

            if full_response:
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": full_response,
                    "timestamp": time.time()
                })

        except Exception as e:
            print(f"Error: {str(e)}")
            placeholder.markdown("I apologize, but I encountered an error. Please try again.")

    def run(self):
        try:
            self.setup_sidebar()
            self.display_chat_interface()

            if prompt := st.chat_input("Ask me anything! I'm here to help you learn 🌟"):
                message(prompt, is_user=True)
                with st.chat_message("assistant"):
                    asyncio.run(self.process_user_input(prompt))

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

        finally:
            if hasattr(self, 'llm'):
                asyncio.run(self.llm.close())

def main():
    try:
        app = StreamlitEducationalApp()
        app.run()
    except Exception as e:
        st.error("A fatal error occurred. Please check the logs and try again.")
        print(f"Fatal error: {str(e)}")

if __name__ == "__main__":
    main()
