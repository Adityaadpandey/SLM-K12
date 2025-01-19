from __future__ import annotations
from typing import Literal, TypedDict, AsyncIterator
import asyncio
import os
import json
import logfire
from supabase import Client, create_client
import google.generativeai as genai
import streamlit as st
from dotenv import load_dotenv
from dataclasses import dataclass

load_dotenv()

# Initialize Google Gemini
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel('gemini-pro')

# Initialize Supabase
supabase = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_KEY")
)

# Configure logfire
logfire.configure(send_to_logfire='never')

class ChatMessage(TypedDict):
    """Format of messages sent to the browser/API."""
    role: Literal['user', 'assistant', 'system']
    content: str
    timestamp: str

class GeminiChat:
    """Wrapper for Gemini chat functionality."""

    def __init__(self, model):
        self.model = model
        self.chat = model.start_chat(history=[])

    async def send_message(self, content: str) -> AsyncIterator[str]:
        """Stream responses from Gemini."""
        try:
            response = await asyncio.to_thread(
                self.chat.send_message,
                content,
                stream=True
            )

            async for chunk in self._async_iterator(response):
                if chunk.text:
                    yield chunk.text

        except Exception as e:
            print(f"Error in send_message: {e}")
            yield f"Error: {str(e)}"

    async def _async_iterator(self, response):
        """Convert synchronous iterator to async iterator."""
        for chunk in response:
            yield chunk
            await asyncio.sleep(0)

@dataclass
class StreamingMessage:
    """Helper class for streaming messages."""
    content: str = ""
    is_complete: bool = False

class DocumentationChat:
    def __init__(self, supabase_client: Client, gemini_model):
        self.supabase = supabase_client
        self.gemini = GeminiChat(gemini_model)
        self.current_message = StreamingMessage()

    async def process_user_message(self, user_input: str) -> AsyncIterator[str]:
        """Process user input and generate streaming response."""
        try:
            # First, get relevant documentation
            result = self.supabase.rpc(
                'match_site_pages',
                {
                    'query_embedding': await self.get_embedding(user_input),
                    'match_count': 3,
                    'filter': {'source': 'pydantic_ai_docs'}
                }
            ).execute()

            context = ""
            if result.data:
                context = "\n\n".join([
                    f"Title: {doc['title']}\nContent: {doc['content'][:500]}..."
                    for doc in result.data
                ])

            # Construct prompt with context
            #
            if context=="": # If no relevant information found in the documentation
                prompt = f"""Based on the following documentation context, answer the user's question.
                If the context doesn't contain relevant information, say so.

                Context:
                {context}

                User Question: {user_input}

                Please provide a clear, detailed answer focusing on the specific information from the documentation. and if in case you don't find any relevant information, find in from your expertise as your will be working as a Teacher and Mentor to Class 1 to 12 students provide them accurate information and guidance."""
            else:
                prompt = f"""Based on the following documentation context, answer the user's question.
                  If the context doesn't contain relevant information, say so.

                  Context:
                  {context}

                  User Question: {user_input}

                  Please provide a clear, detailed answer focusing on the specific information from the documentation. and if in case you don't find any relevant information, find in from your expertise as your will be working as a Teacher and Mentor to Class 1 to 12 students provide them accurate information and guidance."""

            # Stream response using Gemini
            async for chunk in self.gemini.send_message(prompt):
                yield chunk

        except Exception as e:
            print(f"Error processing message: {e}")
            yield f"I encountered an error: {str(e)}"

    async def get_embedding(self, text: str) -> list[float]:
        """Get embedding from Cohere."""
        try:
            import cohere
            client = cohere.Client(os.getenv("COHERE_API_KEY"))

            response = await asyncio.to_thread(
                client.embed,
                texts=[text],
                model='embed-multilingual-v3.0',
                input_type='search_document'
            )

            embedding = response.embeddings[0]
            if len(embedding) != 1536:
                if len(embedding) < 1536:
                    embedding = embedding + [0] * (1536 - len(embedding))
                else:
                    embedding = embedding[:1536]

            return embedding
        except Exception as e:
            print(f"Error getting embedding: {e}")
            return [0] * 1536

async def main():
    st.title("Pydantic AI Documentation Assistant")
    st.write("Ask questions about Pydantic AI and get answers from the documentation.")

    # Initialize session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Initialize chat instance
    if "chat" not in st.session_state:
        st.session_state.chat = DocumentationChat(supabase, model)

    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    user_input = st.chat_input("Ask a question about Pydantic AI...")

    if user_input:
        # Display user message
        with st.chat_message("user"):
            st.markdown(user_input)

        # Add user message to history
        st.session_state.chat_history.append({
            "role": "user",
            "content": user_input,
            "timestamp": str(asyncio.get_event_loop().time())
        })

        # Display assistant's response with streaming
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            full_response = ""

            async for chunk in st.session_state.chat.process_user_message(user_input):
                full_response += chunk
                response_placeholder.markdown(full_response + "▌")

            response_placeholder.markdown(full_response)

            # Add assistant's response to history
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": full_response,
                "timestamp": str(asyncio.get_event_loop().time())
            })

if __name__ == "__main__":
    asyncio.run(main())
