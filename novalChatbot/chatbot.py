import os
import openai
from pypdf import PdfReader
import chromadb
from chromadb.config import Settings

# Set OpenAI API key via environment variable
client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Prompt template used to guide the LLM's answer style and scope
novel_prompt_template = """As a literary assistant, please answer the user's question based on the following text passages from the novel.
Consider these guidelines:
1. For character questions: Analyze their personality, actions, and development
2. For plot questions: Provide relevant story details and context
3. For theme questions: Connect different parts of the text to explain deeper meanings
4. For setting questions: Describe locations and time periods in detail

If you find partial information, share what you found and indicate what's missing.
If you cannot find the information, explain why it might not be in the current passages.

Text passages:
{context}

User question: {query}

Let me analyze the available text and provide a detailed answer:"""


def get_completion(prompt, model="gpt-4o-mini"):
    """Call OpenAI Chat Completions and return the text output."""
    messages = [{"role": "user", "content": prompt}]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.5,
    )
    return response.choices[0].message.content


def build_prompt(template, context, query):
    """Build the full prompt by merging multiple retrieved context passages."""
    context_text = "\n\n".join(context)
    return template.format(
        context=context_text,
        query=query
    )


def identify_question_type(query: str) -> str:
    """Identify question type to adjust retrieval strategy."""
    query = query.lower()
    if any(word in query for word in ['who', 'character', 'personality']):
        return 'character'
    elif any(word in query for word in ['what happened', 'when', 'how']):
        return 'plot'
    elif any(word in query for word in ['where', 'location', 'place']):
        return 'setting'
    elif any(word in query for word in ['why', 'theme', 'meaning']):
        return 'theme'
    return 'general'


def extract_text_from_pdf(filename: str, page_numbers: list = None, min_line_length: int = 1):
    """Extract text from a PDF file with additional paragraph handling."""
    try:
        reader = PdfReader(filename)
        paragraphs = []
        current_paragraph = []

        # Add overlap handling for long paragraphs
        overlap_size = 100  # number of overlapping characters

        print(f"Processing PDF, total pages: {len(reader.pages)}")

        for page_num in range(len(reader.pages)):
            text = reader.pages[page_num].extract_text()

            # Process paragraphs line by line
            lines = text.split('\n')
            for line in lines:
                line = line.strip()

                # Skip empty lines or page numbers; flush current paragraph if any
                if not line or line.isdigit():
                    if current_paragraph:
                        text = ' '.join(current_paragraph)
                        # Split long paragraph with overlap
                        if len(text) > 1000:
                            words = text.split()
                            for i in range(0, len(words), 900):
                                chunk = ' '.join(words[max(0, i - overlap_size):i + 900])
                                if len(chunk) > 100:  # minimal paragraph length
                                    paragraphs.append(chunk)
                        else:
                            paragraphs.append(text)
                        current_paragraph = []
                    continue

                # Detect chapter headings and flush current paragraph before adding the heading
                if (line.lower().startswith('chapter') or
                        len(line) < 50 and line.isupper()):
                    if current_paragraph:
                        text = ' '.join(current_paragraph)
                        if len(text) > 100:
                            paragraphs.append(text)
                        current_paragraph = []
                    paragraphs.append(line)
                    continue

                # Accumulate content into the current paragraph
                current_paragraph.append(line)

        # Flush any remaining paragraph at EOF
        if current_paragraph:
            text = ' '.join(current_paragraph)
            if len(text) > 100:
                paragraphs.append(text)

        print(f"Extracted {len(paragraphs)} paragraphs")
        return paragraphs

    except Exception as e:
        print(f"Error processing PDF: {str(e)}")
        return []


class OpenAIEmbeddingFunction:
    """Thin wrapper around OpenAI embeddings to match Chroma's expected callable signature."""

    def __init__(self, model_name="text-embedding-ada-002", dimensions=None):
        self.model_name = model_name
        self.dimensions = dimensions

    def __call__(self, input: list[str]) -> list[list[float]]:
        # Don't pass dimensions for older models that don't support it
        if self.model_name == "text-embedding-ada-002":
            self.dimensions = None
        if self.dimensions:
            data = client.embeddings.create(
                input=input, model=self.model_name, dimensions=self.dimensions).data
        else:
            data = client.embeddings.create(input=input, model=self.model_name).data
        return [x.embedding for x in data]


class MyVectorDBConnector:
    """Convenience wrapper for a persistent Chroma collection."""

    def __init__(self, collection_name, embedding_function, persist_directory="./chroma_db"):
        self.embedding_fn = embedding_function
        self._cache = {}
        self._cache_size = 100

        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                is_persistent=True
            )
        )

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=embedding_function
        )

    def add_documents(self, documents, ids=None):
        """Clean, (re)chunk, and add documents to the collection."""
        if not documents:
            return

        cleaned_docs = []
        for doc in documents:
            # Normalize whitespace
            cleaned = " ".join(doc.split())
            if cleaned:
                if len(cleaned) > 1000:
                    # Secondary chunking by Chinese full stop '。' to cap at ~1000 chars
                    sentences = cleaned.split('。')
                    current_chunk = []
                    current_length = 0

                    for sentence in sentences:
                        sentence = sentence.strip() + '。'
                        if current_length + len(sentence) > 1000:
                            if current_chunk:
                                cleaned_docs.append(' '.join(current_chunk))
                            current_chunk = [sentence]
                            current_length = len(sentence)
                        else:
                            current_chunk.append(sentence)
                            current_length += len(sentence)

                    if current_chunk:
                        cleaned_docs.append(' '.join(current_chunk))
                else:
                    cleaned_docs.append(cleaned)

        if not cleaned_docs:
            return

        # Auto-generate ids if not provided
        if ids is None:
            ids = [str(i) for i in range(len(cleaned_docs))]

        # Batch insert to the collection
        batch_size = 100
        for i in range(0, len(cleaned_docs), batch_size):
            batch_docs = cleaned_docs[i:i + batch_size]
            batch_ids = ids[i:i + batch_size]

            self.collection.add(
                documents=batch_docs,
                ids=batch_ids
            )

    def search(self, query, n_results=2):
        """Query the vector store and return top-n document texts (with a simple in-memory cache)."""
        # Check cache first
        cache_key = f"{query}_{n_results}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        results = self.collection.query(
            query_texts=[query],
            n_results=n_results
        )

        # Cache only the text payloads for simplicity
        self._cache[cache_key] = results['documents'][0]
        if len(self._cache) > self._cache_size:
            self._cache.pop(next(iter(self._cache)))

        return results['documents'][0]


class RagBot:
    """Minimal RAG bot: retrieve, build prompt, call LLM, and return the answer."""

    def __init__(self, vector_db: MyVectorDBConnector, llm_api: callable, n_results: int = 12):
        self.vector_db = vector_db
        self.llm_api = llm_api
        self.n_results = n_results
        self.context_window = 2  # currently unused; reserved for future heuristics
        self.max_tokens_per_chunk = 1000  # soft cap per retrieved chunk (approximation)

    def truncate_text(self, text, max_tokens=1000):
        """Roughly truncate text to stay within a token-like limit (heuristic)."""
        # Very rough estimate for English: ~1.3 tokens per word
        words = text.split()
        estimated_tokens = len(words) * 1.3
        if estimated_tokens > max_tokens:
            # Keep approximately max_tokens worth of content
            words = words[:int(max_tokens / 1.3)]
            return ' '.join(words)
        return text

    def chat(self, user_query):
        try:
            # Adjust retrieval depth based on question type
            question_type = identify_question_type(user_query)
            if question_type == 'plot':
                self.n_results = 8
            elif question_type == 'character':
                self.n_results = 6
            else:
                self.n_results = 4

            # Retrieve initial results (text-only)
            initial_results = self.vector_db.search(user_query, self.n_results)

            # Truncate per-chunk and control overall prompt length
            processed_results = []
            total_length = 0

            for result in initial_results:
                truncated_text = self.truncate_text(result, self.max_tokens_per_chunk)
                estimated_new_length = total_length + len(truncated_text.split())

                # Keep headroom for the system prompt and the model's answer
                if estimated_new_length * 1.3 < 6000:
                    processed_results.append(truncated_text)
                    total_length = estimated_new_length
                else:
                    break

            # Build final prompt and call the LLM
            prompt = build_prompt(
                novel_prompt_template,
                context=processed_results,
                query=user_query
            )

            response = self.llm_api(prompt)
            if not response.strip():
                return "I apologize, but I cannot generate a valid response."
            return response

        except Exception as e:
            if "context_length_exceeded" in str(e):
                return "I apologize, but the text is too long. I'll try to provide a response based on a smaller portion of the text. Please try asking a more specific question."
            return f"Error generating response: {str(e)}"
