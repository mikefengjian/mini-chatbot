import os
import pickle
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS


def extract_text_from_pdf(pdf_path, min_line_length=10):
    from langchain_community.document_loaders import PyPDFLoader
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    return [doc.page_content for doc in docs if len(doc.page_content) >= min_line_length]


def get_completion(prompt):
    from openai import OpenAI
    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": "You are a helpful assistant."},
                  {"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content


def build_prompt(question, retrieved_docs):
    context = "\n\n".join(retrieved_docs)
    return f"Answer the following question based on the provided context:\n\nContext:\n{context}\n\nQuestion: {question}"


def identify_question_type(question):
    q = question.lower()
    if "who" in q or "character" in q or "personality" in q:
        return "character"
    elif "plot" in q or "what happened" in q:
        return "plot"
    elif "where" in q or "setting" in q:
        return "setting"
    elif "theme" in q:
        return "theme"
    else:
        return "general"


class RagBot:
    def __init__(self, vector_db, llm_api=get_completion, n_results=5):
        self.vector_db = vector_db
        self.llm_api = llm_api
        self.n_results = n_results

    def chat(self, user_query):
        question_type = identify_question_type(user_query)

        if question_type == "plot":
            self.n_results = 8
        elif question_type == "character":
            self.n_results = 6
        else:
            self.n_results = 4

        results = self.vector_db.similarity_search(user_query, k=self.n_results)
        docs = [doc.page_content for doc in results]

        prompt = build_prompt(user_query, docs)
        return self.llm_api(prompt)
