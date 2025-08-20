# DashChatbot – PDF Q&A (LangChain + RAG)

Two Streamlit demos:

1. `dashQA/langchain_qa.py` – FAQ Bot for short/structured PDFs (answers with citations).
2. `novelChatbot/web_bot.py` – Long-text Q&A for novels/long docs (characters/plot/themes).

## Run locally

```bash
pip install -r requirements.txt
export OPENAI_API_KEY=sk-...
streamlit run dashQA/dash.py
# or
streamlit run novelChatbot/app.py
```

## Deploy (Streamlit Cloud)

1. Push this repo to GitHub.
2. On [Streamlit Cloud](https://streamlit.io/cloud), create **two apps** pointing to the entry files above:
   - `dashQA/langchain_qa.py`
   - `novelChatbot/web_bot.py`
3. In each app, set secrets:
   ```
   OPENAI_API_KEY = sk-xxxx
   # Optional: OPENAI_BASE_URL = https://your-gateway/v1
   ```

## Notes

- FAQ Bot is optimized for short, structured PDFs (like app FAQs).
- Long-text Q&A supports novels or large docs, retrieves passages, and provides detailed contextual answers.
- Upload your own PDF to test each demo.
