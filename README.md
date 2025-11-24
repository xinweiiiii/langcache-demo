---
title: Langcache Demo
emoji: 🔥
colorFrom: red
colorTo: blue
sdk: gradio
sdk_version: 5.49.0
app_file: app.py
pinned: false
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

# Redis LangCache — Simple Demo

A simple demo showing **Redis LangCache** + **OpenAI** implementing **semantic caching** with **LangCache Redis Cloud**.

---

## ✨ What This Demo Does

- Demonstrates **semantic caching** for LLM responses to reduce **latency** and **API cost**
- Shows real-time **cache hits** and **cache misses**
- **Scoped caching** by **Company / Business Unit / Person** with adjustable isolation levels
- **Gradio web interface** for easy testing and visualization

---

## 📁 Project Structure

```
.
├── app.py                           # Simple UI chatbot with LangCache
├── requirements.txt                 # Python dependencies
└── .env                             # Environment variables
```

---

## 🔐 Environment Variables

Create a `.env` file in the project root with:

```env
# OpenAI
OPENAI_API_KEY=sk-proj-<your-openai-key>
OPENAI_MODEL=gpt-4o-mini

# LangCache (Redis Cloud)
LANGCACHE_SERVICE_KEY=<your-service-key>
LANGCACHE_CACHE_ID=<your-cache-id>
LANGCACHE_BASE_URL=https://gcp-us-east4.langcache.redis.io
```

> **Note:** Get your LangCache credentials from [Redis Cloud Console](https://redis.io/)

---

## 🚀 Running the Demo

### 1) Install Dependencies

```bash
python -m venv .venv
source .venv/bin/activate      # Linux/Mac
# .venv\Scripts\activate       # Windows PowerShell
pip install -r requirements.txt
```

### 2) Run Simple CLI Demo

```bash
python main.py
```

This will start a command-line chatbot where you can:
- Ask questions
- See **[CACHE HIT]** or **[CACHE MISS]** for each query
- View latency improvements from caching

## 🧑‍💻 Using the Gradio UI

1. Set **Company**, **Business Unit**, and **Person** for both **Scenario A and B**
2. Ask questions in both panels to observe **cache hits/misses**
3. Try asking the **same question** in both scenarios with different isolation settings
4. Use the **🧹 Clear Cache** buttons to delete entries by scope

**Recommended test questions:**

- "What is machine learning?"
- "Explain what is a neural network"
- "What is Python programming?"
- "Explain cloud computing"

**Test semantic caching** by asking similar questions:
- First: "What is machine learning?"
- Then: "Explain machine learning to me" ← Should hit cache!

---

## 🧠 How It Works

1. **Search** Redis LangCache for semantically similar prompts
2. If a **cache hit** (above threshold) is found, return the cached response instantly
3. If a **cache miss** occurs:
   - Query OpenAI LLM
   - Store the response in LangCache
   - Return the response
4. Isolation is managed via attributes: `company`, `business_unit`, and `person`

---

## 📊 Benefits of Semantic Caching

- **Reduced Latency**: Cache hits return in milliseconds vs seconds for LLM calls
- **Cost Savings**: Avoid redundant API calls for similar questions
- **Better UX**: Faster responses improve user experience
- **Scalability**: Handle more users without proportional cost increase

---

## 🔗 Useful Links

- **Redis LangCache Documentation:** https://redis.io/docs/latest/solutions/semantic-caching/langcache/
- **Redis Website:** https://redis.io/
- **LinkedIn (Gabriel Cerioni):** https://www.linkedin.com/in/gabrielcerioni/

