## AI Blog Writing Agent

An end-to-end **AI-powered blog generation system** built using **LangGraph + Ollama + Streamlit**.
It automatically plans, researches (optional), writes, and formats high-quality technical blogs.

<img src="https://github.com/NeelContractor/BlogWritingAgentApp/blob/main/demo.png" width="70%" height="70%">

---

## Project Strucutre
```
.
├── backend.py          # Agent pipeline (LangGraph)
├── frontend.py         # Streamlit frontend
├── blog_outputs/       # Generated Blog .md files
│   ├── blog1.pdf
│   ├── blog2.pdf
│   └── blog3.pdf
├── .env
└── README.md
```

---

## Features

* **Smart Routing**

  * Decides if research is needed (`closed_book`, `hybrid`, `open_book`)

* **Optional Web Research**

  * Uses Tavily API for fresh and relevant content

* **Structured Blog Planning**

  * Generates a full outline (5–7 sections) with goals & word targets

* **Parallel Section Writing**

  * Each section is generated independently using LangGraph workers

* **Markdown + HTML Output**

  * Clean blog export with preview UI

* **Persistence**

  * Saves blogs locally and allows reloading past content

---

## Architecture

```
User Input (Topic)
        ↓
     Router
        ↓
   Research (optional)
        ↓
   Orchestrator (Plan)
        ↓
   Fanout → Workers (parallel writing)
        ↓
     Reducer (merge)
        ↓
   Final Blog Output
```

---

## Tech Stack

* **LangGraph** – workflow orchestration
* **Ollama** – local LLM
* **Streamlit** – UI
* **Tavily API** – web search

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run Ollama

```bash
ollama run llama3.2:1b
```

### 3. Set environment variables (optional)

```bash
export OLLAMA_MODEL=llama3.2:1b
export TAVILY_API_KEY=your_key        
```

### 4. Run app

```bash
streamlit run frontend.py
```

---

## TODO
* Add Image generation logic