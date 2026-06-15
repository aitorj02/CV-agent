# CV Enhancement Agent

A multi-agent system that analyzes your CV against a job description, identifies gaps, rewrites weak bullet points, and scores the match — built with LangChain, LangGraph, and Google Gemini.

## Architecture

```
User Input (CV + Job Description)
          │
          ▼
  ┌───────────────┐
  │  LangGraph    │  StateGraph orchestrating 5 agent nodes
  │  Orchestrator │
  └───────┬───────┘
          │
    ┌─────▼──────┐     ┌──────────────┐
    │ CV Parser  │     │ JD Analyzer  │
    │            │     │              │
    │ skills,    │     │ requirements,│
    │ experience,│     │ keywords,    │
    │ education  │     │ tech stack   │
    └─────┬──────┘     └──────┬───────┘
          └────────┬──────────┘
                   ▼
          ┌────────────────┐
          │  Gap Analyzer  │
          │                │
          │ missing skills,│
          │ weak bullets,  │
          │ ATS keywords   │
          └────────┬───────┘
                   ▼
          ┌────────────────┐
          │  CV Enhancer   │
          │                │
          │ rewrites       │
          │ bullets with   │
          │ action verbs   │
          └────────┬───────┘
                   ▼
          ┌────────────────┐
          │  ATS Scorer    │
          │                │
          │ before / after │
          │ match score    │
          └────────┬───────┘
                   ▼
        Output Report (Streamlit UI)
        Gaps · Enhanced Bullets · ATS Score
```

## Features

- **CV Parser** — extracts structured skills, experience, and education from raw CV text or PDF
- **JD Analyzer** — parses job descriptions for required skills, tech stack, and ATS keywords
- **Gap Analyzer** — identifies missing skills, weak bullet points, and absent ATS keywords
- **CV Enhancer** — rewrites experience bullets using strong action verbs and relevant keywords
- **ATS Scorer** — produces a before/after match score so you can see the improvement

## Tech Stack

| Layer | Tool |
|---|---|
| Agent orchestration | [LangGraph](https://github.com/langchain-ai/langgraph) |
| LLM chains & prompts | [LangChain](https://github.com/langchain-ai/langchain) |
| LLM provider | Google Gemini (`gemini-3.1-flash-lite`) |
| PDF parsing | pypdf |
| UI | Streamlit |
| Language | Python 3.11+ |

## Setup

1. **Clone the repo**
   ```bash
   git clone https://github.com/your-username/cv-agent.git
   cd cv-agent
   ```

2. **Create a virtual environment and install dependencies**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   make install
   ```

3. **Configure environment variables**
   ```bash
   cp .env.example .env
   # Edit .env and add your Google API key
   ```

   ```env
   GOOGLE_API_KEY=your_key_here
   GEMINI_MODEL=gemini-3.1-flash-lite   # optional, this is the default
   ```

## Usage

### Streamlit UI (recommended)
```bash
make run
```
Open `http://localhost:8501`, paste your CV and the job description, then click **Analyze & Enhance**.

Results appear in three tabs:
- **Gaps** — missing skills and keywords
- **Enhanced Bullets** — AI-rewritten experience bullets
- **ATS Score** — before/after match percentage

### CLI (quick test)
```bash
make run-cli
```
Runs the pipeline against sample data in `main.py` and prints the report to stdout.

## Running Tests
```bash
make test
```
15 unit tests covering all 5 agent nodes (LLM calls are mocked — no API key needed).

## Project Structure

```
cv_agent_project/
├── agents/               # One file per LangGraph node
│   ├── cv_parser.py
│   ├── jd_analyzer.py
│   ├── gap_analyzer.py
│   ├── cv_enhancer.py
│   └── ats_scorer.py
├── graph/
│   ├── state.py          # CVState TypedDict shared across all nodes
│   └── graph.py          # LangGraph StateGraph definition
├── tools/
│   ├── config.py         # Environment-based configuration
│   ├── llm_utils.py      # Provider-agnostic text extraction + JSON parsing
│   └── pdf_loader.py     # PDF → plain text via pypdf
├── tests/                # Unit tests (pytest + pytest-mock)
├── app.py                # Streamlit UI
├── main.py               # CLI entry point with sample data
└── pyproject.toml
```

## License

MIT — see [LICENSE](LICENSE).
