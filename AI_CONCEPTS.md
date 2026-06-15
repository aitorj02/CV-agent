# AI Concepts Behind This Project

This document explains the core AI ideas that power the CV Enhancement Agent — what they are, why they matter, and exactly how they appear in this codebase.

---

## 1. Large Language Models (LLMs)

### What they are
A Large Language Model is a neural network trained on massive amounts of text. Through that training it learns statistical patterns in language: grammar, facts, reasoning chains, code, and more. Given a prompt (input text), it generates a continuation (output text) that is statistically likely to be useful and coherent.

LLMs do not "understand" text the way humans do — they predict the next token (roughly, the next word fragment) based on all the tokens that came before. Despite this, they exhibit emergent abilities: summarisation, translation, question answering, and structured data extraction, among others.

### Key properties relevant to this project
- **Instruction following** — modern LLMs are fine-tuned to follow instructions given in a system prompt. This lets us tell the model exactly what format to return.
- **Structured output** — by prompting the model to return only valid JSON, we can extract machine-readable data from unstructured CV and job-description text.
- **Context window** — LLMs process a fixed amount of text at once (the context window). For this MVP the CV and JD are short enough to fit comfortably.

### Which LLM is used here
This project uses **Google Gemini `gemini-3.1-flash-lite`**, accessed via the Google AI API.

Gemini is Google's family of multimodal LLMs. The `flash-lite` variant is optimised for speed and cost efficiency — ideal for a pipeline that makes multiple sequential LLM calls per user request. The model name is not hardcoded; it is read from the `GEMINI_MODEL` environment variable (`tools/config.py`), so swapping models requires only a one-line config change.

---

## 2. Prompt Engineering

### What it is
Prompt engineering is the practice of crafting the text you send to an LLM to get the output you want. Because LLMs are instruction-followers, the phrasing, structure, and constraints in a prompt directly control what the model produces.

### How it is used here
Every agent in this project uses a **two-part prompt**:

- **System prompt** — sets the model's role and output contract. Example from `agents/cv_parser.py`:
  > *"You are a CV parsing expert. Extract structured information from the CV text. Return ONLY valid JSON with these keys: skills (list of strings), experience (list of dicts…)"*

- **Human prompt** — provides the actual data to process (the raw CV text, or the job description).

The phrase "Return ONLY valid JSON" is a deliberate constraint. Without it, models often wrap their answer in prose ("Sure! Here is the JSON: …"), which breaks parsing. This is a common prompt engineering pattern for structured extraction.

The `cv_enhancer` agent adds a small amount of creative latitude (`temperature=0.3`) to produce more varied, natural-sounding rewrites. The other agents use `temperature=0` for deterministic, consistent extraction.

---

## 3. AI Agents

### What an agent is
In AI, an **agent** is a system that uses an LLM to take actions — not just answer a question, but do something: call a tool, read a file, write an output, or pass a result to another system. The LLM acts as the "brain" that decides what to do; the surrounding code acts as the "body" that executes it.

In this project the agents are **task-specialised**: each one does exactly one job, receives a well-defined input, and produces a well-defined output. This is sometimes called the single-responsibility principle applied to agents.

### The five agents in this project

| Agent | File | Role |
|---|---|---|
| CV Parser | `agents/cv_parser.py` | Reads raw CV text and extracts structured data: skills, experience history (title, company, duration, bullet points), education, and a summary |
| JD Analyzer | `agents/jd_analyzer.py` | Reads a job description and extracts: required skills, nice-to-have skills, tech stack, ATS keywords, and a role summary |
| Gap Analyzer | `agents/gap_analyzer.py` | Compares the CV and JD structured data and identifies: missing skills, weak or vague bullet points, and ATS keywords absent from the CV |
| CV Enhancer | `agents/cv_enhancer.py` | Rewrites the CV's experience bullet points using strong action verbs, natural keyword insertion, and quantified impact — targeting the gaps identified in the previous step |
| ATS Scorer | `agents/ats_scorer.py` | Scores the original CV and the enhanced version against the job description on a 0–100 scale, then assembles the final Markdown report |

Each agent node is implemented as a Python function that takes the shared state, calls the LLM once, parses the response, and returns updated state fields. The LLM call is made through a **LangChain chain** (`_chain = _prompt | _llm`).

---

## 4. LangChain

### What it is
[LangChain](https://github.com/langchain-ai/langchain) is a Python framework for building applications with LLMs. It provides abstractions for common patterns: prompt templates, LLM wrappers, output parsers, document loaders, and composable "chains."

### How it is used here
Three LangChain components appear in this project:

**`ChatPromptTemplate`** (`langchain_core.prompts`)
Defines reusable, parameterised prompt templates. Variables like `{cv_raw}` are filled in at call time. This separates the prompt structure from the data, making prompts easy to read and modify.

```python
_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a CV parsing expert. Return ONLY valid JSON…"),
    ("human", "Parse this CV:\n\n{cv_raw}"),
])
```

**`ChatGoogleGenerativeAI`** (`langchain_google_genai`)
LangChain's wrapper around the Google Gemini API. It handles authentication, request formatting, and response parsing. Because every LLM provider has a slightly different API, LangChain's wrappers let you swap providers by changing one import and one class name.

**The pipe operator `|`**
LangChain uses Python's pipe operator to compose components into a chain:
```python
_chain = _prompt | _llm
```
Calling `_chain.invoke({"cv_raw": text})` runs the data through the prompt template and then through the LLM, returning the response. This is LangChain's **LCEL** (LangChain Expression Language) — a declarative way to build processing pipelines.

---

## 5. LangGraph

### What it is
[LangGraph](https://github.com/langchain-ai/langgraph) is a library built on top of LangChain for orchestrating **multi-agent workflows** as a graph (specifically, a state machine). Each node in the graph is an agent or a processing step; edges define the order of execution and can be conditional.

LangGraph solves a problem that plain LangChain chains do not handle well: workflows where multiple agents need to share state, pass results to one another, or branch based on intermediate results.

### Core concepts

**StateGraph**
The graph is defined as a `StateGraph` typed by the shared state class. Every node reads from and writes to this shared state — it is the single source of truth for the entire pipeline.

**Nodes**
Each node is a Python function with the signature `(state: CVState) -> CVState`. It reads what it needs from the state, does its work (usually an LLM call), and returns a dict of updated state fields. LangGraph merges that dict back into the full state.

**Edges**
Edges connect nodes. In this project the graph is linear (sequential), so each node has exactly one outgoing edge to the next:
```
cv_parser → jd_analyzer → gap_analyzer → cv_enhancer → ats_scorer → END
```
LangGraph also supports conditional edges (branching based on state values) and parallel fan-out — features available for future extensions of this project.

**How the graph is built** (`graph/graph.py`):
```python
builder = StateGraph(CVState)
builder.add_node("cv_parser", cv_parser_node)
builder.add_edge("cv_parser", "jd_analyzer")
# … and so on
graph = builder.compile()
```
Calling `graph.invoke({"cv_raw": …, "jd_raw": …})` runs the full pipeline and returns the final state.

---

## 6. Shared State (`CVState`)

`graph/state.py` defines a `TypedDict` called `CVState`. All five agents read from and write to this object.

```python
class CVState(TypedDict, total=False):
    cv_raw: str             # raw CV text (input)
    jd_raw: str             # raw job description text (input)
    cv_parsed: dict         # structured CV data (from cv_parser)
    jd_parsed: dict         # structured JD data (from jd_analyzer)
    gaps: list[str]         # identified gaps (from gap_analyzer)
    enhanced_bullets: list[str]   # rewritten bullets (from cv_enhancer)
    ats_score_before: int   # original match score (from ats_scorer)
    ats_score_after: int    # post-enhancement score (from ats_scorer)
    final_report: str       # markdown report (from ats_scorer)
```

Using a typed state has two benefits: it makes the data flow across agents explicit and self-documenting, and it lets LangGraph validate that nodes are returning the right fields.

---

## 7. ATS (Applicant Tracking System)

### What it is
An ATS is software used by employers to filter job applications before a human reads them. It ranks CVs based on keyword matches against the job description. A CV that does not contain the right keywords — even if the candidate is qualified — may never be seen by a recruiter.

### How this project addresses it
The **JD Analyzer** extracts the keywords that an ATS would look for. The **Gap Analyzer** identifies which of those keywords are absent from the CV. The **CV Enhancer** rewrites bullet points to incorporate them naturally. The **ATS Scorer** quantifies the improvement with a 0–100 match score, computed before and after enhancement.

---

## 8. JSON as the Inter-Agent Protocol

LLMs output text, but agents need to exchange structured data. This project's solution is to instruct every agent to return **only valid JSON**, then parse that JSON back into Python dicts and lists before passing it to the next node.

The shared utility `parse_json_response` in `tools/llm_utils.py` handles:
1. Extracting plain text from the LLM response (Gemini returns content as a list of parts, not a plain string)
2. Stripping markdown code fences (` ```json … ``` `) that models sometimes add despite being told not to
3. Calling `json.loads()` and raising a descriptive `ValueError` if the model output is not valid JSON

This pattern — structured output via prompted JSON — is one of the most common techniques in production LLM applications.

---

## 9. Temperature

Temperature is a parameter that controls how random or deterministic an LLM's output is.

- **Temperature = 0** — the model always picks the most probable next token. Outputs are deterministic and consistent. Used for extraction tasks (CV Parser, JD Analyzer, Gap Analyzer, ATS Scorer) where we want reliable, repeatable JSON.
- **Temperature = 0.3** — introduces a small amount of randomness. Used for the CV Enhancer, where slightly more creative and varied language produces better-quality rewrites.

---

## 10. Provider Abstraction

The project is written against LangChain's interfaces, not Google's API directly. This means switching to a different LLM provider (OpenAI, Anthropic, Mistral, etc.) requires only two changes:

1. Swap `langchain-google-genai` for the relevant LangChain integration package
2. Replace `ChatGoogleGenerativeAI` with the corresponding class in each agent

The prompts, graph, state, and all other logic remain untouched. This is one of LangChain's core value propositions.
