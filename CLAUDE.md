# Project Context — Read This First Every Session

## Who I Am
Rayhane Nouri — final-year Electrical Engineering student,
National Higher Engineering School of Tunis (ENSIT), Tunisia.
Graduating before December 2026.
Seeking internship in AI/ML/DL in aviation in Germany or France.
Target companies: Lufthansa Technik, Airbus, Safran, 
MTU Aero Engines, aviation AI startups Berlin/Paris.

## My Aviation Background
- Aviation maintenance internship at Tunisair Technics
  Airbus A320/A330 cockpit systems, direct hangar access
  Saw CFM56 engines during overhaul procedures
  Applied EASA Part-145 maintenance documentation
- Embedded systems internship at Pixii Motors/ACTIA
  GPS-IMU sensor fusion, Linux, C/C++, Jetson Nano
- Shell Eco-Marathon: Led team to Phase 2 and 3 validation
  Official invitation to race in Poland
- Tunisian National Debate Champion, Top 4 Africa/Middle East

## My Other Project (Context)
Already built: Turbofan Engine Health Monitor
GitHub: github.com/rayhanenouri/cmaps-predictive-maintenance
Live: cmaps-predictive-maintenance.streamlit.app
RMSE: 18.90 cycles, R2: 0.78 on NASA C-MAPS FD001
That project predicts WHEN an engine needs maintenance.
This RAG project helps engineers find HOW to maintain it.
Together they cover the full MRO intelligence stack.

## This Project Purpose
Aviation Maintenance Document Intelligence — RAG System.
Solves a real problem: MRO engineers waste hours searching
thousands of pages of maintenance documentation manually.
This system makes that knowledge instantly retrievable
using semantic search and LLM generation.
Real companies solving this: Lufthansa Technik AVIATAR,
Airbus Skywise, Air France Industries KLM Engineering.

## Technical Stack
- Python 3.12
- LangChain for RAG orchestration
- ChromaDB for vector storage
- sentence-transformers all-MiniLM-L6-v2 for embeddings
- Groq API with openai/gpt-oss-120b for LLM generation
  Fast cloud inference, sub-second response time
  API key required via GROQ_API_KEY environment variable
- Streamlit for dashboard
- PyPDF2 for document loading
- Plotly for radar chart visualizations
- Virtual environment: .venv

## Performance Metrics
- Response time: ~1 second per query
- Evaluation: RAGAS metrics (faithfulness, answer relevancy, context precision)

## GitHub Rules — CRITICAL
- I am the only contributor: rayhanenouri
- Never commit directly — always tell me the git commands
- Never use git config to change author identity
- All commits must come from my terminal under my identity
- The repo must show only rayhanenouri as contributor

## Code Quality Rules — NON NEGOTIABLE
Every single line of code must look like it was written
by a junior MRO software engineer who deeply understands
aviation and knows what they are doing.

NEVER use:
- Separator lines: # === or # --- or # ***
- Step headers: # STEP 1, # STEP 2
- Emojis anywhere in code, comments, or prints
- Em dash in comments
- Verbose over-structured docstrings
- Boilerplate AI-generated language
- Generic comments that repeat what code does

ALWAYS use:
- Short lowercase comments explaining WHY not WHAT
- Aviation domain context in comments where relevant
- Maximum 2-line docstrings
- Clean readable code with meaningful variable names
- Print only essential output, no decorative formatting

## My Role in This Project
I am the engineer who makes decisions.
Claude Code implements what I decide.
Before coding anything — explain the decision to me,
ask for my validation, then implement.
I must understand every line well enough to defend it
in a technical interview at Lufthansa Technik or Safran.

## Dashboard Design
Match exactly my C-MAPS dashboard:
Background: #050810
Surface cards: #0c1220  
Accent: #1e90ff
Text: #e8eef4
No emojis. No rounded decorative elements.
Industrial dark theme. No Streamlit default styling.
Looks like real MRO enterprise software.

## Interview Readiness
Every technical decision must be explainable in an interview.
When Claude Code implements something non-obvious,
it must also provide the interview answer:
"Why did you choose X over Y?"
