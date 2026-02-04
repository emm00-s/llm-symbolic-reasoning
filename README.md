# LLM Symbolic Reasoning Evaluation

Measuring symbolic reasoning capacity in GPT-4o-mini using Z3 theorem prover as ground truth.

## Overview

This project evaluates whether Large Language Models possess genuine symbolic reasoning capabilities by testing GPT-4o-mini on 9 logic puzzles with formal verification via Z3.

**Key Findings:**
- Overall accuracy: 66.7%
- Systematic failures on meta-logic (0%), non-monotonic logic (0%), cardinality constraints (0%)
- Severe miscalibration: 99-100% confidence on incorrect answers
- Role-play prompting ineffective: 100% agreement between strategies

## Requirements
```bash
pip install -r requirements.txt
```

Dependencies:
- `openai` - GPT-4o-mini API
- `z3-solver` - Formal verification
- `python-dotenv` - Environment variables
- `matplotlib` - Visualization
- `numpy` - Data processing

## Setup

1. Clone the repository:
```bash
git clone https://github.com/YOUR_USERNAME/llm-symbolic-reasoning.git
cd llm-symbolic-reasoning
```

2. Create `.env` file with your OpenAI API key:
```
OPENAI_API_KEY=sk-proj-your-key-here
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the experiment:
```bash
python main.py
```

This will:
- Test GPT-4o-mini on 9 logic puzzles
- Compare responses against Z3 ground truth
- Generate `results.csv` and `results_chart.png`
- Cost: ~$0.02, Runtime: ~15 seconds

## Project Structure
```
├── solvers.py         # Z3 verification functions
├── dataset.py         # 9 logic puzzles
├── main.py            # Experimental engine
├── requirements.txt   # Dependencies
├── results.csv        # Raw experimental data
└── results_chart.png  # Visualization
```

## Puzzles

| Category | Puzzle | Ground Truth | Model Accuracy |
|----------|--------|--------------|----------------|
| Transitive | P01, P02 | False | 100% |
| Sets | P03 | True | 100% |
| **Sets** | **P04** | **True** | **0% (Failed)** |
| **Meta-Logic** | **P05** | **True** | **0% (Failed)** |
| **Non-Monotonic** | **P06** | **Unknown** | **0% (Failed)** |
| Fallacy | P07 | Unknown | 100% |
| Propositional | P08 | True | 100% |
| Paradox | P09 | Paradox | 100% |

## Citation

If you use this work, please cite:
```
[Emmanuel Santoro]. (2026). Measuring Symbolic Reasoning in LLMs: A Formal Verification Approach.
```

