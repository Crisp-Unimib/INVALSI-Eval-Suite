# INVALSI Italian LLM Benchmark

A structured benchmark for evaluating Large Language Models' proficiency on Italian student competencies using INVALSI (National Institute for the Evaluation of the Education and Training System) tests.

## Dataset Overview

This benchmark adapts standardized Italian educational assessments to evaluate LLMs' linguistic capabilities in Italian, providing a comprehensive evaluation framework that spans multiple educational levels and competency areas.

- **Curated by:** CRISP research centre https://crispresearch.it/
- **Language(s) (NLP):** Italian
- **License:** MIT

## How to Use

First, clone the repository to your local machine:

```bash
git clone https://github.com/Crisp-Unimib/INVALSI-Eval-Suite.git
cd INVALSI-Eval-Suite
```

To avoid conflicts with system packages, it's recommended to use a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

Our benchmark requires vLLM for serving models. Install it with:

```bash
pip install vllm[all]
```

Run the vLLM server with your preferred model. Here’s an example using meta-llama/Llama-3.3-70B-Instruct:

```bash
vllm serve meta-llama/Llama-3.3-70B-Instruct
```

Once the vLLM server is running, you can tweak setting in config.yaml, such as model name, API endpoint, temperature, max tokens.
You can then execute the evaluation script:

```bash
python Invalsi_run_eval.py
```

### Dataset Composition

- **Tests**: 11 carefully selected tests from different academic years
- **Questions**: 405 questions across 6 educational grades (2nd, 5th, 6th, 8th, 10th, 13th)
- **Items**: 618 individual assessment items
- **Coverage**: Primary school through high school levels

### Question Formats

The benchmark includes three distinct question types:

**Multiple Choice (MC)** - 337 questions (83.2%): Standard questions with four answer options where only one is correct.

**Multiple Complex Choice (MCC)** - 35 questions (8.6%): Questions with multiple binary items requiring affirm/deny responses for each component.

**Unique Response (RU)** - 33 questions (8.1%): Open-ended questions requiring specific answers without provided options.

## Competency Areas

### Text Comprehension (78.5% of questions)

- **Reconstruct meaning** (43.7%): Understanding text context and encyclopedic knowledge
- **Locate information** (26.7%): Identifying specific information within provided text
- **Reflect and evaluate** (8.1%): Interpreting texts and expressing evaluative judgments

### Reflection on Language (21.5% of questions)

- **Lexicon and semantics** (7.2%): Semantic relationships between words
- **Morphology** (5.9%): Lexical categories and grammatical structures
- **Syntax** (4.4%): Italian written language syntactic rules
- **Word formation** (1.7%): Base words and derivatives
- **Textuality and pragmatics** (1.2%): Text organization and cohesion
- **Spelling** (1.0%): Proper use of accents, apostrophes, and capitalization

## Motivation

Current LLM evaluation frameworks predominantly focus on English, creating a significant gap in assessing model performance for other languages. Italian presents unique linguistic challenges with its rich morphology, complex syntax, and cultural nuances. This benchmark addresses the need for rigorous Italian language assessment by leveraging well-established educational standards that have been used across Italy since 2005.

The INVALSI tests provide an ideal foundation because they measure real-world language competencies, include progressive complexity across educational levels, and offer culturally relevant content for Italian speakers. These characteristics make the benchmark valuable not only for Italian language assessment but also for understanding general language model capabilities across different complexity levels.

## Dataset Creation Process

Questions were sourced from the Gestinv database, covering national assessments from 2008 onwards. Manual curation was performed to adapt questions for automated evaluation while preserving their educational integrity. Graphical elements were converted to multiple-choice formats, and questions requiring visual analysis were excluded to ensure compatibility with text-based LLM evaluation.

The selection process prioritized diversity across educational grades and question types while excluding items that would be difficult to reformulate for language model comprehension. All questions maintain their original pedagogical intent and difficulty level.

## Evaluation Methodology

Models are evaluated in a zero-shot setting with temperature 0 for deterministic outputs. The evaluation employs multiple strategies depending on question type:

**Multiple Choice questions** use regex-based answer matching to verify if generated responses contain the target answer.

**Complex questions with multiple items** require all components to be answered correctly for the question to be marked as correct.

**Open-ended questions** utilize word matching, pattern recognition with regular expressions, and BERTScore semantic similarity (threshold 0.7) to assess response correctness.

## Repository Contents

- Dataset files with questions, answers, and metadata
- Evaluation scripts and framework

## Usage and Applications

This benchmark serves multiple purposes: evaluating Italian language capabilities of existing and new LLMs, comparing model performance across different competency areas, assessing readiness for Italian language deployment in industrial applications, and providing a standardized framework for Italian NLP research.

The benchmark is particularly valuable for organizations developing Italian language services, researchers working on multilingual models, and educators interested in AI performance on pedagogical content.

## Limitations and Considerations

Some questions required subjective judgment for evaluation, addressed through empirically validated BERTScore thresholds. Manual curation was necessary due to formatting inconsistencies in source materials. A small number of questions involved missing data or unclear options, which were identified and corrected through manual review.

The benchmark focuses exclusively on linguistic capabilities and does not assess mathematical or multimodal competencies present in complete INVALSI assessments.

## Ethical Considerations

The dataset consists entirely of publicly available educational materials and contains no personal, confidential, or sensitive information. All source materials are derived from official Italian educational assessments available through public databases. No human subjects were involved in data collection, and no privacy concerns exist.

Potential misuse could involve inappropriate conclusions about Italian language model development, but the benchmark provides objective assessment metrics that should be interpreted within their educational context.

## Citation and License

This benchmark is released under the MIT license with no restrictions on academic or commercial use. The evaluation code and complete dataset are publicly available to ensure reproducibility and encourage community contributions.

**BibTeX:**

```
@InProceedings{10.1007/978-3-662-72243-5_17,
author="Mercorio, Fabio and Mezzanzanica, Mario and Potert{\`i}, Daniele and Serino, Antonio and Seveso, Andrea",
title="A Benchmark to Evaluate LLMs' Proficiency on Italian Student Competencies",
booktitle="Machine Learning and Knowledge Discovery in Databases. Research Track and Applied Data Science Track",
year="2026",
publisher="Springer Berlin Heidelberg",
address="Berlin, Heidelberg",
pages="292--309"
}
```

**APA:**

```
Mercorio, F., Mezzanzanica, M., Potertì, D., Serino, A., Seveso, A. (2026). A Benchmark to Evaluate LLMs’ Proficiency on Italian Student Competencies. In: Pfahringer, B., et al. Machine Learning and Knowledge Discovery in Databases. Research Track and Applied Data Science Track. ECML PKDD 2025. Lecture Notes in Computer Science(), vol 16020. Springer, Berlin, Heidelberg. https://doi.org/10.1007/978-3-662-72243-5_17
```

## Dataset Card Contact

Andrea Seveso - andrea.seveso@unimib.it
