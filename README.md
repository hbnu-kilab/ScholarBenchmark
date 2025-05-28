# ScholarBench: A Bilingual Benchmark for Abstraction, Comprehension, and Reasoning Evaluation in Academic Contexts
해당 repository는 ScholarBench를 이용한 모델 답변 생성과 평가 코드를 포함하고 있습니다.

## Overview and Installation

### setup

    conda create -n sb_env python=??
    conda activate sb_env
    pip install requirements.txt <-- 작성해야함


### Overview
- data : 1은 LLM에 문제만 주어서 생성한 답변, 2번은 LLM에 문제와 토픽을 주어 생성한 답변, 3번은 LLM에 문제와 paragraph를 주어서 생성한 답변, 4번은 LLM에 문제와 paragraph CoT결과를 주어서 생성한 답변, 5번은 문제와 category를 주어서 생성한 답변
    - API (1~5) 
        - GPT-4o 
        - o1-mini
        - o3-mini
    - Memory (1~4)
        - Bllossom-8b
        - Bllossom-70b
        - Exaone-8b
        - Exaone-32b
        - Exaone-32b-reasoning
        - Gemma2-9b
        - Gemma2-27b
        - Koni-8b
        - llama-8b
        - llama-70b
        - Mistral-8b
        - Mistral-24b
        - Qwen-7b
        - Qwen-32b-reasoning
        - Qwen-72b
        - Trilion-7b
- eval_scripts
    - eval_a : multiple_select, multiple_choice, true_false의 정확도를 평가하기 위한 스크립트
        - config.py
        - data_loader.py
        - evaluation_utils.py
        - metrics_calculator.py
        - main.py
        - accuracy_eval.py
    - eval_g : summarization, short_answer의 생성 품질을 평가하기 위한 스크립트
        - config.py
        - data_loader.py
        - evaluation_utils.py
        - metrics_calculator.py
        - main.py
        - quality_eval.py


## 모델답변 생성 : generate_model_answer


## 평가 : eval_scripts

| 문제유형   |  Evaluation Metrics      |
|-------------------|-----------------------------|
| Summarization   | rouge-1, rouge-2, rouge-l, bert_score  |
| Short_answer   | exact_match, f1, bert_score, bluert_score, rouge-1, bleu-1    |
| Multiple_choice         | accuracy |
| Multiple_select      |   accuracy  |
| True_false      | accuracy  |

### Run evaluation
모델 답변이 준비됐다면, 정답 파일과 원본 답변이 있는 디렉토리 결과를 저장할 디렉토리를 지정하여 다음과 같이 실행

    # Multiple_choice, Multiple_select, True_false
    python scripts/eval_a/accuracy_eval.py \
        batch \
        --ground-truth [ground-truth-path] \
        --results-dir [model-answer-path] \
        --output [output-path]
    
    # Summarization, Short_answer
    python scripts/eval_g/quality_eval.py \
        --language [ko or en] \
        batch \
        --ground-truth [ground-truth-path] \
        --results-dir [model-answer-path] \
        --output [output-path]
