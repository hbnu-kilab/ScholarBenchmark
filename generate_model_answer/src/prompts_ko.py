# prompts.py
# 프롬프트 템플릿과 few-shot 예제를 정의

inst_dict_1 = {
    'multiple_choice': '''
단일 정답을 가진 객관식 문제가 주어집니다.
Question은 제공된 질문 내용을 포함합니다.
Choices는 질문에 대한 네 개의 답변 옵션을 포함하며, 가장 적합한 답변 하나를 선택해야 합니다.
few_shot1, few_shot2은 Question에 대한 Choices 중 가장 적합한 답변 하나를 선택하는 예시입니다.
질문은 다음 형식으로 제공됩니다:

Question: {question}

Choices:
{choices}

few_shot1:
{few_shot1}

few_shot2:
{few_shot2}

few_shot을 참고하여 Question을 읽고 Choices에서 정답에 해당하는 옵션의 글자를 출력하세요.
추가 설명, 이유 또는 상세한 내용을 제공하지 마세요.
오직 정답에 해당하는 글자만 출력하세요.
''',

    'multiple_select': '''
하나 또는 여러 개의 정답이 있는 객관식 문제가 주어집니다.
Question은 제공된 질문 내용을 포함합니다.
Choices는 질문에 대한 네 개의 답변 옵션을 포함하며, 가장 적합한 하나 또는 여러 개의 답변을 선택해야 합니다.
few_shot1, few_shot2은 Question에 대한 Choices 중 가장 적합한 하나 또는 여러 개의 답변을 선택하는 예시입니다.
질문은 다음 형식으로 제공됩니다:

Question: {question}

Choices:
{choices}

few_shot1:
{few_shot1}

few_shot2:
{few_shot2}

few_shot을 참고하여 Question을 읽고 Choices에서 하나 또는 여러 개의 정답에 해당하는 옵션의 글자를 Python 리스트 형식으로 출력하세요.
추가 설명, 이유 또는 상세한 내용을 제공하지 마세요.
오직 정답에 해당하는 글자 리스트만 출력하세요.
''',

    'short_answer': '''
단답형 문제가 주어집니다.
Question은 제공된 질문 내용을 포함합니다.
few_shot1, few_shot2은 Question에 대한 단답형 답변을 하는 예시입니다.
질문은 다음 형식으로 제공됩니다:

Question: {question}

few_shot1:
{few_shot1}

few_shot2:
{few_shot2}

few_shot을 참고하여 Question을 읽고 단답형으로 답변하세요.
키워드나 짧은 구절로만 답변을 제공하세요.
완전한 문장을 사용하지 말고, 추가적인 세부 사항이나 설명을 피하세요.
오직 정답만 출력하세요.
''',

    'true_false': '''
참(True) 또는 거짓(False) 문제로, 정답은 0 또는 1입니다.
Question은 제공된 질문 내용을 포함합니다.
아래의 few_shot1, few_shot2은 Question에 대해 참인지 거짓인지 판단하는 예시입니다.
질문은 다음 형식으로 제공됩니다:

Question: {question}

few_shot1:
{few_shot1}

few_shot2:
{few_shot2}

few_shot을 참고하여 Question을 읽고 참인지 거짓인지 판단하세요.
참이면 1을 출력하고, 거짓이면 0을 출력하세요.
추가 설명, 이유 또는 상세한 내용을 제공하지 마세요.
오직 정답에 해당하는 숫자만 출력하세요.
''',

    'summarization': '''
Paragraph이 주어집니다.
Paragraph은 요약해야 할 텍스트입니다.
few_shot1, few_shot2은 제공된 paragraph을 읽고 간단하고 명확한 요약을 작성하는 예시입니다.
아래 Paragraph을 읽고 간단하고 명확한 요약을 작성하세요.
오직 요약만 출력하세요.

Paragraph: {paragraph}

few_shot1:
{few_shot1}

few_shot2:
{few_shot2}
'''
}

inst_dict_2 = {
    'multiple_choice': '''단일 정답을 가진 객관식 문제가 주어집니다.
Question은 제공된 질문 내용을 포함합니다.
Choices는 Question에 대한 네 개의 답변 옵션을 포함하며, 가장 적합한 답변 하나를 선택해야 합니다.
Topic은 Question에 대한 주제입니다.
few_shot1, few_shot2은 Question에 대한 Choices 중 가장 적합한 답변 하나를 선택하는 예시입니다.
질문은 다음 형식으로 제공됩니다:

Question: {question}

Choices:
{choices}

Topic: {topic}

few_shot1:
{few_shot1}

few_shot2:
{few_shot2}

few_shot과 topic을 참고하여 Question을 읽고 Choices에서 정답에 해당하는 옵션의 글자를 출력하세요.
추가 설명, 이유 또는 상세한 내용을 제공하지 마세요.
오직 정답에 해당하는 글자만 출력하세요.''',

    'multiple_select': '''하나 또는 여러 개의 정답이 있는 문제가 주어집니다.
Question은 제공된 질문 내용을 포함합니다.
Choices는 Question에 대한 네 개의 답변 옵션을 포함하며, 가장 적합한 하나 또는 여러 개의 답변을 선택해야 합니다.
Topic은 Question에 대한 주제입니다.
few_shot1, few_shot2은 Question에 대한 Choices 중 가장 적합한 하나 또는 여러 개의 답변을 선택하는 예시입니다.
질문은 다음 형식으로 제공됩니다:

Question: {question}

Choices:
{choices}

Topic: {topic}

few_shot1:
{few_shot1}

few_shot2:
{few_shot2}

few_shot과 topic을 참고하여 Question을 읽고 Choices에서 하나 또는 여러 개의 정답에 해당하는 옵션의 글자를 Python 리스트 형식으로 출력하세요.
추가 설명, 이유 또는 상세한 내용을 제공하지 마세요.
오직 정답에 해당하는 글자 리스트만 출력하세요.''',

    'short_answer': '''단답형 문제가 주어집니다.
Question은 제공된 질문 내용을 포함합니다.
Topic은 Question에 대한 주제입니다.
few_shot1, few_shot2은 Question에 대한 단답형 답변을 하는 예시입니다.
질문은 다음 형식으로 제공됩니다:

Question: {question}

Topic: {topic}

few_shot1:
{few_shot1}

few_shot2:
{few_shot2}

few_shot과 topic을 참고하여 Question을 읽고 단답형으로 답변하세요.
키워드나 짧은 구절로만 답변을 제공하세요.
완전한 문장을 사용하지 말고, 추가적인 세부 사항이나 설명을 피하세요.
오직 정답만 출력하세요.''',

    'true_false': '''참(True) 또는 거짓(False) 문제로, 정답은 0 또는 1입니다.
Question은 제공된 질문 내용을 포함합니다.
Topic은 Question에 대한 주제입니다.
few_shot1, few_shot2은 Question에 대해 참인지 거짓인지 판단하는 예시입니다.
질문은 다음 형식으로 제공됩니다:

Question: {question}

Topic: {topic}

few_shot1:
{few_shot1}

few_shot2:
{few_shot2}

few_shot과 topic을 참고하여 Question을 읽고 참인지 거짓인지 판단하세요.
참이면 1을 출력하고, 거짓이면 0을 출력하세요.
추가 설명, 이유 또는 상세한 내용을 제공하지 마세요.
오직 정답에 해당하는 숫자만 출력하세요.'''
}

inst_dict_3 = {
    'multiple_choice': '''
단일 정답을 가진 객관식 문제가 주어집니다.
paragraph는 Question과 관련된 내용을 포함합니다.
Question은 제공된 질문 내용을 포함합니다.
Choices는 Question에 대한 네 개의 답변 옵션을 포함하며, 가장 적합한 답변 하나를 선택해야 합니다.
few_shot1, few_shot2은 Question에 대한 Choices 중 가장 적합한 답변 하나를 선택하는 예시입니다.
질문은 다음 형식으로 제공됩니다:

paragraph: {paragraph}

Question: {question}

Choices:
{choices}

few_shot1:
{few_shot1}

few_shot2:
{few_shot2}

few_shot과 paragraph를 참고하여 Question을 읽고 Choices에서 정답에 해당하는 옵션의 글자를 출력하세요.
추가 설명, 이유 또는 상세한 내용을 제공하지 마세요.
오직 정답에 해당하는 글자만 출력하세요.''',

    'multiple_select': '''
하나 또는 여러 개의 정답이 있는 문제가 주어집니다.
paragraph는 Question과 관련된 내용을 포함합니다.
Question은 제공된 질문 내용을 포함합니다.
Choices는 Question에 대한 네 개의 답변 옵션을 포함하며, 가장 적합한 하나 또는 여러 개의 답변을 선택해야 합니다.
few_shot1, few_shot2은 Question에 대한 Choices 중 가장 적합한 하나 또는 여러 개의 답변을 선택하는 예시입니다.
질문은 다음 형식으로 제공됩니다:

paragraph: {paragraph}

Question: {question}

Choices:
{choices}

few_shot1:
{few_shot1}

few_shot2:
{few_shot2}

few_shot과 paragraph를 참고하여 Question을 읽고 Choices에서 하나 또는 여러 개의 정답에 해당하는 옵션의 글자를 Python 리스트 형식으로 출력하세요.
추가 설명, 이유 또는 상세한 내용을 제공하지 마세요.
오직 정답에 해당하는 글자 리스트만 출력하세요.''',

    'short_answer': '''
단답형 문제가 주어집니다.
paragraph는 Question과 관련된 내용을 포함합니다.
Question은 제공된 질문 내용을 포함합니다.
few_shot1, few_shot2은 Question에 대한 단답형 답변을 하는 예시입니다.
질문은 다음 형식으로 제공됩니다:

paragraph: {paragraph}

Question: {question}

few_shot1:
{few_shot1}

few_shot2:
{few_shot2}

few_shot과 paragraph를 참고하여 Question을 읽고 단답형으로 답변하세요.
키워드나 짧은 구절로만 답변을 제공하세요.
완전한 문장을 사용하지 말고, 추가적인 세부 사항이나 설명을 피하세요.
오직 정답만 출력하세요.''',

    'true_false': '''
참(True) 또는 거짓(False) 문제로, 정답은 0 또는 1입니다.
Question은 제공된 질문 내용을 포함합니다.
paragraph는 Question과 관련된 내용을 포함합니다.
아래의 few_shot1, few_shot2은 Question에 대해 참인지 거짓인지 판단하는 예시입니다.
질문은 다음 형식으로 제공됩니다:

paragraph: {paragraph}

Question: {question}

few_shot1:
{few_shot1}

few_shot2:
{few_shot2}

few_shot과 paragraph를 참고하여 Question을 읽고 참인지 거짓인지 판단하세요.
참이면 1을 출력하고, 거짓이면 0을 출력하세요.
추가 설명, 이유 또는 상세한 내용을 제공하지 마세요.
오직 정답에 해당하는 숫자만 출력하세요.'''
}

inst_dict_4 = {
    'multiple_choice': '''단일 정답을 가진 객관식 문제가 주어집니다.
paragraph는 Question과 관련된 내용을 포함합니다.
Question은 제공된 질문 내용을 포함합니다.
Choices는 Question에 대한 네 개의 답변 옵션을 포함하며, 가장 적합한 답변 하나를 선택해야 합니다.
few_shot1, few_shot2은 Question에 대한 Choices 중 가장 적합한 답변 하나를 선택하는 예시입니다.

질문은 다음 형식으로 제공됩니다:

paragraph: {paragraph}

Question: {question}

Choices:
{choices}

few_shot1:
{few_shot1}

few_shot2:
{few_shot2}

paragraph를 step by step으로 생각하고 few_shot을 참고해서 Question을 읽고 Choices에서 정답에 해당하는 옵션의 글자를 출력하세요.
추가 설명, 이유 또는 상세한 내용을 제공하지 마세요.
오직 정답에 해당하는 글자만 출력하세요.''',

    'multiple_select': '''하나 또는 여러 개의 정답이 있는 문제가 주어집니다.
paragraph는 Question과 관련된 내용을 포함합니다.
Question은 제공된 질문 내용을 포함합니다.
Choices는 Question에 대한 네 개의 답변 옵션을 포함하며, 가장 적합한 하나 또는 여러 개의 답변을 선택해야 합니다.
few_shot1, few_shot2은 Question에 대한 Choices 중 가장 적합한 하나 또는 여러 개의 답변을 선택하는 예시입니다.
질문은 다음 형식으로 제공됩니다:

paragraph: {paragraph}

Question: {question}

Choices:
{choices}

few_shot1:
{few_shot1}

few_shot2:
{few_shot2}

paragraph를 step by step으로 생각하고, few_shot을 참고하여 Question을 읽고 Choices에서 하나 또는 여러 개의 정답에 해당하는 옵션의 글자를 Python 리스트 형식으로 출력하세요.
추가 설명, 이유 또는 상세한 내용을 제공하지 마세요.
오직 정답에 해당하는 글자 리스트만 출력하세요.''',

    'short_answer': '''단답형 문제가 주어집니다.
paragraph는 Question과 관련된 내용을 포함합니다.
Question은 제공된 질문 내용을 포함합니다.
few_shot1, few_shot2은 Question에 대한 단답형 답변을 하는 예시입니다.
질문은 다음 형식으로 제공됩니다:

paragraph: {paragraph}

Question: {question}

few_shot1:
{few_shot1}

few_shot2:
{few_shot2}

paragraph를 step by step으로 생각하고 few_shot을 참고하여 Question을 읽고 단답형으로 답변하세요.
키워드나 짧은 구절로만 답변을 제공하세요.
완전한 문장을 사용하지 말고, 추가적인 세부 사항이나 설명을 피하세요.
오직 정답만 출력하세요.''',

    'true_false': '''참(True) 또는 거짓(False) 문제로, 정답은 0 또는 1입니다.
paragraph는 Question과 관련된 내용을 포함합니다.
Question은 제공된 질문 내용을 포함합니다.
few_shot1, few_shot2은 Question에 대해 참인지 거짓인지 판단하는 예시입니다.
질문은 다음 형식으로 제공됩니다:

paragraph: {paragraph}

Question: {question}

few_shot1:
{few_shot1}

few_shot2:
{few_shot2}

paragraph를 step by step으로 생각하고 few_shot 참고하여 Question을 읽고 참인지 거짓인지 판단하세요.
참이면 1을 출력하고, 거짓이면 0을 출력하세요.
추가 설명, 이유 또는 상세한 내용을 제공하지 마세요.
오직 정답에 해당하는 숫자만 출력하세요.''',

    'summarization': '''Paragraph이 주어집니다.
Paragraph은 요약해야 할 텍스트입니다.
few_shot1, few_shot2은 제공된 paragraph을 읽고 간단하고 명확한 요약을 작성하는 예시입니다.
아래 Paragraph을 읽고 step by step으로 생각한 후, few_shot을 참고하여 paragraph에 대해 간단하고 명확한 요약을 작성하세요.

Paragraph:
{paragraph}

few_shot1:
{few_shot1}

few_shot2:
{few_shot2}

오직 요약만 출력하세요.
'''
}

inst_dict_5 = {
    'multiple_choice': '''
단일 정답을 가진 객관식 문제가 주어집니다.
Question은 제공된 질문 내용을 포함합니다.
Choices는 질문에 대한 네 개의 답변 옵션을 포함하며, 가장 적합한 답변 하나를 선택해야 합니다.
Category는 Question에 대한 카테고리입니다.
few_shot1, few_shot2은 Question에 대한 Choices 중 가장 적합한 답변 하나를 선택하는 예시입니다.
질문은 다음 형식으로 제공됩니다:

Question: {question}

Choices:
{choices}

Category:
{category}

few_shot1:
{few_shot1}

few_shot2:
{few_shot2}

Category와 few_shot을 참고하여 Question을 읽고 Choices에서 정답에 해당하는 옵션의 글자를 출력하세요.
추가 설명, 이유 또는 상세한 내용을 제공하지 마세요.
오직 정답에 해당하는 글자만 출력하세요.
''',

    'multiple_select': '''
하나 또는 여러 개의 정답이 있는 객관식 문제가 주어집니다.
Question은 제공된 질문 내용을 포함합니다.
Choices는 질문에 대한 네 개의 답변 옵션을 포함하며, 가장 적합한 하나 또는 여러 개의 답변을 선택해야 합니다.
Category는 Question에 대한 카테고리입니다.
few_shot1, few_shot2은 Question에 대한 Choices 중 가장 적합한 하나 또는 여러 개의 답변을 선택하는 예시입니다.
질문은 다음 형식으로 제공됩니다:

Question: {question}

Choices:
{choices}

Category:
{category}

few_shot1:
{few_shot1}

few_shot2:
{few_shot2}

Category와 few_shot을 참고하여 Question을 읽고 Choices에서 하나 또는 여러 개의 정답에 해당하는 옵션의 글자를 Python 리스트 형식으로 출력하세요.
추가 설명, 이유 또는 상세한 내용을 제공하지 마세요.
오직 정답에 해당하는 글자 리스트만 출력하세요.
''',

    'short_answer': '''
단답형 문제가 주어집니다.
Question은 제공된 질문 내용을 포함합니다.
Category는 Question에 대한 카테고리입니다.
few_shot1, few_shot2은 Question에 대한 단답형 답변을 하는 예시입니다.
질문은 다음 형식으로 제공됩니다:

Question: {question}

Category:
{category}

few_shot1:
{few_shot1}

few_shot2:
{few_shot2}

Category와 few_shot을 참고하여 Question을 읽고 단답형으로 답변하세요.
키워드나 짧은 구절로만 답변을 제공하세요.
완전한 문장을 사용하지 말고, 추가적인 세부 사항이나 설명을 피하세요.
오직 정답만 출력하세요.
''',

    'true_false': '''
참(True) 또는 거짓(False) 문제로, 정답은 0 또는 1입니다.
Question은 제공된 질문 내용을 포함합니다.
Category는 Question에 대한 카테고리입니다.
아래의 few_shot1, few_shot2은 Question에 대해 참인지 거짓인지 판단하는 예시입니다.
질문은 다음 형식으로 제공됩니다:

Question: {question}

Category:
{category}

few_shot1:
{few_shot1}

few_shot2:
{few_shot2}

Category와 few_shot을 참고하여 Question을 읽고 참인지 거짓인지 판단하세요.
참이면 1을 출력하고, 거짓이면 0을 출력하세요.
추가 설명, 이유 또는 상세한 내용을 제공하지 마세요.
오직 정답에 해당하는 숫자만 출력하세요.
''',

    'summarization': '''
Paragraph이 주어집니다.
Paragraph은 요약해야 할 텍스트입니다.
Category는 Paragraph에 대한 카테고리입니다.
few_shot1, few_shot2은 제공된 paragraph을 읽고 간단하고 명확한 요약을 작성하는 예시입니다.
아래 Category와 few_shot을 참고하여 Paragraph을 읽고 간단하고 명확한 요약을 작성하세요.
오직 요약만 출력하세요.

Paragraph: {paragraph}

Category:
{category}

few_shot1:
{few_shot1}

few_shot2:
{few_shot2}
'''
}

few_shot_mc1 = '''
"multiple_choice": {
            "question": "지체장애인의 우울 수준이 높아질 가능성이 가장 낮은 집단은?",
            "choices": [
                "a) 고령의 여성",
                "b) 경제적 지위가 낮은 남성",
                "c) 자존감이 높은 청년",
                "d) 중년의 저소득층"
            ],
            "answer": "c"
        }
'''

few_shot_ms1 = '''
"multiple_select": {
            "question": "지체장애인의 우울 감소를 위한 정책 도입 시 고려해야 할 요인은? (모두 선택)",
            "choices": [
                "a) 성별에 따른 맞춤형 지원 프로그램 개발",
                "b) 연령대별 특화된 심리 상담 서비스 제공",
                "c) 경제적 지원 및 일자리 창출 정책 강화",
                "d) 자존감 향상을 위한 사회 참여 기회 확대"
            ],
            "answer": [
                "a",
                "b",
                "c",
                "d"
            ]
        }
'''

few_shot_sa1 = '''
"short_answer": {
            "question": "우울의 예측 요인으로 나타난 심리적 특성은 무엇인가?",
            "answer": "자존감"
        }
'''

few_shot_tf1 = '''
"true_false": {
            "question": "자존감이 높을수록 높은 우울 상승 후 하강 집단에 속할 확률이 높다. (참/거짓)",
            "answer": "거짓"
        }
'''

few_shot_sum1 = '''
"summarization": {
        "summary_text": "비우울 유지 집단을 기저집단으로 하였을 때, 높은 우울 상승 후 하강 집단 에 소속될 확률은 여성일수록, 나이가 많을수록, 국민기초생활보장제도 수급자일수록 높은 반면에 자존감은 높을수록 높은 우울 상승 후 하강 집단에 소속될 확률이 낮아졌다."
    }
'''

few_shot_mc2 = '''
"multiple_choice": {
            "question": "PBA의 공공격자결함 억제 방법 중 틀린 것은?",
            "choices": [
                "a) 반응속도 제어",
                "b) 산화제와의 복합체 형성",
                "c) 진공 열처리",
                "d) 합성 후 고온 열처리"
            ],
            "answer": "d"
        }
'''

few_shot_ms2 = '''
"multiple_select": {
            "question": "PBA의 전기화학적 성능을 향상시킬 수 있는 방법은 무엇인가? (모두 선택)",
            "choices": [
                "a) 반응속도 제어",
                "b) 공침반응",
                "c) 비활성 전이금속 도핑",
                "d) 결정수 생성"
            ],
            "answer": [
                "a",
                "c"
            ]
        }
'''

few_shot_sa2 = '''
"short_answer": {
            "question": "PBA의 전기화학성능에 부정적인 영향을 미치는 것들은 무엇인가?",
            "answer": "결정수"
        }
'''

few_shot_tf2 = '''
"true_false": {
            "question": "PBA는 수용액 상에서 합성되기에 친환경적으로 생산된다. (참/거짓)",
            "answer": "참"
        }
'''

few_shot_sum2 = '''
"summarization": {
            "summary_text": "PBA는 상온에서 수용액 상 단일 반응으로 경제적인 합성이 가능하다. 보다 높은 전기화학적 성능을 가진 PBA를 얻기 위해서는 결정 내에 발생한 공공격자결함과 결정수가 어떠한 메커니즘을 통해 PBA의 전기화학반응에 영향을 미치는지에 대한 깊은 연구가 필요하다."
        }
'''