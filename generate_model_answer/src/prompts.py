# prompts.py
# 프롬프트 템플릿과 few-shot 예제를 정의

inst_dict_1 = {
    'multiple_choice': '''
A multiple-choice question with a single correct answer is provided.
The Question contains the given question text.
The Choices include four answer options for the question, and you must select the most appropriate one.
few_shot1 and few_shot2 provide examples of selecting the most appropriate answer from the Choices for the given Question.
The question is presented in the following format:

Question: {question}

Choices:
{choices}

few_shot1:
{few_shot1}

few_shot2:
{few_shot2}

Refer to the few_shot examples, read the Question, and output only the letter corresponding to the correct answer from the Choices.
Do not provide any additional explanations, reasons, or detailed content.
Only output the letter of the correct answer.
''',

    'multiple_select': '''
A multiple-choice question with one or more correct answers is provided.
The Question contains the given question text.
The Choices include four answer options for the question, and you must select the most appropriate one or more answers.
few_shot1 and few_shot2 provide examples of selecting the most appropriate one or more answers from the Choices for the given Question.
The question is presented in the following format:

Question: {question}

Choices:
{choices}

few_shot1:
{few_shot1}

few_shot2:
{few_shot2}

Refer to the few_shot examples, read the Question, and output the letter(s) corresponding to the correct answer(s) from the Choices in Python list format.
Do not provide any additional explanations, reasons, or detailed content.
Only output the list of correct answer letters.
''',

    'short_answer': '''
A short-answer question is provided.
The Question contains the given question text.
few_shot1 and few_shot2 provide examples of short-answer responses to the Question.
The question is presented in the following format:

Question: {question}

few_shot1:
{few_shot1}

few_shot2:
{few_shot2}

Refer to the few_shot examples, read the Question, and provide a short-answer response.
Answer only with keywords or short phrases.
Do not use complete sentences or provide additional details or explanations.
Only output the correct answer.
''',

    'true_false': '''
A True or False question is provided, where the correct answer is either 0 or 1.
The Question contains the given question text.
few_shot1 and few_shot2 provide examples of determining whether the Question is true or false.
The question is presented in the following format:

Question: {question}

few_shot1:
{few_shot1}

few_shot2:
{few_shot2}

Refer to the few_shot examples, read the Question, and determine whether it is true or false.
Output 1 if true and 0 if false.
Do not provide any additional explanations, reasons, or details.
Only output the corresponding number.
''',

    'summarization': '''
A paragraph is provided.
The Paragraph is the text to be summarized.
few_shot1 and few_shot2 provide examples of creating a simple and clear summary of the given paragraph.
Read the following Paragraph and provide a brief and clear summary.
Output only the summary.

Paragraph: {paragraph}

few_shot1:
{few_shot1}

few_shot2:
{few_shot2}
'''
}

inst_dict_2 = {
    'multiple_choice': '''
A multiple-choice question with a single correct answer is provided.
The Question contains the given question text.
The Choices include four answer options for the question, and you must select the most appropriate one.
The Topic provides the subject related to the Question.
few_shot1 and few_shot2 provide examples of selecting the most appropriate answer from the Choices for the given Question.

The question is presented in the following format:

Question: {question}

Choices:
{choices}

Topic: {topic}

few_shot1:
{few_shot1}

few_shot2:
{few_shot2}

Refer to the few_shot examples and the topic, read the Question, and output only the letter corresponding to the correct answer from the Choices.
Do not provide any additional explanations, reasons, or detailed content.
Only output the letter of the correct answer.
''',

    'multiple_select': '''
A question with one or more correct answers is provided.
The Question contains the given question text.
The Choices include four answer options for the question, and you must select the most appropriate one or more answers.
The Topic provides the subject related to the Question.
few_shot1 and few_shot2 provide examples of selecting the most appropriate one or more answers from the Choices for the given Question.

The question is presented in the following format:

Question: {question}

Choices:
{choices}

Topic: {topic}

few_shot1:
{few_shot1}

few_shot2:
{few_shot2}

Refer to the few_shot examples and the topic, read the Question, and output the letter(s) corresponding to the correct answer(s) from the Choices in Python list format.
Do not provide any additional explanations, reasons, or details.
Only output the list of correct answer letters.
''',

    'short_answer': '''
A short-answer question is provided.
The Question contains the given question text.
The Topic provides the subject related to the Question.
few_shot1 and few_shot2 provide examples of short-answer responses to the Question.

The question is presented in the following format:

Question: {question}

Topic: {topic}

few_shot1:
{few_shot1}

few_shot2:
{few_shot2}

Refer to the few_shot examples and the topic, read the Question, and provide a short-answer response.
Answer only with keywords or short phrases.
Do not use complete sentences or provide additional details or explanations.
Only output the correct answer.
''',

    'true_false': '''
A True or False question is provided, where the correct answer is either 0 or 1.
The Question contains the given question text.
The Topic provides the subject related to the Question.
few_shot1 and few_shot2 provide examples of determining whether the Question is true or false.

The question is presented in the following format:

Question: {question}

Topic: {topic}

few_shot1:
{few_shot1}

few_shot2:
{few_shot2}

Refer to the few_shot examples and the topic, read the Question, and determine whether it is true or false.
Output 1 if true and 0 if false.
Do not provide any additional explanations, reasons, or details.
Only output the corresponding number.
'''
}

inst_dict_3 = {
    'multiple_choice': '''
A multiple-choice question with a single correct answer is provided.
The paragraph contains content related to the Question.
The Question contains the given question text.
The Choices include four answer options for the question, and you must select the most appropriate one.
few_shot1 and few_shot2 provide examples of selecting the most appropriate answer from the Choices for the given Question.

The question is presented in the following format:

paragraph: {paragraph}

Question: {question}

Choices:
{choices}

few_shot1:
{few_shot1}

few_shot2:
{few_shot2}

Refer to the few_shot examples and the paragraph, read the Question, and output the letter corresponding to the correct answer from the Choices.
Do not provide any additional explanations, reasons, or details.
Only output the letter of the correct answer.
''',

    'multiple_select': '''
A question with one or more correct answers is provided.
The paragraph contains content related to the Question.
The Question contains the given question text.
The Choices include four answer options for the question, and you must select the most appropriate one or more answers.
few_shot1 and few_shot2 provide examples of selecting the most appropriate one or more answers from the Choices for the given Question.

The question is presented in the following format:

paragraph: {paragraph}

Question: {question}

Choices:
{choices}

few_shot1:
{few_shot1}

few_shot2:
{few_shot2}

Refer to the few_shot examples and the paragraph, read the Question, and output the letters of the correct answer(s) from the Choices in a Python list format.
Do not provide any additional explanations, reasons, or details.
Only output the list of correct answer letters.
''',

    'short_answer': '''
A short-answer question is provided.
The paragraph contains content related to the Question.
The Question contains the given question text.
few_shot1 and few_shot2 provide examples of short-answer responses for the given Question.

The question is presented in the following format:

paragraph: {paragraph}

Question: {question}

few_shot1:
{few_shot1}

few_shot2:
{few_shot2}

Refer to the few_shot examples and the paragraph, read the Question, and provide a short-answer response.
Only use keywords or short phrases for the answer.
Do not use complete sentences, provide additional details, or explanations.
Only output the correct answer.
''',

    'true_false': '''
A True/False question is provided, where the answer is either 0 or 1.
The Question contains the given question text.
The paragraph contains content related to the Question.
few_shot1 and few_shot2 provide examples of determining whether the statement in the Question is True or False.

The question is presented in the following format:

paragraph: {paragraph}

Question: {question}

few_shot1:
{few_shot1}

few_shot2:
{few_shot2}

Refer to the few_shot examples and the paragraph, read the Question, and determine if it is True or False.
Output 1 for True and 0 for False.
Do not provide any explanations, reasons, or additional details.
Only output the correct number (1 or 0).
'''
}

inst_dict_4 = {
    'multiple_choice': '''
A multiple-choice question with a single correct answer will be provided.
The paragraph contains information related to the Question.
The Question includes the provided question content.
The Choices include four answer options for the Question, and the most appropriate one must be selected.
few_shot1 and few_shot2 are examples of selecting the most appropriate answer from the Choices for the Question.

The question will be provided in the following format:

paragraph: {paragraph}
Question: {question}
Choices:
{choices}

few_shot1:
{few_shot1}

few_shot2:
{few_shot2}

Think through the paragraph step by step and refer to the few_shot to read the Question.
Then, select the correct option from the Choices and output only the corresponding letter.
Do not provide additional explanations, reasons, or details.
Only output the letter corresponding to the correct answer.
''',

    'multiple_select': '''
A question with one or more correct answers will be provided.
The paragraph contains information related to the Question.
The Question includes the provided question content.
The Choices include four answer options for the Question, and the most appropriate one or more answers must be selected.
few_shot1 and few_shot2 are examples of selecting the most appropriate one or more answers from the Choices for the Question.

The question will be provided in the following format:

paragraph: {paragraph}
Question: {question}
Choices:
{choices}

few_shot1:
{few_shot1}

few_shot2:
{few_shot2}

Think through the paragraph step by step, refer to the few_shot, and read the Question.
Then, select the correct one or more options from the Choices and output them as a Python list.
Do not provide additional explanations, reasons, or details.
Only output the list of letters corresponding to the correct answers.
''',

    'short_answer': '''
A short-answer question will be provided.
The paragraph contains information related to the Question.
The Question includes the provided question content.
few_shot1 and few_shot2 are examples of short-answer responses to the Question.

The question will be provided in the following format:

paragraph: {paragraph}
Question: {question}

few_shot1:
{few_shot1}

few_shot2:
{few_shot2}

Think through the paragraph step by step, refer to the few_shot, and read the Question.
Then, provide a short-answer response.
Only provide keywords or short phrases as answers.
Do not use full sentences or provide additional details or explanations.
Only output the correct answer.
''',

    'true_false': '''
A true/false question will be provided, where the answer is 0 or 1.
The paragraph contains information related to the Question.
The Question includes the provided question content.
few_shot1 and few_shot2 are examples that determine whether the statement is true or false.

The question will be provided in the following format:

paragraph: {paragraph}
Question: {question}

few_shot1:
{few_shot1}

few_shot2:
{few_shot2}

Think through the paragraph step by step, refer to the few_shot, and read the Question.
Then, determine if the statement is true or false.
Output 1 if true and 0 if false.
Do not provide additional explanations, reasons, or details.
Only output the correct number corresponding to the answer.
''',

    'summarization': '''
A paragraph will be given.
The paragraph is the text that needs to be summarized.
few_shot1 and few_shot2 are examples of simple and clear summaries based on the given paragraph.
Read the paragraph carefully, step by step, and refer to the few_shot examples to create a simple and clear summary of the paragraph.

Paragraph:
{paragraph}

few_shot1:
{few_shot1}

few_shot2:
{few_shot2}

Only output the summary.
'''
}

inst_dict_5 = {
    'multiple_choice': '''
A multiple-choice question with a single correct answer is provided.
The Question contains the given question text.
The Choices include four answer options for the question, and you must select the most appropriate one.
The Category provides the subject related to the Question.
few_shot1 and few_shot2 provide examples of selecting the most appropriate answer from the Choices for the given Question.

The question is presented in the following format:

Question: {question}

Choices:
{choices}

Category: {category}

few_shot1:
{few_shot1}

few_shot2:
{few_shot2}

Refer to the few_shot examples and the Category, read the Question, and output only the letter corresponding to the correct answer from the Choices.
Do not provide any additional explanations, reasons, or detailed content.
Only output the letter of the correct answer.
''',

    'multiple_select': '''
A question with one or more correct answers is provided.
The Question contains the given question text.
The Choices include four answer options for the question, and you must select the most appropriate one or more answers.
The Category provides the subject related to the Question.
few_shot1 and few_shot2 provide examples of selecting the most appropriate one or more answers from the Choices for the given Question.

The question is presented in the following format:

Question: {question}

Choices:
{choices}

Category: {category}

few_shot1:
{few_shot1}

few_shot2:
{few_shot2}

Refer to the few_shot examples and the Category, read the Question, and output the letter(s) corresponding to the correct answer(s) from the Choices in Python list format.
Do not provide any additional explanations, reasons, or details.
Only output the list of correct answer letters.
''',

    'short_answer': '''
A short-answer question is provided.
The Question contains the given question text.
The Category provides the subject related to the Question.
few_shot1 and few_shot2 provide examples of short-answer responses to the Question.

The question is presented in the following format:

Question: {question}

Category: {category}

few_shot1:
{few_shot1}

few_shot2:
{few_shot2}

Refer to the few_shot examples and the Category, read the Question, and provide a short-answer response.
Answer only with keywords or short phrases.
Do not use complete sentences or provide additional details or explanations.
Only output the correct answer.
''',

    'true_false': '''
A True or False question is provided, where the correct answer is either 0 or 1.
The Question contains the given question text.
The Category provides the subject related to the Question.
few_shot1 and few_shot2 provide examples of determining whether the Question is true or false.

The question is presented in the following format:

Question: {question}

Category: {category}

few_shot1:
{few_shot1}

few_shot2:
{few_shot2}

Refer to the few_shot examples and the Category, read the Question, and determine whether it is true or false.
Output 1 if true and 0 if false.
Do not provide any additional explanations, reasons, or details.
Only output the corresponding number.
''',

    'summarization': '''
A paragraph is provided.
The Paragraph is the text to be summarized.
The Category represents the subject related to the paragraph.
few_shot1 and few_shot2 provide examples of creating a simple and clear summary of the given paragraph.
Read the paragraph below and refer to the Category and few-shot examples to write a concise and clear summary.
Only output the summary.

Paragraph: {paragraph}

Category: {category}

few_shot1:
{few_shot1}

few_shot2:
{few_shot2}

Only output the summary.
'''
}

few_shot_mc1 = '''
"multiple_choice": {
    "question": "Which aspect was assessed for each therapeutic target according to the TTD update?",
    "choices": [
        "a) Genetic variability",
        "b) Network characteristics",
        "c) Similarity to human proteins",
        "d) Drug toxicity"
    ],
    "answer": "c"
}
'''

few_shot_ms1 = '''
"multiple_select": {
    "question": "What factors about the TTD update were considered in evaluating the druggability of targets? (Select all that apply)",
    "choices": [
        "a) Expression patterns across tissues",
        "b) Drug approval status",
        "c) Similarity to human proteins",
        "d) Involvement in crucial biological pathways"
    ],
    "answer": [
        "a",
        "c",
        "d"
    ]
}
'''

few_shot_sa1 = '''
"short_answer": {
    "question": "What specific update was made to the number of therapeutic targets in TTD 2024?",
    "answer": "Increased targets"
}
'''

few_shot_tf1 = '''
"true_false": {
    "question": "The TTD update shows that there has been a reduction in the number of drugs over the past decade. (True/False)",
    "answer": "False"
}
'''

few_shot_sum1 = '''
"summarization": {
    "summary_text": "The TTD 2024 provides updated statistics on therapeutic targets and drugs, revealing an increase in both areas over the last decade. It features details on target-drug interactions and emphasizes the importance of understanding target similarities, their presence in essential pathways, and tissue distributions for effective drug development."
}
'''

few_shot_mc2 = '''
"multiple_choice": {
    "question": "What type of catalytic activity is probe 40 designed to target?",
    "choices": [
        "a) Furin",
        "b) Matrix metalloproteinase",
        "c) Lipase",
        "d) Amylase"
    ],
    "answer": "a"
}
'''

few_shot_ms2 = '''
"multiple_select": {
    "question": "Which characteristics can be attributed to probe 40? (Select all that apply)",
    "choices": [
        "a) Specific to furin",
        "b) Non-fluorescent",
        "c) Utilizes a self-immolative linker",
        "d) Enables real-time tracking"
    ],
    "answer": [
        "a",
        "c",
        "d"
    ]
}
'''

few_shot_sa2 = '''
"short_answer": {
    "question": "Which probe is designed for the long-term in situ detection of endogenous furin activity?",
    "answer": "Endogenous furin activity"
}
'''

few_shot_tf2 = '''
"true_false": {
    "question": "Probe 40 displays low selectivity towards furin, making it ineffective for imaging. (True/False)",
    "answer": "False"
}
'''

few_shot_sum2 = '''
"summarization": {
    "summary_text": "The Zhang group developed a Golgi-targeting probe 40 in 2018 for detecting endogenous furin activity. This probe showcases high sensitivity and selectivity due to its unique design using a furin-specific peptide, contributing to long-term bioimaging. The advancements in this mechanism are significant for studying furin-related cellular processes."
}
'''