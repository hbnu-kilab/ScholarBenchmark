model_nicks = {
    'llama-70b': 'meta-llama/Llama-3.3-70B-Instruct',
    'llama-8b': 'meta-llama/Llama-3.1-8B-Instruct',
    'Mistral-24b': 'mistralai/Mistral-Small-24B-Instruct-2501',
    'Mistral-8b': 'mistralai/Ministral-8B-Instruct-2410',
    'Qwen-72b': 'Qwen/Qwen2.5-72B-Instruct',
    'Qwen-32b-reasoning': 'Qwen/QwQ-32B',
    'Qwen-7b': 'Qwen/Qwen2.5-7B-Instruct',
    'Trilion-7b': 'trillionlabs/Trillion-7B-preview',
    'Gemma2-27b': 'google/gemma-2-27b-it',
    'Gemma2-9b': 'google/gemma-2-9b-it',
    'Bllossom-70b': 'Bllossom/llama-3-Korean-Bllossom-70B',
    'Bllossom-8b': 'MLP-KTLim/llama-3-Korean-Bllossom-8B',
    'Koni-8b': 'KISTI-KONI/KONI-Llama3.1-8B-R-Preview-20250320',
    'Exaone-32b-reasoning': 'LGAI-EXAONE/EXAONE-Deep-32B',
    'Exaone-32b': 'LGAI-EXAONE/EXAONE-3.5-32B-Instruct',
    'Exaone-8b': 'LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct'
}

template_dict = {
    'multiple_select': '''
Question: 
The content of the question to be solved

Choices: 
a) The first choice
b) The second choice
c) The third choice
d) The fourth choice
''',
    'multiple_choice': '''
Question: 
The content of the question to be solved

Choices: 
a) The first choice
b) The second choice
c) The third choice
d) The fourth choice
''',
    'true_false': '''
Question: 
The content of the question to be solved
''',
    'short_answer': '''
Question: 
The content of the question to be solved
''',
    'summarization': '''
Paragraph: 
The text to be summarized
'''
}

inst_dict = {
    'multiple_choice': '''A multiple-choice question is given with a single correct answer.
The Question includes the content of the provided question.
The Choices contain four answer options for the question, and you must select the single most appropriate answer.
The question is provided in the following format:
{}
Read the Question and output the letter corresponding to the correct answer from the Choices.
Do not provide additional explanation, reason, or details.
Only output the letter corresponding to the correct answer.''',
    
    'multiple_select': '''A question is given that may have one or multiple correct answers.
The Question includes the content of the provided question.
The Choices contain four answer options for the question, and you must select one or more of the most appropriate answers.
The question is provided in the following format:
{}
Read the Question and output the letters of one or more correct answers from the Choices in Python list format.
Do not provide additional explanation, reason, or details.
Only output the list of letters corresponding to the correct answer(s).''',
    
    'short_answer': '''A short-answer question is given.
The Question includes the content of the provided question.
The question is provided in the following format:
{}
Read the Question and provide a short answer.
Respond only with keywords or a brief phrase.
Do not use complete sentences or include additional details or explanations.
Only output the answer.''',
    
    'true_false': '''A True/False problem is given, where the correct answer is either 0 or 1.
The Question includes the content of the provided question.
The question is provided in the following format:
{}
Read the Question and decide whether it is True or False.
If it is True, output 1; if it is False, output 0.
Do not provide additional explanation, reason, or details.
Only output the number corresponding to the correct answer.''',
    
    'summarization': '''A Paragraph is given.
The Paragraph is a text to be summarized.
Read the Paragraph and write a concise and clear summary.
Only output the summary.'''
}

cot_inst_dict = {
    'multiple_choice': '''A multiple-choice question is given with a single correct answer.
The Question includes the content of the provided question.
The Choices contain four answer options for the question, and you must select the single most appropriate answer.
The question is provided in the following format:
{}
Read the Question and output the letter corresponding to the correct answer from the Choices.
At the end of the COT path, print </think> and then print your answer.
Let's think step by step.''',
    
    'multiple_select': '''A question is given that may have one or multiple correct answers.
The Question includes the content of the provided question.
The Choices contain four answer options for the question, and you must select one or more of the most appropriate answers.
The question is provided in the following format:
{}
Read the Question and output the letters of one or more correct answers from the Choices in Python list format.
At the end of the COT path, print </think> and then print your answer.
Let's think step by step.''',
    
    'short_answer': '''A short-answer question is given.
The Question includes the content of the provided question.
The question is provided in the following format:
{}
Read the Question and provide a short answer.
Respond only with keywords or a brief phrase.
At the end of the COT path, print </think> and then print your answer.
Let's think step by step.''',
    
    'true_false': '''A True/False problem is given, where the correct answer is either 0 or 1.
The Question includes the content of the provided question.
The question is provided in the following format:
{}
Read the Question and decide whether it is True or False.
If it is True, output 1; if it is False, output 0.
At the end of the COT path, print </think> and then print your answer.
Let's think step by step.''',
    
    'summarization': '''A Paragraph is given.
The Paragraph is a text to be summarized.
Read the Paragraph and write a concise and clear summary.
At the end of the COT path, print </think> and then print your answer.
Let's think step by step.'''
}

cot_path = {
    "multiple_choice_1": "The primary factors affecting the depression levels of individuals with physical disabilities are gender (the higher risk for females), age (the older, the higher the risk), economic status (the lower, the higher the risk), and self-esteem (the lower, the higher the risk). Among the answer choices, ‘a young adult with high self-esteem’ is younger in age and higher in self-esteem, making it the group least likely to experience elevated levels of depression. Therefore, the correct answer is (c) a young adult with high self-esteem.</think>\n\n",
    "multiple_choice_2": "Methods for suppressing vacant lattice defects generally include (1) controlling the reaction rate during the synthesis stage, (2) forming a complex with an oxidizer after synthesis, and (3) removing crystal water via vacuum heat treatment after synthesis. Among the options provided in the question, (a) reaction rate control, (b) forming a complex with an oxidizer, and (c) vacuum heat treatment are all actually used. However, (d) high-temperature heat treatment after synthesis is not mentioned in the text, and applying high heat to a PBA structure typically poses a risk of structural damage, so it is not an appropriate method for suppressing vacant lattice defects. Therefore, the incorrect answer is (d) high-temperature heat treatment after synthesis.</think>\n\n",
    "multiple_select_1": "The major factors influencing the depression of individuals with physical disabilities are gender, age, economic status, and self-esteem. Therefore, when introducing policies, it is necessary to consider (a) developing customized support programs based on gender, (b) providing age-specific psychological counseling services, (c) strengthening economic support and job creation policies, and (d) expanding opportunities for social participation to enhance self-esteem.</think>\n\n",
    "multiple_select_2": "To enhance the electrochemical performance of PBA, it is effective to reduce vacant lattice defects by controlling the reaction rate and to improve structural stability through the doping of inactive transition metals. Co-precipitation (b) is a common synthesis method, whereas the formation of crystal water (d) actually contributes to performance deterioration. Therefore, the methods that help improve performance are (a) controlling the reaction rate and (c) doping with inactive transition metals.</think>\n\n",
    "short_answer_1": "Among the predictive factors for depression, self-esteem emerged as a psychological characteristic. According to the text, higher self-esteem generally correlates with lower levels of depression.</think>\n\n",
    "short_answer_2": "Factors that negatively affect the electrochemical performance of PBA include crystal water and vacant lattice defects (vacancies). These are easily created during the rapid crystal formation process, ultimately leading to diminished electrochemical performance.</think>\n\n",
    "true_false_1": "As self-esteem increases, the likelihood of belonging to the high-depression-then-decline group “decreases.” According to the study results in the text, higher self-esteem lowers the probability of being in both the high-depression-then-decline group and the moderate-depression-maintenance group. Therefore, the statement (“the higher the self-esteem, the more likely one is to be in the high-depression-then-decline group”) is false.</think>\n\n",
    "true_false_2": "According to the text, PBA is synthesized through a co-precipitation reaction in an aqueous solution at room temperature, which is considered an economical and eco-friendly production method. Therefore, this statement is “true.”</think>\n\n",
    "summarization_1": "1. The passage introduces TTD 2024, highlighting how the database incorporates diverse types of data reflecting druggability of therapeutic targets.\n2. It emphasizes expression profiles of targets across various human tissues and organs, noting that protein-level and RNA-level expression can differ.\n3. The text details how expression patterns are influenced by both exogenous (e.g., drug treatment, infection, environment) and endogenous (e.g., mutations, over-expression) factors, collected through GEO and Expression Atlas.\n4. The passage also describes ongoing updates to TTD, including newly approved drugs, drugs in clinical trials, and preclinical or patented drugs.\n5. Finally, it explains the new and improved search and download functionalities for users and discusses why ensuring data accuracy (e.g., avoiding false positives) remains a priority.\n\nSummary (English):\nTTD 2024 integrates comprehensive data sets on therapeutic targets and their ‘druggability’ features. This includes target expression profiles across human tissues, varied cell lines from numerous disease types, and differential expression changes triggered by exogenous and endogenous factors. Updated regularly with newly approved drugs, clinical trials, and preclinical/patented agents, TTD 2024 also refines its search capabilities for broader and more user-friendly access. Overall, these enhancements aim to improve the systematic evaluation of target druggability and support more effective drug discovery and development.",
    "summarization_2": "1. Figure 35(A-E) highlights probe 37’s utility in monitoring polarity and demonstrates co-localization with Golgi Tracker Red in live cells. It also shows that glutamate-induced polarity fluctuations can be visualized in PC12 cells, and the probe allows imaging in mouse brains exhibiting depression-like behavior. Furthermore, siRNA knockdown experiments targeting BDNF genes illustrate the probe’s capacity for tracking changes in fluorescence signals.\n2. The Zhang group introduced probe 40, integrating a furin-recognition peptide (RVRR) with an insoluble solid-state fluorophore via a self-immolative linker. This design ensures that when furin cleaves the peptide, the fluorophore is released and precipitates near the enzyme’s active site, allowing for prolonged, localized imaging of intracellular furin.\n3. Figure 36(A-D) describes a new probe (38) for detecting Zn2+. The probe displays distinct fluorescence changes corresponding to Zn2+ concentration, enabling researchers to observe Zn2+ fluctuations in living cells, including conditions of oxidative stress.\n\nSummary (English):\nThese sections detail the development of specialized fluorescent probes for real-time, in situ imaging of various cellular processes. Probe 37 visualizes polarity changes, co-localizes with the Golgi apparatus, and tracks neurochemical perturbations in both cell culture and mouse brain models. Probe 40, engineered by the Zhang group, selectively monitors intracellular furin by releasing a bright, solid-state fluorophore near the enzyme’s active site. Lastly, probe 38 provides a robust method to detect Zn2+ fluctuations in cells under different conditions, including oxidative stress. Collectively, these advances illustrate how targeted probes can reveal dynamic biological events at subcellular resolution."
  }