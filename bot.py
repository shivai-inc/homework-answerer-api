from transformers import pipeline

question_answerer = pipeline("question-answering", model='distilbert-base-cased-distilled-squad')

question = "what is MLM?"

context= """So, after some research I decided to use the the roBERTa model for the job.
It is a a pre trained model made using MLM( Masked Language Modeling ) on the english language.
Which basically means that it is and AI that can find keywords in a large amount of text and then recognise it based on the question that is given.
"""

result = question_answerer(question = question, context=context)

print(f"Answer: '{result['answer']}', score: {round(result['score'], 4)}, start: {result['start']}, end: {result['end']}")
