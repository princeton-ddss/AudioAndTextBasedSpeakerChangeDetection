# Set up prompt
# Five typical examples are enough for few shot learning
# ~ 4000 examples are needed for the fine-tuning

systemprompt = """You are the expert of speaker detection. 
You are very knowledgable with how people talk to each other.
You are very proficient in identifying the conversation flow to determine if the speaker changes from the previous segment to next segment.
"""

# Formulate main question in a concise way
main_question_bgn = """
Question:
Could you identify if the speaker changes from previous segment to current segment in each part of the following conversation? 
Please put the answer in the speaker changes section of the json format below.
Please do not include any additional information besides the answer in the same json format provided by previous answer.
Please double check to ensure that the answer is in the correct json format which is the same as previous answer.
"""

examples = """
Sample Question:
Could you identify if the speaker changes from previous segment to current segment in each part of the following conversation? 
Please put the answer in the speaker changes section of the json format below.

{"conversation":[
{"segment id": "0", "previous segment": "None", "current segment": "Let's go, buddy. School time.", "speaker changes": ""},
{"segment id": "1", "previous segment": "Let's go, buddy. School time.", "current segment": "Gloria, if you want to get together with the girls later,", "speaker changes": ""},
{"segment id": "2", "previous segment": "Gloria, if you want to get together with the girls later,", "current segment": "I could just watch the football game or something.", "speaker changes": ""},
{"segment id": "3", "previous segment": "I could just watch the football game or something.", "current segment": "That means he wants to watch the football game.", "speaker changes": ""},
{"segment id": "4", "previous segment": "That means he wants to watch the football game.", "current segment": "I'm not talking to you", "speaker changes": ""},
{"segment id": "5", "previous segment": "I'm not talking to you", "current segment": "And what are you drinking coffee for, anyway?", "speaker changes": ""},
{"segment id": "6", "previous segment": "And what are you drinking coffee for, anyway?", "current segment": "It's my culture. I'm Colombian.", "speaker changes": ""},
{"segment id": "7", "previous segment": "It's my culture. I'm Colombian.", "current segment": "Oh yeah? What part of Colombia are those French toaster sticks from?", "speaker changes": ""}
]
}

Sample Answer:
{"conversation":[
{"segment id": "0", "previous segment": "None", "current segment": "Let's go, buddy. School time.", "speaker changes":"true"},
{"segment id": "1", "previous segment": "Let's go, buddy. School time.", "current segment": "Gloria, if you want to get together with the girls later,", "speaker changes": "false"},
{"segment id": "2", "previous segment": "Gloria, if you want to get together with the girls later,", "current segment": "I could just watch the football game or something.", "speaker changes": "false"},
{"segment id": "3", "previous segment": "I could just watch the football game or something.", "current segment": "That means he wants to watch the football game.", "speaker changes": "true"},
{"segment id": "4", "previous segment": "That means he wants to watch the football game.", "current segment": "I'm not talking to you", "speaker changes": "true"},
{"segment id": "5", "previous segment": "I'm not talking to you", "current segment": "And what are you drinking coffee for, anyway?", "speaker changes": "false"},
{"segment id": "6", "previous segment": "And what are you drinking coffee for, anyway?", "current segment": "It's my culture. I'm Colombian.", "speaker changes": "true"},
{"segment id": "7", "previous segment": "It's my culture. I'm Colombian.", "current segment": "Oh yeah? What part of Colombia are those French toaster sticks from?", "speaker changes": "true"}
]
}
"""


