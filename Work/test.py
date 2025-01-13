'''
refer to:
https://www.youtube.com/watch?v=czvVibB2lRA
https://github.com/ShawhinT/YouTube-Blog/tree/main/LLMs/openai-api
'''

import openai
import time

openai.api_key = "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"

# create a chat completion
chat_completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=[{"role": "user", "content": "Listen to your"}])
chat_completion.to_dict()

# print the chat completion
print(chat_completion.choices[0].message.content)


# # create a chat completion
# chat_completion = openai.ChatCompletion.create(
#     model="gpt-3.5-turbo", 
#     messages=[{"role": "user", "content": "Listen to your"}],
#     max_tokens = 1
# )

# # print the chat completion
# print(chat_completion.choices[0].message.content)

# # create a chat completion
# chat_completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", 
#                                 messages=[{"role": "user", "content": "Listen to your"}],
#                                 max_tokens = 2,
#                                 n=5)

# # print the chat completion
# for i in range(len(chat_completion.choices)):
#     print(chat_completion.choices[i].message.content)


# # create a chat completion
# chat_completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", 
#                                 messages=[{"role": "user", "content": "Listen to your"}],
#                                 max_tokens = 2,
#                                 n=5,
#                                 temperature=0)

# # print the chat completion
# for i in range(len(chat_completion.choices)):
#     print(chat_completion.choices[i].message.content)


# # create a chat completion
# chat_completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", 
#                                 messages=[{"role": "user", "content": "Listen to your"}],
#                                 max_tokens = 2,
#                                 n=5,
#                                 temperature=2)

# # print the chat completion
# for i in range(len(chat_completion.choices)):
#     print(chat_completion.choices[i].message.content)

