import openai
from openai import OpenAI
import time
import random

client = OpenAI()


def generate_response_multiagent(engine, max_tokens, system_role, user_input):
    # print("Generating response for engine: ", engine)
    start_time = time.time()
    response = client.chat.completions.create(
                    model=engine,
                    temperature=0,
                    max_tokens=max_tokens,
                    messages=[  
                        {"role": "system", "content": system_role},
                        {"role": "user", "content": user_input}
                    ],
                    timeout = 200
                )
    end_time = time.time()
    # print('Finish!')
    # print("Time taken: ", end_time - start_time)

    return response

class api_handler:
    def __init__(self, model):
        self.model = model
        if self.model == 'gpt4.1':
            self.engine = 'gpt-4.1'
        else:
            raise NotImplementedError

    def get_output_multiagent(self, system_role, user_input, max_tokens):
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                response = generate_response_multiagent(self.engine, max_tokens, system_role, user_input)
                if response.choices and response.choices[0].message and response.choices[0].message.content != "":
                    return response.choices[0].message.content
                else:
                    return "ERROR." 
            except (openai.APITimeoutError, openai.APIConnectionError, openai.APIError, Exception) as error:
                print(f'Attempt {attempt+1} of {max_attempts} failed with error: {error}')
                if attempt == max_attempts - 1:
                    return "ERROR."