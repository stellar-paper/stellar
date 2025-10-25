from openai import AzureOpenAI
import os
from dotenv import load_dotenv

# Load variables from .env
load_dotenv()

azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
api_version = os.getenv("AZURE_OPENAI_API_VERSION")
api_key = os.getenv("AZURE_OPENAI_API_KEY")
deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT")

# Initialize Azure OpenAI client
client = AzureOpenAI(
    api_key=api_key,
    api_version=api_version,
    azure_endpoint=azure_endpoint
)


def call_openai(prompt, 
             max_tokens = 200, 
             temperature = 0, 
             system_prompt = None,
             model = deployment_name):
    response = client.chat.completions.create(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
    )
    input_tokens = response.usage.prompt_tokens
    output_tokens = response.usage.completion_tokens

    response_msg = response.choices[0].message.content
    return response_msg, input_tokens, output_tokens

def call_openai_gpt5_models(prompt, 
             max_completion_tokens = 200, 
             temperature = 0, 
             system_prompt = None,
             model = deployment_name,
             reasoning_effort = "minimal",
             ):
    
    if model == "gpt-5-chat":
        response = client.chat.completions.create(
                model=model,
                temperature=temperature,
                max_completion_tokens=max_completion_tokens,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ]
            )
    else:
        response = client.chat.completions.create(
            model=model,
            temperature=temperature,
            reasoning_effort=reasoning_effort,
            max_completion_tokens=max_completion_tokens,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
        )
    input_tokens = response.usage.prompt_tokens
    output_tokens = response.usage.completion_tokens

    response_msg = response.choices[0].message.content
    return response_msg, input_tokens, output_tokens

if __name__ == "__main__":
    prompt = """
                rephrase the following user utterance, leaving the variables. give 10 examples:
                'bitte zur <App> <NP_appSynonyms> gehen'
            """
    print(call_openai(prompt=prompt))