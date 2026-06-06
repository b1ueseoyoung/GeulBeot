import os
from openai import OpenAI, AzureOpenAI
from anthropic import Anthropic

def initialize_llm_client(api_type: str = "openai"):
    if api_type == "openai":
        return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    elif api_type == "azure":
        return AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
        )
    elif api_type == "anthropic":
        return Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    else:
        raise ValueError(f"Unknown API type: {api_type}")

def get_llm_response(prompt, model, api_type, llm_client, **kwargs):
    if api_type == "openai" or api_type == "azure":
        # Filter out Claude-specific parameters
        filtered_kwargs = {k: v for k, v in kwargs.items() 
                          if k not in ['thinking', 'thinking_budget_tokens', 'reasoning_effort']}
        response = llm_client.chat.completions.create(
            model=model,
            messages=prompt,
            **filtered_kwargs
        )
        return response, response.choices[0].message.content
    elif api_type == "anthropic":
        response = llm_client.messages.create(
            model=model,
            messages=prompt,
            **kwargs
        )
        return response, response.content[0].text
    else:
        raise ValueError(f"Unknown API type: {api_type}")

def get_num_tokens(response, model, api_type):
    if api_type == "openai" or api_type == "azure":
        reasoning_tokens = 0
        if hasattr(response.usage, 'completion_tokens_details') and response.usage.completion_tokens_details:
            if hasattr(response.usage.completion_tokens_details, 'reasoning_tokens'):
                reasoning_tokens = response.usage.completion_tokens_details.reasoning_tokens or 0
        return (
            reasoning_tokens,
            response.usage.completion_tokens,
            response.usage.prompt_tokens
        )
    elif api_type == "anthropic":
        return 0, response.usage.output_tokens, response.usage.input_tokens
    else:
        return 0, 0, 0
