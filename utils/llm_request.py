from groq import Groq


def fill_placeholders(prompt: str, values: dict) -> str:
    """Replaces placeholders with their corresponding values"""
    return prompt.format(**values)


def get_answer(client: Groq, prompt: str, user_input: str,
               config: dict, model: str) -> str:

    messages = [{'role': 'system', 'content': prompt},
                {'role': 'user', 'content': user_input}]

    kwargs = {
        'messages': messages,
        'model': model,
        'temperature': config['temperature'],
        'top_p': config['top_p'],
        'stop': None,
        'max_completion_tokens': config['max_completion_tokens'],
        'stream': False
    }

    if client.__class__.__module__.startswith("openai"):
        # OpenAI uses 'max_tokens'
        kwargs.pop("max_completion_tokens", None)

    answer = client.chat.completions.create(**kwargs)

    return answer.choices[0].message.content.strip()
