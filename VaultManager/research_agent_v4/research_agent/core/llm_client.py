"""Reusable OpenAI-compatible transport. Keeps raw completions before parsing."""
from typing import Protocol
import os
from pydantic import BaseModel
from .config import ModelConfig

class LLMClient(Protocol):
    identity: str
    def complete(self, messages: list[dict], schema: type[BaseModel], config: ModelConfig) -> dict: ...

class OpenAIClient:
    def __init__(self, *, provider='openrouter', base_url=None, api_key=None, timeout=120):
        from openai import OpenAI
        if provider not in {'openrouter','openai'}: raise ValueError('Use openrouter or openai, or inject an LLMClient.')
        key = api_key or os.getenv('OPENROUTER_API_KEY' if provider=='openrouter' else 'OPENAI_API_KEY')
        if not key: raise RuntimeError(f'Missing API key for {provider}; set environment or .env.')
        self.provider = provider
        url = base_url or ('https://openrouter.ai/api/v1' if provider=='openrouter' else 'https://api.openai.com/v1')
        self.identity = f'{provider}:{url}'
        self.client = OpenAI(api_key=key, base_url=url, timeout=timeout, max_retries=0)

    def complete(self, messages, schema, config):
        # create rather than parse: retain the raw response even on truncation,
        # refusal, invalid JSON or failed Pydantic validation.
        response_format = {'type':'json_schema','json_schema':{'name':schema.__name__,'strict':True,'schema':schema.model_json_schema()}} if config.response_mode=='json_schema' else {'type':'json_object'}
        body = dict(config.extra_body)
        if self.provider=='openrouter':
            body['provider'] = {**body.get('provider',{}),'require_parameters':True}
        params = dict(model=config.model, messages=messages, response_format=response_format, max_completion_tokens=config.max_tokens)
        if config.temperature is not None: params['temperature']=config.temperature
        if body: params['extra_body']=body
        # json_object mode is explicit, not an automatic downgrade; include schema.
        if config.response_mode=='json_object':
            import json
            params['messages'] = [*messages, {'role':'user','content':'Return JSON matching this schema exactly:\n'+json.dumps(schema.model_json_schema())}]
        return self.client.chat.completions.create(**params).model_dump(mode='json')

def parse_completion(raw: dict, schema: type[BaseModel]):
    choices = raw.get('choices') or []
    if not choices: raise ValueError('No choices returned')
    choice=choices[0]; message=choice.get('message') or {}
    if message.get('refusal'): raise ValueError('Model refused: '+str(message['refusal']))
    if choice.get('finish_reason')!='stop': raise ValueError(f"Unsuccessful finish_reason: {choice.get('finish_reason')}")
    content=message.get('content')
    if not isinstance(content,str) or not content.strip(): raise ValueError('Missing JSON content')
    return schema.model_validate_json(content)
