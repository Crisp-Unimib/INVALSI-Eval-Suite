#Fixare prompt e matching-->provare fast matching

import json
import logging
import re
import threading
import time
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import hydra
import numpy as np
import pandas as pd
import tenacity
from loguru import logger
from omegaconf import DictConfig
from pydantic import BaseModel
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from evaluate import load as load_evaluate

# Import special cases (promptXX_YY, evalXX_YY)
from special_cases import *


logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logger.opt(colors=True)

# Dizionario key -> [prompt_fn, eval_fn]
special_cases = {
#     #'23_14': [prompt23_14, eval23_14],
     '24_14': [prompt24_14, eval24_14],
     '26_14': [prompt26_14, eval26_14],
     '29_14': [prompt29_14, eval29_14],
     '21_15': [prompt21_15, eval21_15],
     '35_15': [prompt35_15, eval35_15],
     '54_15': [prompt54_15, eval54_15],
     '55_15': [prompt55_15, eval55_15],
     '57_15': [prompt57_15, eval57_15],
     '10_17': [prompt10_17, eval10_17],
     '19_17': [prompt19_17, eval19_17],
     '1_30': [prompt1_30, eval1_30],
     '2_30': [prompt2_30, eval2_30],
     '3_30': [prompt3_30, eval3_30],
     '4_30': [prompt4_30, eval4_30],
#     '34_30': [prompt34_30, eval34_30],
#     # di questi due dovremmo ampliare semanticamente le frasi di riferimento (forse valutabile solo a mano)
     '51_30': [prompt51_30, eval51_30],
#     # di questi due dovremmo ampliare semanticamente le frasi di riferimento (forse valutabile solo a mano)
     '52_30': [prompt52_30, eval52_30],
     '53_30': [prompt53_30, eval53_30],
     '55_30': [prompt55_30, eval55_30],
     '58_30': [prompt58_30, eval58_30],
     '69_30': [prompt69_30, eval69_30],
     '71_30': [prompt71_30, eval71_30],
     '73_30': [prompt73_30, eval73_30],
     '74_30': [prompt74_30, eval74_30],
     '75_30': [prompt75_30, eval75_30],
     '39_49': [prompt39_49, eval39_49],
     '40_49': [prompt40_49, eval40_49],
     '41_49': [prompt41_49, eval41_49],
     '45_49': [prompt45_49, eval45_49],
     '51_53': [prompt51_53, eval51_53],
     '52_53': [prompt52_53, eval52_53],
     '53_53': [prompt53_53, eval53_53],
     '54_53': [prompt54_53, eval54_53],
#     # di questi due dovremmo ampliare semanticamente le frasi di riferimento (forse valutabile solo a mano)
     '25_57': [prompt25_57, eval25_57],
     '27_57': [prompt27_57, eval27_57],
     '28_57': [prompt28_57, eval28_57],
     '48_57': [prompt48_57, eval48_57],
 }

############################################################
# 1) Costanti e template
############################################################

DEFAULT_SYSTEM_MESSAGE = "Sei un assistente utile."

# QUERY_TEMPLATE_MULTICHOICE = """
# Rispondi alla seguente domanda a scelta multipla. 
# L'ultima riga della tua risposta deve essere nel seguente formato: 
# 'Risposta: LETTERA' (senza virgolette) dove LETTERA è una tra {merged_letters}. 


# Domanda:
# {question}

# {options}
# """.strip()

QUERY_TEMPLATE_MULTICHOICE = """Rispondi alla seguente domanda a scelta multipla. 
La tua risposta deve essere nel seguente formato: 'LETTERA' (senza virgolette) dove LETTERA è una tra {merged_letters}. 
Scrivi solo la lettera corrispondente alla tua risposta senza spiegazioni.

Domanda:
{question}

Opzioni
{options}

Risposta:
""".strip()


QUERY_TEMPLATE_MULTICHOICE_CNTX = """
Rispondi alla seguente domanda a scelta multipla basandoti sul seguente contesto: \"""{context}\""".
 La tua risposta deve essere nel seguente formato: 'LETTERA' (senza virgolette) dove LETTERA è una tra {merged_letters}. 
 Scrivi solo la lettera corrispondente alla tua risposta senza spiegazioni.

Domanda:
{question}

Opzioni
{options}

Risposta:
""".strip()

# QUERY_TEMPLATE_MULTICHOICE_CNTX = """
# Rispondi alla seguente domanda a scelta multipla basandoti sul seguente contesto: \"""{context}\""" 
# L'ultima riga della tua risposta deve essere nel seguente formato: 
# 'Risposta: LETTERA' (senza virgolette) dove LETTERA è una tra {merged_letters}. 

# Domanda:
# {question}

# {options}
# """.strip()

QUERY_TEMPLATE_OPEN = """
Rispondi alla seguente domanda a risposta aperta.

Domanda:
{question}

Risposta:
""".strip()

QUERY_TEMPLATE_OPEN_CNTX = """
Rispondi alla seguente domanda a risposta aperta basandoti sul seguente contesto: \"""{context}\""".

Domanda:
{question}

Risposta:
""".strip()

############################################################
# 2) Provider multipli
############################################################

class ProviderEnum(str, Enum):
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    GOOGLE = "google"
    CUSTOM_OPENAI = "custom_openai"
    LOCAL = "local"
    OPENROUTER = "openrouter"

class BaseProvider(ABC):
    @abstractmethod
    def complete(
        self,
        model: str,
        messages: List[Dict[str, str]],  # Chat style
        temperature: float,
        max_tokens: int,
    ) -> str:
        pass

class GoogleProvider(BaseProvider):
    def __init__(self, api_key: str):
        import google.generativeai as genai  # type: ignore
        self.genai = genai
        self.genai.configure(api_key=api_key)

    def complete(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
    ) -> str:
        # costruiamo la history
        # role=system => system_instruction
        # role=user => user, role=assistant => ...
        # assumiamo che messages[0] = system
        system_message = messages[0]["content"] if messages else DEFAULT_SYSTEM_MESSAGE
        generation_config = {
            "temperature": temperature,
            "max_output_tokens": max_tokens,
            "response_mime_type": "text/plain",
        }
        from google.generativeai import GenerativeModel
        mq = GenerativeModel(
            model_name=model,
            generation_config=generation_config,
            system_instruction=system_message
        )

        # Convert the rest to google's format
        # e.g. role=user => "parts": message["content"]
        google_history = []
        if len(messages) > 1:
            for m in messages[1:-1]:
                # user or assistant
                if m["role"] == "user":
                    google_history.append({"role": "user", "parts": m["content"]})
                else:
                    google_history.append({"role": "assistant", "parts": m["content"]})

        chat_session = mq.start_chat(history=google_history)
        user_msg = messages[-1]["content"]
        response = chat_session.send_message(user_msg)
        return response.text.strip()


class AnthropicProvider(BaseProvider):
    def __init__(self, api_key: str):
        import anthropic
        self.client = anthropic.Anthropic(api_key=api_key)

    def complete(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
    ) -> str:
        # Nel pacchetto anthropic di solito c'è un "prompt" unico.
        # Puoi unire i messages in un unico prompt con i tag speciali
        # se ti serve una chat simulation. Ecco un esempio semplice:
        from anthropic import HUMAN_PROMPT, AI_PROMPT
        system = messages[0]["content"] if messages else DEFAULT_SYSTEM_MESSAGE
        # costruiamo uno pseudo chat
        chat_text = f"{system}\n\n"
        for m in messages[1:]:
            if m["role"] == "user":
                chat_text += f"{HUMAN_PROMPT} {m['content']}\n"
            else:
                chat_text += f"{AI_PROMPT} {m['content']}\n"

        # anthropic max_tokens si intende tokens di output
        # se la lunghezza di chat_text è troppa, potresti trunkare.
        response = self.client.completions.create(
            model=model,
            prompt=chat_text,
            max_tokens_to_sample=max_tokens,
            temperature=temperature,
        )
        return response.completion.strip()

class LocalHFProvider(BaseProvider):
    def __init__(self, model_repo: str = "DeepMount00/Llama-3.1-8b-Ita", **kwargs):
        self.tokenizer = AutoTokenizer.from_pretrained(model_repo)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_repo,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            **kwargs
        ).eval()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            self.model.resize_token_embeddings(len(self.tokenizer))

    def complete(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
    ) -> str:
        # Uniamo system e user messages in un unico prompt testuale
        # (non esiste uno standard chat come openai)
        final_prompt = ""
        for m in messages:
            if m["role"] == "system":
                final_prompt += f"[SYSTEM] {m['content']}\n"
            elif m["role"] == "user":
                final_prompt += f"[USER] {m['content']}\n"
            else:
                final_prompt += f"[ASSISTANT] {m['content']}\n"

        inputs = self.tokenizer(final_prompt, return_tensors='pt', padding=True, truncation=True).to(self.device)
        generate_kwargs = {
            "max_length": inputs.input_ids.size(1) + max_tokens,
            "eos_token_id": self.tokenizer.eos_token_id,
            "temperature": temperature
        }
        output = self.model.generate(**inputs, **generate_kwargs)
        return self.tokenizer.decode(output[0], skip_special_tokens=True)

# Esempio di un eventuale OpenRouterProvider
class OpenRouterProvider(BaseProvider):
    def __init__(self, api_key: str, **kwargs):
        # Conservi l'api_key e altri parametri
        self.api_key = api_key
        self.kwargs = kwargs

    def complete(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
    ) -> str:
        # Esempio: potresti dover mandare una POST a openrouter.ai
        # con un body simile a:
        # {
        #   "model": model,
        #   "messages": [...],
        #   "max_tokens": max_tokens,
        #   "temperature": temperature
        # }
        #
        # NB: Questo è pseudo-codice, personalizzalo in base a come funziona la tua API
        import requests
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        openrouter_messages = [{"role": m["role"], "content": m["content"]} for m in messages]
        data = {
            "model": model,
            "messages": openrouter_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        resp_json = response.json()
        # Supponiamo che la risposta stia in resp_json["choices"][0]["message"]["content"]
        return resp_json["choices"][0]["message"]["content"].strip()


def model_factory(provider: ProviderEnum, api_key: str, **kwargs) -> BaseProvider:
    if provider == ProviderEnum.OPENAI:
        return OpenAIProvider(api_key=api_key, **kwargs)
    elif provider == ProviderEnum.ANTHROPIC:
        return AnthropicProvider(api_key=api_key)
    elif provider == ProviderEnum.GOOGLE:
        return GoogleProvider(api_key=api_key)
    elif provider == ProviderEnum.CUSTOM_OPENAI:
        return OpenAIProvider(api_key=api_key, **kwargs)
    elif provider == ProviderEnum.LOCAL:
        return LocalHFProvider(**kwargs)
    elif provider == ProviderEnum.OPENROUTER:
        return OpenRouterProvider(api_key=api_key, **kwargs)
    else:
        raise ValueError(f"Provider {provider} not supported.")


class Provider(BaseProvider):
    def __init__(self, api_key: str, provider: ProviderEnum, **kwargs):
        self.provider = model_factory(provider, api_key, **kwargs)

    def complete(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int
    ) -> str:
        return self.provider.complete(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )

############################################################
# 3) Strutture di dati e RateLimiter
############################################################

class ChatCompletionRequest(BaseModel):
    index: int
    provider: ProviderEnum
    model: str
    messages: List[Dict[str, str]]  # Chat style
    answer: List[str]  # Più possibili risposte corrette
    temperature: float = 0.7
    max_tokens: int = 150

class ChatCompletionResponse(ChatCompletionRequest):
    output: str


class OpenAIProvider(BaseProvider):
    def __init__(self, api_key: str, **kwargs):
        from openai import OpenAI  # type: ignore

        self.client = OpenAI(api_key=api_key, **kwargs)

    def complete(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
    ) -> ChatCompletionResponse:
        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        return response.choices[0].message.content.strip()
    
class RateLimiter:
    def __init__(self, requests_per_minute: int):
        self.rate = requests_per_minute
        self.interval = 60.0 / requests_per_minute
        self.lock = threading.Lock()
        self.request_times = deque()

    def wait(self):
        current_time = time.time()
        with self.lock:
            while self.request_times and current_time - self.request_times[0] >= 60:
                self.request_times.popleft()
            if len(self.request_times) < self.rate:
                self.request_times.append(current_time)
                return True
            else:
                return False

    def throttle_requests(self):
        while not self.wait():
            time.sleep(0.1)

    def get_total_requests(self):
        with self.lock:
            return len(self.request_times)

############################################################
# 4) Estrazione lettera e template
############################################################

def fallback_extract_letter(text: str) -> str:
    bracket_match = re.search(r"\[([A-Z])\]", text, re.IGNORECASE)
    if bracket_match:
        return bracket_match.group(1).upper()

    def _find(pattern: str, txt: str) -> str:
        flags = re.DOTALL | re.IGNORECASE
        match = re.search(pattern, txt, flags)
        if match:
            ans = re.sub(r"[è:)(?+-,;.]", "", match.group(1)).strip()
            ans = re.sub(r"^(?:sarà\s+la\s+|la\s+)?", "", ans).strip()
            return ans.upper()
        return ""

    def_pattern = r"Risposta:\s*(.*?)\s*(?=\n[A-Z]\)|\Z)"
    fallback_patterns = [
        #  r"quindi, la risposta è\s*(.*?)\s*(?=\n[A-Z]\)|\Z)",
        #  r"risposta\s*(?:corretta|giusta|appropriata|esatta|migliore|ottimale|finale|definitiva)?\s*[:è]+\s*([A-Z])\b"
        #  r"risposta\s*più\s*(?:corretta|appropriata)\s*[:è]*\s*(.*?)\s*(?=\n[A-Z]\)|\Z)",
        #  r"(?:soluzione|opzione|scelta|alternativa)\s*(?:corretta)?\s*[:è]*\s*(.*?)\s*(?=\n[A-Z]\)|\Z)",
        # #r"(?:quindi|in\s*conclusione,?)?\s*(?:la\s*)?risposta\s*è\s*[:è]?\s*(.+?)(?:[\.\n]|$)"
        # r"(?:la\s*)?(?:risposta|opzione|scelta)\s*(?:corretta|giusta|esatta)\s*è\s*(?:la\s*)?(?:lettera\s*)?([A-Z])",
        r"\b([A-Z])\.\s",  # Nuovo: cattura lettere con punto e spazio (es. "B. ") #nuovo. lettera e spazio ?
        r"\b([A-Z])\.\s*",  # Nuovo: cattura "C." anche senza confine di parola (\b)
        r"\b([A-Z])\s*(?:\n|$)",
        r"^([A-Z])$", #solo lettera maiuscola
        r"([A-Z])\)", #lettera e parentesi
        r"\s([A-Z])(?:\.\s*)",  # Nuovo: cattura " C." anche con spazio prima
        r"\.([A-Z])",# (ad esempio :A o .A)
        r":([A-Z])",
        r"[^\w\s]*([A-Z])"
    ]

    letter_found = _find(def_pattern, text)
    if letter_found:
        return letter_found[0]
    if "nessuna delle opzioni" in text.lower():
        return ""

    for pat in fallback_patterns:
        letter_found = _find(pat, text)
        if letter_found:
            return letter_found[0].upper()

    return ""

def build_chat_messages(row: Dict[str, Any]) -> List[Dict[str, str]]:
    key = row["key"]
    print(key)
    if key in special_cases:
        print(f'gestisco caso {key}')
        prompt_fn = special_cases[key][0]
        user_prompt = prompt_fn(row)
        return [
            {"role": "system", "content": DEFAULT_SYSTEM_MESSAGE},
            {"role": "user", "content": user_prompt},
        ]

    question = row["question"]
    category = row["macro_category"].strip()
    context = row['context']
    options_list = row.get("options", [])

    formatted_options = ""
    merged_letters = ""
    for opt in options_list:
        splitted = opt["scelta"].split(".", 1)
        letter = splitted[0].strip()
        merged_letters += letter
        formatted_options += f"{opt['scelta']}\n"

    if not options_list and category != "MCC":
        # RU aperta
        if context: 
            user_content = QUERY_TEMPLATE_OPEN_CNTX.format(question=question, context=context)
        else:
            user_content = QUERY_TEMPLATE_OPEN.format(question=question)
            return [
                {"role": "system", "content": DEFAULT_SYSTEM_MESSAGE},
                {"role": "user", "content": user_content},
            ]
    if context:
        user_content = QUERY_TEMPLATE_MULTICHOICE_CNTX.format(
            question=question,
            options=formatted_options,
            merged_letters=merged_letters,
            context = context) 
    else:
        user_content = QUERY_TEMPLATE_MULTICHOICE.format(
            question=question,
            options=formatted_options,
            merged_letters=merged_letters
        )
    return [
        {"role": "system", "content": DEFAULT_SYSTEM_MESSAGE},
        {"role": "user", "content": user_content},
    ]

############################################################
# 5) Singola request
############################################################

def before_retry(retry_state):
    # Access the request from the retry state (if it's the first argument)
    request = retry_state.args[0]
    attempt = retry_state.attempt_number
    print(f"Retry attempt {attempt} for request with index {request.index}. Received null/empty response.")

@tenacity.retry(
    retry=tenacity.retry_if_exception_type(Exception),
    wait=tenacity.wait_exponential(multiplier=1, max=5),
    stop=tenacity.stop_after_attempt(50),
    before_sleep=before_retry
)
def process_request(
    request: ChatCompletionRequest,
    client: Provider,
    rate_limiter: Optional[RateLimiter] = None
) -> ChatCompletionResponse:
    if rate_limiter:
        rate_limiter.throttle_requests()
    output = client.complete(
        model=request.model,
        messages=request.messages,
        temperature=request.temperature,
        max_tokens=request.max_tokens
    )
    if not output or not output.strip():
        raise ValueError("Received null or empty response")
    
    return ChatCompletionResponse(**request.dict(), output=output)

############################################################
# 6) Group MCC e match
############################################################

def group_mcc_and_fix_matches(results: List[ChatCompletionResponse], data: List[Dict[str, Any]]) -> None:
    index2row = {i: row for i, row in enumerate(data)}
    mcc_groups = defaultdict(list)

    # Raggruppa le MCC
    for i, resp in enumerate(results):
        row_dict = index2row[resp.index]
        if row_dict["macro_category"].strip() == "MCC":
            group_key = (row_dict["Prova_PK"], row_dict["Unit_PK"], row_dict["Domanda_PK"])
            mcc_groups[group_key].append(i)

    # Valutazione
    for i, resp in enumerate(results):
        row_dict = index2row[resp.index]
        key = row_dict["key"]

        if key in special_cases:
            # special case
            eval_fn = special_cases[key][1]
            match_val = eval_fn(resp.output, row_dict["answer"])
            object.__setattr__(resp, "match", match_val)
            # Non sappiamo la lettera esatta estratta dal case, potresti anche 
            # gestirla dentro evalXX_YY e restituirla, ma per semplicità:
            object.__setattr__(resp, "predicted_answer", "special-case")
        else:
            cat = row_dict["macro_category"].strip()
            answers_list = row_dict["answer"]
            match_val = 0
            predicted = ""  # qui salveremo la risposta estratta

            if cat in ["MC", "MC ", "MCC"]:
                letter = fallback_extract_letter(resp.output)
                predicted = letter  # Salviamo la lettera
                if any(letter.upper() == ans.upper() for ans in answers_list):
                    match_val = 1

            elif cat == "RU":
                opts = row_dict.get("options", [])
                if opts:
                    # RU con opzioni => trattiamola come multi-choice
                    letter = fallback_extract_letter(resp.output)
                    predicted = letter
                    # Se la lettera estratta è in row["answer"], match=1
                    if any(letter.upper() == ans.upper() for ans in answers_list):
                        match_val = 1
                    else:
                        match_val = 0
                else:
                    # RU senza opzioni => domanda aperta pura
                    # Tuo meccanismo di matching personalizzato.
                    # Ad esempio, BERTScore, un check di substring, ecc.
                    # Se non hai logica, lo lasci a 0:
                    match_val = 0
                    predicted = ""


            else:
                match_val = 0

            # Salviamo i risultati
            object.__setattr__(resp, "match", match_val)
            object.__setattr__(resp, "predicted_answer", predicted)

    # Correzione MCC
    for group_key, idxs in mcc_groups.items():
        if not all(results[idx].match for idx in idxs):
            for idx in idxs:
                object.__setattr__(results[idx], "match", 0)
        else:
            for idx in idxs:
                object.__setattr__(results[idx], "match", 1)


############################################################
# 7) Calcolo accuracy e salvataggio CSV
############################################################

def _compute_stat(values: List[float], stat: str):
    if stat == "mean":
        return float(np.mean(values)) if values else 0
    elif stat == "std":
        return float(np.std(values)) if values else 0
    elif stat == "min":
        return float(np.min(values)) if values else 0
    elif stat == "max":
        return float(np.max(values)) if values else 0
    raise ValueError(f"Unknown stat: {stat}")

def aggregate_results(
    responses: List[ChatCompletionResponse],
) -> Dict[str, float]:
    """ Calcola solo l'accuracy complessiva """
    total_matches = sum(getattr(resp, "match", 0) for resp in responses)
    accuracy = total_matches / len(responses) if responses else 0
    return {"accuracy": accuracy}





def save_responses_csv(
    responses: List[ChatCompletionResponse],
    data: List[Dict[str, Any]],
    csv_path: Path
):
    import csv
    fieldnames = [
        "index", "key", "Prova_PK", "Unit_PK", "Domanda_PK", "Grado",
        "Tipologia", "Sezione", "MacroAspetto",
        "Domanda", "Opzioni", "Contesto",
        "PromptGenerato", "Output", "RisposteCorrette", 
        "PredictedAnswer",  # <--- colonna nuova
        "Match"
    ]

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for resp in sorted(responses, key=lambda x: x.index):
            row = data[resp.index]
            domanda = row["question"]
            contesto = row.get("context", "")
            opzioni = row.get("options", [])
            str_opzioni = "\n".join(o["scelta"] for o in opzioni)
            prompt_text = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in resp.messages)
            correct_answers = ", ".join(resp.answer)
            match_val = getattr(resp, "match", 0)

            # Ecco il nostro pred. Se non c'è, vuoto.
            pred_ans = getattr(resp, "predicted_answer", "")

            writer.writerow({
                "index": resp.index,
                "key": row["key"],
                "Prova_PK": row["Prova_PK"],
                "Unit_PK": row["Unit_PK"],
                "Domanda_PK": row["Domanda_PK"],
                "Grado": row['Grado'],
                "Tipologia":row['macro_category'],
                "Sezione": row['section'], 
                "MacroAspetto":row['lang_aspect'],
                "Domanda": domanda,
                "Opzioni": str_opzioni,
                "Contesto": contesto,
                "PromptGenerato": prompt_text,
                "Output": resp.output.strip(),
                "RisposteCorrette": correct_answers,
                "PredictedAnswer": pred_ans,
                "Match": match_val
            })



def save_metrics_csv(
    metrics: Dict[str, float],
    csv_path: Path
):
    import csv
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for k, v in metrics.items():
            writer.writerow([k, v])



############################################################
# 8) Checkpoint in JSON
############################################################

def save_intermediate_results_json(
    responses: List[ChatCompletionResponse],
    checkpoint_file: Path
):
    sorted_resps = sorted(responses, key=lambda x: x.index)
    data_to_save = [r.dict() for r in sorted_resps]
    with checkpoint_file.open("w", encoding="utf-8") as f:
        json.dump(data_to_save, f, indent=2, ensure_ascii=False)

def load_intermediate_results_json(
    checkpoint_file: Path
) -> Tuple[List[ChatCompletionResponse], Set[int]]:
    if not checkpoint_file.exists():
        logger.info(f"No checkpoint found at {checkpoint_file}, starting fresh.")
        return [], set()
    logger.info(f"Loading checkpoint from {checkpoint_file}")
    with checkpoint_file.open("r", encoding="utf-8") as f:
        data_loaded = json.load(f)
    responses = [ChatCompletionResponse(**item) for item in data_loaded]
    processed_ids = set(r.index for r in responses)
    return responses, processed_ids


############################################################
# 9) Pipeline
############################################################

def process(
    requests: List[ChatCompletionRequest],
    client: Provider,
    config: DictConfig,
    data_rows: List[Dict[str, Any]]
):
    # Limit
    if config.limit:
        requests = requests[: config.limit]

    # Crea subdir
    def clean_model(x: str) -> str:
        return x.replace("/", "_").replace(" ", "_").replace("-", "_").replace(":", "_")
    model_subdir = clean_model(config.model)
    model_dir = Path(config.data.output_dir) / model_subdir
    model_dir.mkdir(parents=True, exist_ok=True)

    # Checkpoint JSON
    checkpoint_file = model_dir / "checkpoint.json"

    rate_limiter = None
    if config.rate_limiting.enabled:
        logger.info("Rate limiting enabled.")
        rate_limiter = RateLimiter(config.rate_limiting.requests_per_minute)

    # Carichiamo i partial se auto_resume
    all_responses: List[ChatCompletionResponse] = []
    processed_ids: Set[int] = set()
    if config.checkpointing.enabled and config.auto_resume:
        prev_res, prev_ids = load_intermediate_results_json(checkpoint_file)
        all_responses.extend(prev_res)
        processed_ids |= prev_ids

    # Prepara requests da eseguire
    remaining = [req for req in requests if req.index not in processed_ids]
    if not remaining:
        logger.info("<red>All requests have been processed in previous runs.</red>")
        return

    pbar = tqdm(total=len(remaining), desc="Processing responses")
    counter = 0
    with ThreadPoolExecutor(max_workers=min(config.num_threads, len(remaining))) as executor:
        futs = [
            executor.submit(process_request, req, client, rate_limiter)
            for req in remaining
        ]
        for fut in as_completed(futs):
            resp = fut.result()
            all_responses.append(resp)
            pbar.update(1)
            counter += 1
            # checkpoint
            if config.checkpointing.enabled and config.checkpointing.checkpoint_interval:
                if counter % config.checkpointing.checkpoint_interval == 0:
                    # Salviamo JSON
                    save_intermediate_results_json(all_responses, checkpoint_file)
                    logger.info("[CHECKPOINT] partial JSON saved.")
    pbar.close()

    # Calcolo match + grouping MCC
    group_mcc_and_fix_matches(all_responses, data_rows)

    # Calcolo metriche
    metrics = aggregate_results(all_responses)
    logger.info(f"Metrics: {metrics}")

    # Salvataggio CSV finali
    results_csv = model_dir / "generations_debug.csv"
    metrics_csv = model_dir / "metrics.csv"

    save_responses_csv(all_responses, data_rows, results_csv)
    logger.info(f"[INFO] Results CSV => {results_csv}")

    save_metrics_csv(metrics, metrics_csv)
    logger.info(f"[INFO] Metrics CSV => {metrics_csv}")

    # Salvataggio finale del checkpoint con TUTTE le risposte
    if config.checkpointing.enabled:
        save_intermediate_results_json(all_responses, checkpoint_file)
        logger.info("[CHECKPOINT] final JSON saved.")


############################################################
# 10) Caricamento requests
############################################################

def load_requests(config: DictConfig) -> Tuple[List[ChatCompletionRequest], List[Dict[str, Any]]]:
    df = pd.read_json(config.data.data_file, lines=True)
    data = df.to_dict(orient="records")
    for row in data:
        row["key"] = f"{row['ID']}_{row['Prova_PK']}"

    

    out_requests = []
    for i, row in enumerate(data):
        messages = build_chat_messages(row)
        raw_ans = row.get("answer", [])
        if isinstance(raw_ans, str):
            raw_ans = [raw_ans.strip()]
        elif isinstance(raw_ans, list):
            raw_ans = [str(a).strip() for a in raw_ans if a]

        req = ChatCompletionRequest(
            index=i,
            provider=ProviderEnum(config.provider),
            model=config.model,
            messages=messages,
            answer=raw_ans,
            temperature=config.temperature,
            max_tokens=config.max_tokens
        )
        out_requests.append(req)
    return out_requests, data


############################################################
# 11) Entry point Hydra
############################################################

@hydra.main(version_base=None, config_path=".", config_name="Invalsi_config")
def run(config: DictConfig):
    logger.info("<bold>🔎 | Running advanced evaluation pipeline | 🔎</bold>")

    client = Provider(
        api_key=config.api_key,
        provider=ProviderEnum(config.provider),
        **config.get("provider_kwargs", {}),
    )

    requests, data_rows = load_requests(config)
    #print(data_rows)
    process(requests, client, config, data_rows)


if __name__ == "__main__":
    run()






