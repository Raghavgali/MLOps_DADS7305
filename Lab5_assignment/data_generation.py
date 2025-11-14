import asyncio
import json
import logging
import os
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import aiohttp
import yaml
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

CONFIG_PATH = Path("config.yaml")
if not CONFIG_PATH.exists():
    raise FileNotFoundError("config.yaml not found in current directory")

with CONFIG_PATH.open("r", encoding="utf-8") as cfg_file:
    CONFIG = yaml.safe_load(cfg_file)

MODEL_CONFIG = CONFIG.get("model_config") or {}
PROMPT_TEMPLATE = MODEL_CONFIG.get("prompt")
SEED_QUESTIONS: List[str] = MODEL_CONFIG.get("seed_questions") or []
MODEL_NAME = MODEL_CONFIG.get("openai_model")
TEMPERATURE = MODEL_CONFIG.get("temperature", 0.0)
NUM_SAMPLES = int(MODEL_CONFIG.get("num_samples", 0))

if not PROMPT_TEMPLATE:
    raise ValueError("model_config.prompt must be provided in config.yaml")
if "{question}" not in PROMPT_TEMPLATE:
    raise ValueError("model_config.prompt must include a {question} placeholder")
if not SEED_QUESTIONS:
    raise ValueError("model_config.seed_questions must contain at least one question")
if not MODEL_NAME:
    raise ValueError("model_config.openai_model must be set")
if NUM_SAMPLES <= 0:
    raise ValueError("model_config.num_samples must be a positive integer")

API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise RuntimeError("OPENAI_API_KEY must be set in the environment")

API_URL = "https://api.openai.com/v1/responses"
BATCH_SIZE = 15  # Must stay within 10-20 as requested
MAX_RETRIES = 3
BACKOFF_FACTOR = 2.0
OUTPUT_PATH = Path("generated_data") / "dataset.json"


def normalize_question(question: Any) -> str:
    if isinstance(question, str):
        normalized = question.strip()
        return normalized if normalized else ""
    return ""


def extract_output_text(response_json: Dict[str, Any]) -> str:
    output_text = response_json.get("output_text")
    if isinstance(output_text, list) and output_text:
        return "\n".join(str(part) for part in output_text if part is not None).strip()

    text_segments: List[str] = []
    for item in response_json.get("output", []):
        for content in item.get("content", []):
            text_value = content.get("text") or content.get("content")
            if text_value:
                text_segments.append(str(text_value))
    if text_segments:
        return "\n".join(text_segments).strip()
    return json.dumps(response_json)


def fallback_record(original_question: str, raw_output: str) -> Dict[str, Any]:
    return {
        "original_question": original_question,
        "original_solution": raw_output,
        "new_question": None,
        "new_solution": None,
    }


def parse_model_output(original_question: str, model_output: str) -> Dict[str, Any]:
    try:
        parsed = json.loads(model_output)
    except (json.JSONDecodeError, TypeError):
        return fallback_record(original_question, model_output)

    if not isinstance(parsed, dict):
        return fallback_record(original_question, model_output)

    parsed.setdefault("original_question", original_question)
    parsed.setdefault("original_solution", None)
    parsed.setdefault("new_question", None)
    parsed.setdefault("new_solution", None)
    return parsed


async def call_openai(
    session: aiohttp.ClientSession,
    formatted_prompt: str,
    original_question: str,
    model_name: str,
    temperature: float,
    headers: Dict[str, str],
) -> Dict[str, Any]:
    payload = {
        "model": model_name,
        "temperature": temperature,
        "input": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": formatted_prompt,
                    }
                ],
            }
        ],
    }

    delay = 1.0
    last_error = ""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            async with session.post(API_URL, headers=headers, json=payload) as response:
                response_text = await response.text()
                if response.status != 200:
                    raise RuntimeError(f"OpenAI API error {response.status}: {response_text}")

                response_json = json.loads(response_text)
                output_text = extract_output_text(response_json)
                return parse_model_output(original_question, output_text)
        except (aiohttp.ClientError, asyncio.TimeoutError, RuntimeError, json.JSONDecodeError) as exc:
            last_error = str(exc)
            logger.warning(
                "OpenAI call failed (attempt %s/%s) for question '%s': %s",
                attempt,
                MAX_RETRIES,
                original_question,
                exc,
            )
            if attempt == MAX_RETRIES:
                break
            await asyncio.sleep(delay)
            delay *= BACKOFF_FACTOR

    return fallback_record(original_question, f"Call failed after retries: {last_error}")


async def expand_questions_batch(
    session: aiohttp.ClientSession,
    batch_questions: List[str],
    prompt_template: str,
    model_name: str,
    temperature: float,
    headers: Dict[str, str],
) -> Tuple[List[Dict[str, Any]], Set[str]]:
    tasks = []
    for question in batch_questions:
        formatted_prompt = prompt_template.replace("{question}", question)
        tasks.append(
            call_openai(
                session=session,
                formatted_prompt=formatted_prompt,
                original_question=question,
                model_name=model_name,
                temperature=temperature,
                headers=headers,
            )
        )

    responses = await asyncio.gather(*tasks, return_exceptions=True)

    results: List[Dict[str, Any]] = []
    new_questions: Set[str] = set()

    for question, response in zip(batch_questions, responses):
        if isinstance(response, Exception):
            logger.error("Unexpected error processing question '%s': %s", question, response)
            record = fallback_record(question, str(response))
        else:
            record = response
        results.append(record)

        candidate = normalize_question(record.get("new_question")) if isinstance(record, dict) else ""
        if candidate:
            new_questions.add(candidate)

    return results, new_questions


async def generate_data() -> None:
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }

    question_queue: deque[str] = deque(SEED_QUESTIONS)
    seen_questions: Set[str] = set()
    for seed_question in SEED_QUESTIONS:
        normalized = normalize_question(seed_question)
        if normalized:
            seen_questions.add(normalized)

    dataset: List[Dict[str, Any]] = []
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    timeout = aiohttp.ClientTimeout(total=120)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        with tqdm(total=NUM_SAMPLES, desc="Generating samples") as progress:
            while len(dataset) < NUM_SAMPLES:
                batch_target = min(BATCH_SIZE, NUM_SAMPLES - len(dataset))
                current_batch: List[str] = []
                while len(current_batch) < batch_target:
                    if not question_queue:
                        raise RuntimeError("Question pool exhausted unexpectedly")
                    question = question_queue.popleft()
                    current_batch.append(question)
                    question_queue.append(question)

                batch_results, new_questions = await expand_questions_batch(
                    session=session,
                    batch_questions=current_batch,
                    prompt_template=PROMPT_TEMPLATE,
                    model_name=MODEL_NAME,
                    temperature=TEMPERATURE,
                    headers=headers,
                )

                for new_question in new_questions:
                    if new_question and new_question not in seen_questions:
                        seen_questions.add(new_question)
                        question_queue.append(new_question)

                for record in batch_results:
                    dataset.append(record)
                    progress.update(1)
                    if len(dataset) >= NUM_SAMPLES:
                        break

    with OUTPUT_PATH.open("w", encoding="utf-8") as output_file:
        json.dump(dataset, output_file, indent=2, ensure_ascii=False)
    logger.info("Saved %s samples to %s", len(dataset), OUTPUT_PATH)


if __name__ == "__main__":
    asyncio.run(generate_data())
