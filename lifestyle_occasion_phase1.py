#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import json
import logging
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from enum import Enum
from io import BytesIO
from pathlib import Path
from urllib.parse import urlsplit

import psycopg
import requests
from PIL import Image
from openai import OpenAI
from pydantic import BaseModel, Field, model_validator


# ============================================================
# Schema / candidates
# ============================================================


class LifestyleOccasion(str, Enum):
    EVERYDAY_CASUAL = "everyday_casual"
    WORK_OFFICE = "work_office"
    DATE_NIGHT = "date_night"
    COCKTAIL_PARTY = "cocktail_party"
    WEDDING_GUEST = "wedding_guest"
    VACATION_RESORT = "vacation_resort"
    BRUNCH_SOCIAL = "brunch_social"
    FESTIVAL_EVENT = "festival_event"
    ACTIVE_OUTDOOR = "active_outdoor"
    FORMAL_EVENT = "formal_event"


ALLOWED_LABELS = [member.value for member in LifestyleOccasion]
ALLOWED_LABELS_SET = set(ALLOWED_LABELS)


class Choice(BaseModel):
    label: LifestyleOccasion
    confidence: float = Field(..., ge=0.0, le=1.0)


class ResultModel(BaseModel):
    occasions: list[Choice] = Field(..., min_length=3, max_length=3)

    @model_validator(mode="after")
    def validate_distinct_labels(self) -> "ResultModel":
        labels = [item.label.value for item in self.occasions]
        if len(set(labels)) != 3:
            raise ValueError("occasions must contain exactly 3 distinct labels")
        return self


RESULT_SCHEMA = ResultModel.model_json_schema()


SYSTEM_PROMPT = """
You are a strict fashion lifestyle-and-occasion classifier.
You must output valid JSON only.
You must choose exactly 3 DISTINCT labels from the provided schema enum.
Use the featured product image as the primary source of truth.
Use the product title only for disambiguation.
Do not invent labels, synonyms, explanations, prose, or extra keys.
""".strip()


USER_PROMPT_TEMPLATE = """
Classify the SINGLE MAIN FASHION PRODUCT in the featured image into exactly 3 DISTINCT lifestyle/occasion labels.

Allowed labels:
everyday_casual
work_office
date_night
cocktail_party
wedding_guest
vacation_resort
brunch_social
festival_event
active_outdoor
formal_event

Rules:
- Use the image as the PRIMARY source of truth.
- Use the title only to disambiguate details that are not visually obvious.
- Focus only on the main product.
- Pick the 3 strongest-fitting lifestyle/occasion labels.
- The 3 labels must be distinct.
- Confidence must be a float between 0.0 and 1.0 for each label.
- Choose labels based on how the item would most naturally be worn by a shopper.
- Use everyday_casual for easy daily wear, lounge-like casual wear, or relaxed basics.
- Use work_office for polished, professional, office-appropriate styles.
- Use date_night for flattering, stylish, going-out looks that feel suitable for a dinner or evening date.
- Use cocktail_party for partywear, clubwear, sequins, bodycon, or elevated social-night looks.
- Use wedding_guest for dresses or outfits appropriate for attending a wedding as a guest.
- Use vacation_resort for resortwear, beachy, tropical, breezy vacation dressing.
- Use brunch_social for daytime social dressing that feels elevated but not formal.
- Use festival_event for expressive, bold, bohemian, rave, or event-oriented styling.
- Use active_outdoor for athletic, sport, hike, workout, or movement-oriented dressing.
- Use formal_event for gowns, highly formal occasionwear, or clearly dressy ceremonial looks.
- Do NOT output explanations.
- Return JSON only in this exact shape:
  {
    "occasions": [
      {"label": "<allowed_label>", "confidence": <float 0.0-1.0>},
      {"label": "<allowed_label>", "confidence": <float 0.0-1.0>},
      {"label": "<allowed_label>", "confidence": <float 0.0-1.0>}
    ]
  }

Product title:
{title}
""".strip()


_thread_local = threading.local()


def get_http_session() -> requests.Session:
    if not hasattr(_thread_local, "http_session"):
        _thread_local.http_session = requests.Session()
    return _thread_local.http_session


def get_openai_client(base_url: str, api_key: str) -> OpenAI:
    if not hasattr(_thread_local, "openai_client"):
        _thread_local.openai_client = OpenAI(base_url=base_url, api_key=api_key)
    return _thread_local.openai_client


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def utc_now_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def setup_logger(log_dir: Path, name: str) -> logging.Logger:
    ensure_dir(log_dir)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    fh = logging.FileHandler(log_dir / f"{name}.log", encoding="utf-8")
    fh.setFormatter(fmt)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def append_jsonl(path: Path, record: dict) -> None:
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_state(work_dir: Path) -> dict:
    path = work_dir / "state.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def save_state(work_dir: Path, state: dict) -> None:
    ensure_dir(work_dir)
    (work_dir / "state.json").write_text(json.dumps(state, indent=2), encoding="utf-8")


def is_valid_http_url(url: str) -> tuple[bool, str]:
    if not isinstance(url, str):
        return False, "non_string_url"

    url = url.strip()
    if not url:
        return False, "empty_url"

    try:
        parsed = urlsplit(url)
    except Exception as e:
        return False, f"urlsplit_error:{e}"

    if parsed.scheme not in {"http", "https"}:
        return False, f"invalid_scheme:{parsed.scheme}"
    if not parsed.netloc:
        return False, "missing_netloc"
    if not parsed.hostname:
        return False, "missing_hostname"

    hostname = parsed.hostname
    if len(hostname) > 253:
        return False, "hostname_too_long"

    for label in hostname.split("."):
        if not label:
            return False, "empty_dns_label"
        if len(label) > 63:
            return False, "dns_label_too_long"

    return True, "ok"


def prepare_image_data_url(image_url: str, max_side: int, timeout: int) -> str:
    session = get_http_session()
    r = session.get(image_url, timeout=timeout)
    r.raise_for_status()

    img = Image.open(BytesIO(r.content)).convert("RGB")
    img.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)

    out = BytesIO()
    img.save(out, format="JPEG", quality=90, optimize=True)
    b64 = base64.b64encode(out.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def extract_content_text(content) -> str:
    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
                elif item.get("type") == "text" and isinstance(item.get("text"), str):
                    parts.append(item["text"])
        return "\n".join(parts).strip()

    return str(content).strip()


def extract_json_object(text: str) -> str:
    text = text.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in response")
    return text[start : end + 1]


def canonicalize_label(label: str) -> str:
    if not isinstance(label, str):
        return label

    v = label.strip().lower().replace("-", "_").replace(" ", "_")
    alias_map = {'casual': 'everyday_casual', 'everyday': 'everyday_casual', 'office': 'work_office', 'work': 'work_office', 'date': 'date_night', 'party': 'cocktail_party', 'cocktail': 'cocktail_party', 'wedding': 'wedding_guest', 'resort': 'vacation_resort', 'vacation': 'vacation_resort', 'brunch': 'brunch_social', 'festival': 'festival_event', 'event': 'festival_event', 'active': 'active_outdoor', 'outdoor': 'active_outdoor', 'formal': 'formal_event'}
    return alias_map.get(v, v)


def normalize_result_payload(raw_obj: dict) -> dict:
    if isinstance(raw_obj, dict) and isinstance(raw_obj.get("occasions"), list):
        for item in raw_obj["occasions"]:
            if isinstance(item, dict) and isinstance(item.get("label"), str):
                item["label"] = canonicalize_label(item["label"])
        return raw_obj

    fallback = []
    for key in ("first", "primary", "first_choice"):
        if key in raw_obj and isinstance(raw_obj[key], dict):
            fallback.append(raw_obj[key])
            break
    for key in ("second", "secondary", "second_choice"):
        if key in raw_obj and isinstance(raw_obj[key], dict):
            fallback.append(raw_obj[key])
            break
    for key in ("third", "tertiary", "third_choice"):
        if key in raw_obj and isinstance(raw_obj[key], dict):
            fallback.append(raw_obj[key])
            break

    if len(fallback) == 3:
        for item in fallback:
            if isinstance(item.get("label"), str):
                item["label"] = canonicalize_label(item["label"])
        return {"occasions": fallback}

    raise ValueError("Response JSON does not contain expected occasions payload")


FETCH_SQL = """
SELECT
    p.product_id,
    p.title,
    img.image_url
FROM products p
JOIN LATERAL (
    SELECT pi.image_url
    FROM product_images pi
    WHERE
        pi.product_id = p.product_id
        AND pi.is_featured = TRUE
        AND pi.image_url IS NOT NULL
    ORDER BY pi.product_image_id
    LIMIT 1
) AS img ON TRUE
LEFT JOIN LATERAL (
    SELECT 1 AS has_existing
    FROM lifestyle_occasion t
    WHERE t.product_id = p.product_id
    LIMIT 1
) AS existing ON TRUE
WHERE
    p.product_id > %s
    AND p.title IS NOT NULL
    AND (%s OR existing.has_existing IS NULL)
ORDER BY p.product_id
LIMIT %s
"""

DELETE_EXISTING_SQL = "DELETE FROM lifestyle_occasion WHERE product_id = %s"
INSERT_SQL = """
INSERT INTO lifestyle_occasion (product_id, label, confidence, source_run_id)
VALUES (%s, %s, %s, %s)
"""


def fetch_candidates(conn: psycopg.Connection, after_product_id: int, limit: int, overwrite: bool):
    with conn.cursor() as cur:
        cur.execute(FETCH_SQL, (after_product_id, overwrite, limit))
        return cur.fetchall()


def flush_db_batch(
    conn: psycopg.Connection,
    grouped_rows: dict[int, list[tuple[int, str, float, str]]],
    logger: logging.Logger,
    max_retries: int,
    replace_existing: bool,
) -> None:
    if not grouped_rows:
        return

    product_ids = list(grouped_rows.keys())
    flat_rows = [row for rows in grouped_rows.values() for row in rows]

    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            with conn.cursor() as cur:
                if replace_existing:
                    cur.executemany(DELETE_EXISTING_SQL, [(pid,) for pid in product_ids])
                cur.executemany(INSERT_SQL, flat_rows)
            conn.commit()
            return
        except Exception as e:
            conn.rollback()
            last_error = e
            wait = min(30, 2 ** (attempt - 1))
            logger.warning(
                "DB batch write failed attempt=%s/%s products=%s rows=%s error=%s wait=%ss",
                attempt,
                max_retries,
                len(product_ids),
                len(flat_rows),
                e,
                wait,
            )
            if attempt < max_retries:
                time.sleep(wait)

    raise RuntimeError(f"DB batch write failed after retries: {last_error}")


def classify_one(
    product_id: int,
    title: str,
    image_url: str,
    model: str,
    base_url: str,
    api_key: str,
    image_max_side: int,
    image_timeout: int,
    request_timeout: int,
    api_max_attempts: int,
) -> dict:
    ok_url, reason = is_valid_http_url(image_url)
    if not ok_url:
        return {
            "ok": False,
            "product_id": product_id,
            "stage": "url_validation",
            "error": reason,
            "image_url": image_url,
        }

    client = get_openai_client(base_url, api_key)
    last_error = None
    last_raw = None

    for attempt in range(1, api_max_attempts + 1):
        try:
            image_data_url = prepare_image_data_url(
                image_url=image_url,
                max_side=image_max_side,
                timeout=image_timeout,
            )

            response = client.chat.completions.create(
                model=model,
                temperature=0.0,
                max_tokens=160,
                timeout=request_timeout,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": USER_PROMPT_TEMPLATE.format(title=title)},
                            {"type": "image_url", "image_url": {"url": image_data_url}},
                        ],
                    },
                ],
                extra_body={
                    "structured_outputs": {
                        "json": RESULT_SCHEMA,
                    }
                },
            )

            raw_text = extract_content_text(response.choices[0].message.content)
            last_raw = raw_text

            raw_obj = json.loads(extract_json_object(raw_text))
            normalized = normalize_result_payload(raw_obj)
            parsed = ResultModel.model_validate(normalized)

            return {
                "ok": True,
                "product_id": product_id,
                "title": title,
                "image_url": image_url,
                "occasions": [
                    {
                        "label": item.label.value,
                        "confidence": float(item.confidence),
                    }
                    for item in parsed.occasions
                ],
                "raw_output": raw_text,
            }

        except Exception as e:
            last_error = e
            if attempt < api_max_attempts:
                time.sleep(min(10, 2 ** (attempt - 1)))

    return {
        "ok": False,
        "product_id": product_id,
        "title": title,
        "stage": "inference_or_validation",
        "error": str(last_error),
        "image_url": image_url,
        "raw_output": last_raw,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Lifestyle/occasion classifier via local vLLM vision model")
    parser.add_argument("--db-url", default=os.getenv("SUPABASE_DB_URL"))
    parser.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1"))
    parser.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY", os.getenv("VLLM_API_KEY", "localtest")))
    parser.add_argument("--model", default=os.getenv("MODEL_NAME", "Qwen/Qwen2.5-VL-7B-Instruct"))

    parser.add_argument("--work-dir", default="./lifestyle_occasion_phase1_work")
    parser.add_argument("--log-name", default="lifestyle_occasion_phase1")

    parser.add_argument("--start-after-product-id", type=int, default=0)
    parser.add_argument("--max-products", type=int, default=25)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--db-batch-size", type=int, default=50)

    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--replace-existing", action="store_true")

    parser.add_argument("--image-max-side", type=int, default=1024)
    parser.add_argument("--image-timeout", type=int, default=20)
    parser.add_argument("--request-timeout", type=int, default=120)
    parser.add_argument("--api-max-attempts", type=int, default=2)
    parser.add_argument("--db-max-attempts", type=int, default=5)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--print-raw", action="store_true")

    args = parser.parse_args()

    if not args.db_url:
        raise RuntimeError("Missing --db-url or SUPABASE_DB_URL")

    work_dir = Path(args.work_dir)
    logs_dir = work_dir / "logs"
    results_dir = work_dir / "results"
    errors_dir = work_dir / "errors"

    ensure_dir(logs_dir)
    ensure_dir(results_dir)
    ensure_dir(errors_dir)

    logger = setup_logger(logs_dir, args.log_name)

    if args.resume:
        state = load_state(work_dir)
        if not state:
            raise RuntimeError("No existing state.json found for --resume")
        source_run_id = state["source_run_id"]
        logger.info(
            "Resuming source_run_id=%s last_product_id=%s attempted=%s success=%s failed=%s",
            source_run_id,
            state["last_product_id"],
            state["attempted_count"],
            state["success_count"],
            state["failed_count"],
        )
    else:
        source_run_id = f"lifestyle_occasion_phase1_{utc_now_compact()}"
        state = {
            "source_run_id": source_run_id,
            "model": args.model,
            "started_at": utc_now_iso(),
            "last_product_id": args.start_after_product_id,
            "attempted_count": 0,
            "success_count": 0,
            "failed_count": 0,
        }
        save_state(work_dir, state)

    results_jsonl = results_dir / f"results_{source_run_id}.jsonl"
    errors_jsonl = errors_dir / f"errors_{source_run_id}.jsonl"

    logger.info(
        "Starting source_run_id=%s model=%s start_after=%s max_products=%s workers=%s chunk_size=%s save=%s image_max_side=%s",
        source_run_id,
        args.model,
        state["last_product_id"],
        args.max_products,
        args.workers,
        args.chunk_size,
        args.save,
        args.image_max_side,
    )

    pending_upserts: dict[int, list[tuple[int, str, float, str]]] = {}

    with psycopg.connect(args.db_url) as read_conn, psycopg.connect(args.db_url) as write_conn:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            while state["attempted_count"] < args.max_products:
                remaining = args.max_products - state["attempted_count"]
                fetch_size = min(args.chunk_size, remaining)

                rows = fetch_candidates(
                    conn=read_conn,
                    after_product_id=int(state["last_product_id"]),
                    limit=fetch_size,
                    overwrite=args.overwrite,
                )

                if not rows:
                    logger.info("No more matching rows found. Stopping.")
                    break

                futures = []
                chunk_max_product_id = max(r[0] for r in rows)

                for product_id, title, image_url in rows:
                    futures.append(
                        executor.submit(
                            classify_one,
                            product_id,
                            title,
                            image_url,
                            args.model,
                            args.base_url,
                            args.api_key,
                            args.image_max_side,
                            args.image_timeout,
                            args.request_timeout,
                            args.api_max_attempts,
                        )
                    )

                for future in as_completed(futures):
                    result = future.result()

                    state["attempted_count"] += 1
                    state["last_product_id"] = max(int(state["last_product_id"]), int(result["product_id"]))

                    if result["ok"]:
                        state["success_count"] += 1

                        logger.info(
                            "product_id=%s title=%s occasions=%s",
                            result["product_id"],
                            result["title"],
                            ", ".join(
                                f"{item['label']} ({item['confidence']:.2f})" for item in result["occasions"]
                            ),
                        )

                        if args.print_raw:
                            logger.info("raw_output product_id=%s %s", result["product_id"], result["raw_output"])

                        append_jsonl(
                            results_jsonl,
                            {
                                "time": utc_now_iso(),
                                "source_run_id": source_run_id,
                                "product_id": result["product_id"],
                                "title": result["title"],
                                "image_url": result["image_url"],
                                "occasions": result["occasions"],
                            },
                        )

                        if args.save:
                            pending_upserts[int(result["product_id"])] = [
                                (
                                    int(result["product_id"]),
                                    item["label"],
                                    float(item["confidence"]),
                                    source_run_id,
                                )
                                for item in result["occasions"]
                            ]

                            if len(pending_upserts) >= args.db_batch_size:
                                flush_db_batch(
                                    conn=write_conn,
                                    grouped_rows=pending_upserts,
                                    logger=logger,
                                    max_retries=args.db_max_attempts,
                                    replace_existing=args.replace_existing,
                                )
                                pending_upserts.clear()
                    else:
                        state["failed_count"] += 1
                        logger.warning(
                            "product_id=%s failed stage=%s error=%s",
                            result["product_id"],
                            result.get("stage"),
                            result.get("error"),
                        )
                        append_jsonl(
                            errors_jsonl,
                            {
                                "time": utc_now_iso(),
                                "source_run_id": source_run_id,
                                "product_id": result["product_id"],
                                "title": result.get("title"),
                                "stage": result.get("stage"),
                                "error": result.get("error"),
                                "image_url": result.get("image_url"),
                                "raw_output": result.get("raw_output"),
                            },
                        )

                    if state["attempted_count"] % args.log_every == 0:
                        logger.info(
                            "Progress attempted=%s success=%s failed=%s last_product_id=%s",
                            state["attempted_count"],
                            state["success_count"],
                            state["failed_count"],
                            state["last_product_id"],
                        )
                        save_state(work_dir, state)

                if args.save and pending_upserts:
                    flush_db_batch(
                        conn=write_conn,
                        grouped_rows=pending_upserts,
                        logger=logger,
                        max_retries=args.db_max_attempts,
                        replace_existing=args.replace_existing,
                    )
                    pending_upserts.clear()

                state["last_product_id"] = max(int(state["last_product_id"]), int(chunk_max_product_id))
                save_state(work_dir, state)

    state["finished_at"] = utc_now_iso()
    save_state(work_dir, state)

    logger.info(
        "Finished source_run_id=%s attempted=%s success=%s failed=%s last_product_id=%s",
        source_run_id,
        state["attempted_count"],
        state["success_count"],
        state["failed_count"],
        state["last_product_id"],
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Fatal error: {e}", file=sys.stderr)
        sys.exit(1)
