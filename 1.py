import os
import time
import random
import logging
import requests
import pandas as pd
from dateutil.parser import isoparse
from datetime import datetime, timezone

BASE_URL = "https://api.timepad.ru/v1/events.json"

TIMEPAD_TOKEN = 
USE_TOKEN = bool(TIMEPAD_TOKEN)

FIELDS = [
    "location",
    "ticket_types",
    "categories",
    "tickets_limit",
    "organization",
    "registration_data",
    "age_limit",
    "properties",
    "moderation_status",
    "access_status",
    "created_at",
    "description_short",
]

OUT_RAW = "events_raw_stream.csv"         # все проведенные (даже если participants пустой)
OUT_ML = "events_ml_stream.csv"           # только где participants не пустой
LOG_LEVEL = logging.INFO

def setup_logger():
    logging.basicConfig(
        level=LOG_LEVEL,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    return logging.getLogger("timepad")

log = setup_logger()

def season(month: int) -> str:
    if month in (12, 1, 2): return "winter"
    if month in (3, 4, 5): return "spring"
    if month in (6, 7, 8): return "summer"
    return "autumn"

def build_headers() -> dict:
    h = {
        "Accept": "application/json",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) timepad-parser/1.0",
        "Accept-Language": "ru-RU,ru;q=0.9,en;q=0.8",
        "Connection": "keep-alive",
    }
    if USE_TOKEN and TIMEPAD_TOKEN:
        h["Authorization"] = f"Bearer {TIMEPAD_TOKEN}"
    return h

def request_events(params_items, timeout=40, max_retries=8):
    headers = build_headers()

    for attempt in range(1, max_retries + 1):
        try:
            r = requests.get(BASE_URL, params=params_items, headers=headers, timeout=timeout)
        except requests.RequestException as e:
            sleep_s = 1.0 + attempt * 0.8 + random.random()
            log.warning(f"network error: {e} | retry {attempt}/{max_retries} | sleep {sleep_s:.2f}s")
            time.sleep(sleep_s)
            continue

        status = r.status_code

        if status == 429:
            sleep_s = 1.5 + attempt * 0.7 + random.random()
            log.warning(f"429 rate limit | retry {attempt}/{max_retries} | sleep {sleep_s:.2f}s")
            time.sleep(sleep_s)
            continue

        if status in (500, 502, 503, 504):
            sleep_s = 1.0 + attempt * 0.8 + random.random()
            log.warning(f"{status} server error | retry {attempt}/{max_retries} | sleep {sleep_s:.2f}s")
            time.sleep(sleep_s)
            continue

        if status == 403:
            body = r.text[:500]
            raise RuntimeError(
                "403 Forbidden.\n"
                "Причины: антибот по IP, лимиты, токен без прав.\n"
                f"Ответ: {body}"
            )

        if status >= 400:
            body = r.text[:500]
            raise RuntimeError(f"HTTP {status}. Ответ: {body}")

        return r.json()

    raise RuntimeError("Не удалось получить данные после повторов (rate limit/сеть).")

def iter_events(starts_at_min, starts_at_max, limit=100, skip_start=0,
               sort=("+starts_at", "+id"),
               moderation_statuses=("featured", "shown", "not_moderated"),
               show_empty_fields=True):
    skip = int(skip_start)
    total = None

    log.info(f"fetch events | range=[{starts_at_min}..{starts_at_max}] | limit={limit}")

    while True:
        params = []
        params += [("limit", int(limit)), ("skip", int(skip))]
        params += [("starts_at_min", starts_at_min), ("starts_at_max", starts_at_max)]
        if show_empty_fields:
            params += [("show_empty_fields", "true")]

        for s in sort:
            params += [("sort", s)]
        for f in FIELDS:
            params += [("fields", f)]
        if moderation_statuses:
            for ms in moderation_statuses:
                params += [("moderation_statuses", ms)]

        data = request_events(params)

        if total is None:
            total = int(data.get("total", 0))
            log.info(f"total={total}")

        values = data.get("values", [])
        log.info(f"page | skip={skip} | got={len(values)}")

        if not values:
            break

        for ev in values:
            yield ev

        skip += len(values)
        if skip >= total:
            break

        time.sleep(0.25)

def parse_dt_safe(v):
    if not v:
        return None
    try:
        return isoparse(v)
    except Exception:
        return None

def is_conducted(ev, now_utc):
    """
    Проведено:
      - если ends_at есть и ends_at < now
      - иначе starts_at < now
    """
    s = parse_dt_safe(ev.get("starts_at"))
    e = parse_dt_safe(ev.get("ends_at"))
    if e is not None:
        return e.astimezone(timezone.utc) < now_utc
    if s is not None:
        return s.astimezone(timezone.utc) < now_utc
    return False

def normalize_one_event(ev, year, now_utc):
    # только проведенные
    if not is_conducted(ev, now_utc):
        return None

    starts_raw = ev.get("starts_at")
    if not starts_raw:
        return None

    starts_dt = parse_dt_safe(starts_raw)
    ends_dt = parse_dt_safe(ev.get("ends_at"))

    loc = ev.get("location") or {}
    cats = ev.get("categories") or []
    ttypes = ev.get("ticket_types") or []
    reg = ev.get("registration_data") or {}
    org = ev.get("organization") or {}

    # цены
    tt_prices = []
    for tt in ttypes:
        p = tt.get("price")
        if p is not None:
            try:
                tt_prices.append(float(p))
            except Exception:
                pass

    reg_min = reg.get("price_min")
    reg_max = reg.get("price_max")

    min_price = None
    max_price = None
    if reg_min is not None:
        try: min_price = float(reg_min)
        except: min_price = None
    if reg_max is not None:
        try: max_price = float(reg_max)
        except: max_price = None

    if min_price is None and tt_prices:
        min_price = min(tt_prices)
    if max_price is None and tt_prices:
        max_price = max(tt_prices)

    if min_price is None: min_price = 0.0
    if max_price is None: max_price = 0.0

    is_free = int(max_price == 0.0)

    # участники: attended > sold > tickets_total
    attended_vals = []
    sold_vals = []
    for tt in ttypes:
        a = tt.get("attended")
        s = tt.get("sold")
        if a is not None:
            try: attended_vals.append(float(a))
            except: pass
        if s is not None:
            try: sold_vals.append(float(s))
            except: pass

    attended_sum = float(sum(attended_vals)) if attended_vals else None
    sold_sum = float(sum(sold_vals)) if sold_vals else None
    tickets_total = reg.get("tickets_total")
    try:
        tickets_total = float(tickets_total) if tickets_total is not None else None
    except Exception:
        tickets_total = None

    participants = attended_sum if attended_sum is not None else (sold_sum if sold_sum is not None else tickets_total)

    # вместимость
    tickets_limit = ev.get("tickets_limit")
    if tickets_limit is None and reg:
        tickets_limit = reg.get("tickets_limit")
    try:
        tickets_limit = float(tickets_limit) if tickets_limit is not None else None
    except Exception:
        tickets_limit = None

    # длительность
    duration_hours = None
    if starts_dt and ends_dt:
        try:
            duration_hours = (ends_dt - starts_dt).total_seconds() / 3600.0
        except Exception:
            duration_hours = None

    # лаг создания -> старт (если created_at есть)
    created_dt = parse_dt_safe(ev.get("created_at"))
    lead_days = None
    if created_dt and starts_dt:
        try:
            lead_days = (starts_dt - created_dt).total_seconds() / 86400.0
        except Exception:
            lead_days = None

    # категория
    cat_main = cats[0] if cats else {}
    cat_id = cat_main.get("id")
    cat_name = cat_main.get("name")

    # календарные признаки
    dt_date = starts_dt.date()
    weekday = dt_date.weekday()
    month = dt_date.month
    hour = starts_dt.hour if starts_dt else None
    is_weekend = int(weekday >= 5)

    row = {
        "event_id": ev.get("id"),
        "name": ev.get("name"),
        "url": ev.get("url"),
        "created_at": ev.get("created_at"),
        "starts_at": starts_raw,
        "ends_at": ev.get("ends_at"),

        "year": year,
        "weekday": weekday,
        "month": month,
        "season": season(month),
        "hour": hour,
        "is_weekend": is_weekend,

        "city": loc.get("city"),
        "address": loc.get("address"),

        "category_id": cat_id,
        "category": cat_name,

        "age_limit": ev.get("age_limit"),
        "properties": ",".join(ev.get("properties", []) or []),

        "moderation_status": ev.get("moderation_status"),
        "access_status": ev.get("access_status"),

        "org_id": org.get("id"),
        "org_name": org.get("name"),

        "tickets_limit": tickets_limit,
        "tickets_total": tickets_total,
        "is_registration_open": reg.get("is_registration_open") if reg else None,

        "min_price": min_price,
        "max_price": max_price,
        "is_free": is_free,

        "sold": sold_sum,
        "attended": attended_sum,
        "participants": participants,

        "duration_hours": duration_hours,
        "lead_days": lead_days,
    }
    return row

def load_existing_event_ids(path: str) -> set:
    if not os.path.exists(path):
        return set()
    try:
        s = pd.read_csv(path, usecols=["event_id"])
        ids = set(s["event_id"].dropna().astype(int).tolist())
        log.info(f"resume: loaded existing event_ids={len(ids)} from {path}")
        return ids
    except Exception as e:
        log.warning(f"resume: failed to read {path}: {e}. Will start fresh.")
        return set()

def append_rows_csv(path: str, rows: list, columns: list):
    if not rows:
        return
    df = pd.DataFrame(rows).reindex(columns=columns)
    write_header = not os.path.exists(path)
    df.to_csv(path, mode="a", header=write_header, index=False, encoding="utf-8-sig")
    log.info(f"saved chunk -> {path} | +{len(df)} rows")

def stream_events_conducted(year_from: int, year_to: int, flush_every: int = 200):
    columns = [
        "event_id","name","url","created_at","starts_at","ends_at",
        "year","weekday","month","season","hour","is_weekend",
        "city","address","category_id","category","age_limit","properties",
        "moderation_status","access_status","org_id","org_name",
        "tickets_limit","tickets_total","is_registration_open",
        "min_price","max_price","is_free","sold","attended","participants",
        "duration_hours","lead_days"
    ]

    existing_ids = load_existing_event_ids(OUT_RAW)
    buf_raw = []
    buf_ml = []

    now_utc = datetime.now(timezone.utc)

    processed = 0
    saved_raw = 0
    saved_ml = 0
    dup_skipped = 0
    not_conducted = 0

    for y in range(year_from, year_to + 1):
        start_min = f"{y}-01-01T00:00:00"
        start_max = f"{y}-12-31T23:59:59"

        ev_iter = iter_events(starts_at_min=start_min, starts_at_max=start_max, limit=100)

        for ev in ev_iter:
            processed += 1
            ev_id = ev.get("id")
            if ev_id is None:
                continue
            try:
                ev_id_int = int(ev_id)
            except Exception:
                continue

            if ev_id_int in existing_ids:
                dup_skipped += 1
                continue

            row = normalize_one_event(ev, year=y, now_utc=now_utc)
            if row is None:
                # либо без даты, либо не проведено
                not_conducted += 1
                continue

            existing_ids.add(ev_id_int)
            buf_raw.append(row)
            saved_raw += 1

            if row.get("participants") is not None:
                buf_ml.append(row)
                saved_ml += 1

            if saved_raw % flush_every == 0:
                append_rows_csv(OUT_RAW, buf_raw, columns)
                append_rows_csv(OUT_ML, buf_ml, columns)
                buf_raw.clear()
                buf_ml.clear()
                log.info(f"progress | processed={processed} raw_saved={saved_raw} ml_saved={saved_ml} dup={dup_skipped} not_conducted={not_conducted}")

        append_rows_csv(OUT_RAW, buf_raw, columns)
        append_rows_csv(OUT_ML, buf_ml, columns)
        buf_raw.clear()
        buf_ml.clear()
        log.info(f"year done {y} | processed={processed} raw_saved={saved_raw} ml_saved={saved_ml} dup={dup_skipped} not_conducted={not_conducted}")

    log.info(f"done | processed={processed} raw_saved={saved_raw} ml_saved={saved_ml} dup={dup_skipped} not_conducted={not_conducted}")

if __name__ == "__main__":
    stream_events_conducted(year_from=2019, year_to=2025, flush_every=200)

    if os.path.exists(OUT_ML):
        df_ml = pd.read_csv(OUT_ML)
        print("ML rows:", len(df_ml))
        print("participants non-null:", int(df_ml["participants"].notna().sum()))
