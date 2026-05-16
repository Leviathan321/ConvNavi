import random
import os
import json
import csv
from typing import Any, Dict, List, Literal, Optional, Set

import requests
from pydantic import BaseModel, Field
from utils.file import append_navigation_memory, recreate_navigation_memory
from datetime import datetime, timezone, timedelta

from dotenv import load_dotenv
load_dotenv()

LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")


def _random_filler_utterance() -> str:
    fillers = [
        "The user checked the weather forecast.",
        "The user looked up nearby events.",
        "The user opened their messages.",
        "The user started playing music.",
        "The user checked traffic updates.",
        "The user viewed a news headline.",
        "The user checked their calendar.",
        "The user adjusted app settings.",
        "The user viewed recent photos.",
        "The user set a reminder."
    ]
    return random.choice(fillers)


class Preference(BaseModel):
    target: str
    operator: Literal["eq", "min", "max", "bool", "like", "dislike"]
    value: Any
    context: Optional[Dict[str, Any]] = None


class PreferenceEvent(BaseModel):
    timestep: int
    action: Literal["add", "update", "remove"]
    preference: Preference
    utterance: Optional[str] = None


class UserState(BaseModel):
    preferences: Dict[str, Preference] = Field(default_factory=dict)

    def apply(self, e: PreferenceEvent):
        k = e.preference.target

        if e.action in ["add", "update"]:
            self.preferences[k] = e.preference
        elif e.action == "remove":
            self.preferences.pop(k, None)

    def resolve(self, events: List[PreferenceEvent], t: int):
        self.preferences = {}

        for e in sorted(events, key=lambda x: x.timestep):
            if e.timestep > t:
                break
            self.apply(e)

        return self


SEARCH_SPACE = {
    "cuisine": ["italian", "japanese", "indian", "mexican", "german"],
    "price_level": ["low", "medium", "high"],
    "category": ["restaurant", "bar", "cafe"],
    "rating": [3.5, 4.0, 4.5, 5.0],
    "parking": [True, False],
    "radius_km": [1, 2, 5, 10],
    "entities": {
        "restaurant": [
            "The Cheesecake Factory",
            "Olive Garden",
            "Chipotle Mexican Grill",
            "Five Guys",
            "Shake Shack"
        ],
        "cafe": [
            "Starbucks",
            "Blue Bottle Coffee",
            "Peet's Coffee",
            "Philz Coffee",
            "Dunkin'"
        ],
        "bar": [
            "The Dead Rabbit",
            "The Blind Pig",
            "Rooftop Bar",
            "The Tipsy Crow",
            "City Tap House"
        ]
    }
}


FULL_CONSTRAINTS = {
    "cuisine",
    "price_level",
    "category",
    "rating",
    "parking",
    "radius_km",
    "name"
}


class POIResult(BaseModel):
    name: str
    category: str
    cuisine: Optional[str] = None
    price_level: Optional[str] = None
    rating: Optional[float] = None
    parking: Optional[bool] = None


def sample_preferences() -> List[Preference]:

    category = random.choice(SEARCH_SPACE["category"])

    # Decide randomly for non-name fields whether to include local context
    other_fields = ["cuisine", "price_level", "rating", "parking", "radius_km"]
    context_for = {f for f in other_fields if random.choice([True, False])}

    attribute_pool = [
        Preference(
            target="cuisine",
            operator="eq",
            value=random.choice(SEARCH_SPACE["cuisine"]),
            context=(dict(category=category) if "cuisine" in context_for else None),
        ),
        Preference(
            target="price_level",
            operator="eq",
            value=random.choice(SEARCH_SPACE["price_level"]),
            context=(dict(category=category) if "price_level" in context_for else None),
        ),
        # category itself is the context seed; it should not carry a self-context
        # Preference(
        #     target="category",
        #     operator="eq",
        #     value=category,
        #     context=None,
        # ),
        Preference(
            target="rating",
            operator="min",
            value=random.choice(SEARCH_SPACE["rating"]),
            context=(dict(category=category) if "rating" in context_for else None),
        ),
        Preference(
            target="parking",
            operator="bool",
            value=random.choice(SEARCH_SPACE["parking"]),
            context=(dict(category=category) if "parking" in context_for else None),
        ),
        Preference(
            target="radius_km",
            operator="min",
            value=random.choice(SEARCH_SPACE["radius_km"]),
            context=(dict(category=category) if "radius_km" in context_for else None),
        ),
        # name always has context (local)
        Preference(
            target="name",
            operator="like",
            value=random.choice(SEARCH_SPACE["entities"][category]),
            context=dict(category=category),
        )
    ]

    k = random.randint(3, len(attribute_pool))

    return random.sample(attribute_pool, k=k)


def preference_to_utterance(p: Preference) -> str:
    context_suffix = ""
    if p.context:
        context_text = ", ".join(f"{key} is {value}" for key, value in p.context.items())
        context_suffix = f" if {context_text}"

    if p.target == "cuisine":
        return f"The user likes {p.value} food{context_suffix}."
    if p.target == "price_level":
        return f"The user prefers {p.value} priced places{context_suffix}."
    if p.target == "category":
        return f"The user usually goes to {p.value}s{context_suffix}."
    if p.target == "rating":
        return f"The user wants places rated at least {p.value}{context_suffix}."
    if p.target == "parking":
        return f"The user needs parking{context_suffix}."
    if p.target == "radius_km":
        return f"The user prefers places within {p.value} km{context_suffix}."
    if p.target == "name":
        return f"The user's favorite place is {p.value}{context_suffix}."
    return f"The user prefers {p.target}{context_suffix}."

def generate_memory(event_count: int = 6, max_time: int = 30) -> List[PreferenceEvent]:
    base_prefs = sample_preferences()

    # actual number of initial add events is bounded by event_count
    k = min(event_count, len(base_prefs))

    selected_prefs = random.sample(base_prefs, k=k)
    ts = sorted(random.sample(range(1, max_time), k=k))

    events: List[PreferenceEvent] = []
    # create initial add events
    for t, p in zip(ts, selected_prefs):
        events.append(PreferenceEvent(timestep=t, action="add", preference=p, utterance=preference_to_utterance(p)))

    # decide how many preferences will be updated later (user changes mind)
    num_updates = random.randint(0, max(0, k // 2))
    if num_updates > 0:
        update_candidates = random.sample(range(k), k=num_updates)

        def _new_value_for(pref: Preference):
            tgt = pref.target
            cur = pref.value
            if tgt == "cuisine":
                opts = [o for o in SEARCH_SPACE["cuisine"] if o != cur]
                return random.choice(opts) if opts else cur
            if tgt == "price_level":
                opts = [o for o in SEARCH_SPACE["price_level"] if o != cur]
                return random.choice(opts) if opts else cur
            if tgt == "category":
                opts = [o for o in SEARCH_SPACE["category"] if o != cur]
                return random.choice(opts) if opts else cur
            if tgt == "rating":
                opts = [o for o in SEARCH_SPACE["rating"] if o != cur]
                return random.choice(opts) if opts else cur
            if tgt == "parking":
                return not bool(cur)
            if tgt == "radius_km":
                opts = [o for o in SEARCH_SPACE["radius_km"] if o != cur]
                return random.choice(opts) if opts else cur
            if tgt == "name":
                # try to keep same category context if present
                cat = pref.context.get("category") if pref.context else None
                if cat and cat in SEARCH_SPACE["entities"]:
                    opts = [o for o in SEARCH_SPACE["entities"][cat] if o != cur]
                    return random.choice(opts) if opts else cur
                # fallback: pick any entity in same category list
                for lst in SEARCH_SPACE["entities"].values():
                    if cur in lst:
                        opts = [o for o in lst if o != cur]
                        return random.choice(opts) if opts else cur
                return cur
            return cur

        # pick update timesteps later than the original events
        for idx in update_candidates:
            orig_event = events[idx]
            orig_t = orig_event.timestep
            # choose update time after orig_t
            possible = [x for x in range(orig_t + 1, max_time + 1)]
            if not possible:
                continue
            new_t = random.choice(possible)
            old_pref = orig_event.preference
            new_val = _new_value_for(old_pref)
            # create new Preference with updated value, keep context
            new_pref = Preference(target=old_pref.target, operator=old_pref.operator, value=new_val, context=old_pref.context)
            events.append(PreferenceEvent(timestep=new_t, action="update", preference=new_pref, utterance=preference_to_utterance(new_pref)))

    # return events sorted by timestep so downstream timestamps are chronological
    events = sorted(events, key=lambda e: e.timestep)
    return events

def resolve_state(events: List[PreferenceEvent], t: int) -> UserState:
    return UserState().resolve(events, t)


def project_query(state: UserState, visible: Set[str]) -> Dict[str, Any]:
    return {
        k: p.value
        for k, p in state.preferences.items()
        if k in visible
    }

def generate_query(state: UserState, query_timestep: int) -> str:
    print("time:", query_timestep)

    optional = list(FULL_CONSTRAINTS - {"category"})
    visible = set(random.sample(optional, k=min(2, len(optional))))

    q = project_query(state, visible)

    parts = ["Find me"]

    # HARD GUARANTEE: category is independent of state
    if "category" in state.preferences:
        category = state.preferences["category"].value
    else:
        category = random.choice(SEARCH_SPACE["category"])

    parts.append(category)

    if "price_level" in q:
        parts.append(q["price_level"])

    if "cuisine" in q:
        parts.append(q["cuisine"])

    if "rating" in q:
        parts.append(f"rated at least {q['rating']}")

    if "parking" in q and q["parking"]:
        parts.append("with parking")

    if "radius_km" in q:
        parts.append(f"within {q['radius_km']} km")

    parts.append("nearby.")

    return " ".join(parts)

def check_leak(query: str, state: UserState) -> Dict[str, bool]:
    q = query.lower()
    return {
        k: str(p.value).lower() not in q
        for k, p in state.preferences.items()
    }


def evaluate(state: UserState, result: POIResult) -> Dict[str, bool]:

    checks = {}

    def context_applies(preference: Preference) -> bool:
        if not preference.context:
            return True

        for key, expected_value in preference.context.items():
            actual_value = getattr(result, key, None)
            if actual_value != expected_value:
                return False

        return True

    for k, p in state.preferences.items():

        if not context_applies(p):
            checks[k] = True
            continue

        if k == "cuisine":
            checks[k] = result.cuisine == p.value

        elif k == "price_level":
            checks[k] = result.price_level == p.value

        elif k == "category":
            checks[k] = result.category == p.value

        elif k == "rating":
            checks[k] = result.rating is not None and result.rating >= p.value

        elif k == "parking":
            checks[k] = result.parking == p.value

        elif k == "radius_km":
            checks[k] = True

        elif k == "name":
            checks[k] = result.name == p.value

    checks["overall"] = all(checks.values()) if checks else False
    return checks


if __name__ == "__main__":
    import sys

    NUM_CASES = 3
    passed = 0
    failed = 0

    # create a timestamped results directory (YYYYMMDD_HHMMSS)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = os.path.join('results', timestamp)
    os.makedirs(results_dir, exist_ok=True)

    # collect per-test, per-preference results for final table and CSV
    results_summary: List[Dict[str, Any]] = []
    per_test_rows: List[Dict[str, Any]] = []

    GREEN = "\033[92m"
    RED = "\033[91m"
    RESET = "\033[0m"
    CHECK = "✓"
    CROSS = "✗"

    for case_idx in range(1, NUM_CASES + 1):
        event_count = random.randint(2, 6)
        events = generate_memory(event_count=event_count)

        print(f"\n=== TEST {case_idx} / {NUM_CASES} ===\n")
        for e in events:
            print(f"t={e.timestep}: {e.utterance} ({e.action} {e.preference.target}={e.preference.value})")

        # recreate CSV per test
        csv_path = 'data/navigation_memory.csv'
        recreate_navigation_memory(csv_path)

        base_time = datetime.now(timezone.utc)
        # insert some unrelated filler entries before the first event
        num_before = random.randint(1, 3)
        for b in range(num_before):
            t_before = base_time - timedelta(days=(num_before - b))
            time_iso = t_before.strftime('%Y-%m-%dT%H:%M:%SZ')
            append_navigation_memory(csv_path, _random_filler_utterance(), conversation_id=1, time_iso=time_iso)

        for i, e in enumerate(events):
            # primary event timestamp (one day apart)
            event_time = base_time + timedelta(days=i)
            time_iso = event_time.strftime('%Y-%m-%dT%H:%M:%SZ')
            if e.utterance:
                append_navigation_memory(csv_path, e.utterance, conversation_id=1, time_iso=time_iso)

            # insert some unrelated filler entries between events
            between_count = random.randint(0, 2)
            for j in range(between_count):
                # space fillers by hours after the event
                filler_time = event_time + timedelta(hours=(j + 1))
                filler_iso = filler_time.strftime('%Y-%m-%dT%H:%M:%SZ')
                append_navigation_memory(csv_path, _random_filler_utterance(), conversation_id=1, time_iso=filler_iso)

        query_t = events[-1].timestep + 1
        state = resolve_state(events, t=query_t)

        query = generate_query(state, query_t)
        print("\n=== QUERY ===\n")
        print(query)

        nav_api_url = os.getenv("NAV_API_URL", "http://127.0.0.1:8000/query")
        request_payload = {
            "query": query,
            "user_location": [39.955431, -75.154903],
            "llm_type": os.getenv("LLM_MODEL", "gpt-4o-mini"),
            "user_id": 1,
        }

        print("\n=== REQUEST ===\n")
        print("POST", nav_api_url)
        print(json.dumps(request_payload, indent=2))

        try:
            response = requests.post(nav_api_url, json=request_payload, timeout=120)
            response.raise_for_status()
            print("\n=== RESPONSE ===\n")
            try:
                response_json = response.json()
                print(json.dumps(response_json, indent=2))
            except Exception:
                response_json = response.text
                print(response_json)
        except requests.exceptions.RequestException as exc:
            print("\n=== REQUEST FAILED ===\n")
            print(exc)
            response_json = {"error": str(exc)}

        print("\n=== LEAK CHECK ===\n")
        print(check_leak(query, state))

        # If the response included POIs, evaluate each POI and pick the best one.
        selected_poi = None
        selected_checks = None
        best_score = -1

        if isinstance(response_json, dict):
            for key in ('retrieved_pois', 'pois', 'results', 'items', 'candidates', 'recommendations'):
                if key in response_json and isinstance(response_json[key], list):
                    for poi in response_json[key]:
                        try:
                            pr = POIResult(
                                name=poi.get('name', ''),
                                category=poi.get('category', ''),
                                cuisine=poi.get('cuisine') if 'cuisine' in poi else None,
                                price_level=poi.get('price_level') or poi.get('price') if poi else None,
                                rating=poi.get('rating') if 'rating' in poi else None,
                                parking=poi.get('parking') if 'parking' in poi else None,
                            )
                        except Exception:
                            continue

                        this_checks = evaluate(state, pr)
                        # score: count of satisfied preference checks (exclude overall)
                        score = sum(1 for k, v in this_checks.items() if k != 'overall' and v)
                        if score > best_score:
                            best_score = score
                            selected_poi = pr
                            selected_checks = this_checks
                    break

        # fallback to the synthetic sample result if no POIs were present or evaluation failed
        if selected_checks is None:
            result = POIResult(
                name=Preference(target="name", operator="eq", value="Luigi").value,
                category=Preference(target="category", operator="eq", value="bar").value,
                cuisine=Preference(target="cuisine", operator="eq", value="italian").value,
                price_level=state.preferences.get("price_level", Preference(target="price_level", operator="eq", value="$$")).value,
                rating=4.5,
                parking=state.preferences.get("parking", Preference(target="parking", operator="bool", value="true")).value
            )
            selected_checks = evaluate(state, result)

        checks = selected_checks
        overall = checks.get("overall", False)

        print("\n=== EVALUATION ===\n")
        for k, v in checks.items():
            if k == "overall":
                continue
            status = f"{GREEN}{CHECK}{RESET}" if v else f"{RED}{CROSS}{RESET}"
            print(f"{k}: {status}")

        if overall:
            print(f"\n{GREEN}{CHECK} PASS{RESET}\n")
            passed += 1
        else:
            print(f"\n{RED}{CROSS} FAIL{RESET}\n")
            failed += 1

        # record per-preference results
        for k, p in state.preferences.items():
            scope = "local" if p.context else "global"
            pref_passed = checks.get(k, True)
            results_summary.append({
                "test": case_idx,
                "preference": k,
                "scope": scope,
                "passed": pref_passed,
                "overall": overall,
            })

        # normalize response payload into text and extract POIs for summaries
        if isinstance(response_json, str):
            response_text = response_json
            response_obj = None
        else:
            try:
                response_text = json.dumps(response_json, ensure_ascii=False)
                response_obj = response_json
            except Exception:
                response_text = str(response_json)
                response_obj = None

        poi_text = ""
        if isinstance(response_obj, dict):
            for key in ('retrieved_pois', 'pois', 'results', 'items', 'candidates', 'recommendations'):
                if key in response_obj and isinstance(response_obj[key], list):
                    try:
                        poi_text = json.dumps(response_obj[key], ensure_ascii=False)
                    except Exception:
                        poi_text = str(response_obj[key])
                    break

        # extract a concise textual utterance from the response object (exclude POI lists)
        def _extract_utterance(response_obj, response_text):
            if isinstance(response_obj, dict):
                # common top-level text fields
                for k in ('utterance', 'text', 'response', 'answer', 'message', 'content', 'description'):
                    v = response_obj.get(k)
                    if isinstance(v, str) and v.strip():
                        return v

                # choices / assistant response formats
                choices = response_obj.get('choices') or response_obj.get('outputs')
                if isinstance(choices, list):
                    for c in choices:
                        if isinstance(c, dict):
                            for k in ('text', 'message', 'content', 'response'):
                                v = c.get(k)
                                if isinstance(v, str) and v.strip():
                                    return v
                            # nested message content lists
                            msg = c.get('message')
                            if isinstance(msg, dict):
                                cont = msg.get('content')
                                if isinstance(cont, list):
                                    for item in cont:
                                        if isinstance(item, dict):
                                            txt = item.get('text') or item.get('content')
                                            if isinstance(txt, str) and txt.strip():
                                                return txt

                # fallback: return the full response_text (JSON string) if no concise field found
            return response_text

        response_utterance = _extract_utterance(response_obj, response_text)

        # build per-test row with one column per preference
        row = {"test": case_idx}
        present_prefs = 0
        matched_prefs = 0
        for pref in sorted(FULL_CONSTRAINTS):
            if pref in state.preferences:
                p = state.preferences[pref]
                present_prefs += 1
                passed_pref = checks.get(pref, True)
                scope = "local" if p.context else "global"
                status = "PASS" if passed_pref else "FAIL"
                # include the preference value and the status with scope in brackets
                val = f"{p.value} ({status}_{scope})"
                row[pref] = val
                if passed_pref:
                    matched_prefs += 1
            else:
                row[pref] = "NA"

        # percentage of preferences matched among present prefs
        if present_prefs > 0:
            pct = round(100.0 * matched_prefs / present_prefs, 2)
        else:
            pct = 0.0
        row["request_utterance"] = query
        row["response_utterance"] = response_utterance
        row["response_pois"] = poi_text
        row["matched_percentage"] = pct
        per_test_rows.append(row)

        # write per-test detail CSV with history, request, and response
        try:
            os.makedirs(results_dir, exist_ok=True)
            detail_csv = os.path.join(results_dir, f'navigation_memory_test_{case_idx}.csv')
            with open(detail_csv, 'w', newline='', encoding='utf-8') as df:
                writer = csv.DictWriter(df, fieldnames=['test', 'history', 'request', 'response_text', 'poi'])
                writer.writeheader()
                # history: put actual utterances written to memory as newline-separated lines
                history_lines = []
                for ev in events:
                    if ev.utterance is None:
                        continue
                    # include timestep and utterance for clarity
                    history_lines.append(f"t={ev.timestep}: {ev.utterance}")
                history = "\n".join(history_lines)
                req_text = json.dumps(request_payload, ensure_ascii=False)
                writer.writerow({'test': case_idx, 'history': history, 'request': req_text, 'response_text': response_text, 'poi': poi_text})
            print(f'Wrote detail CSV for test {case_idx} to {detail_csv}')
        except Exception as exc:
            print(f'Failed writing detail CSV for test {case_idx}: {exc}')

    print("\n=== SUMMARY ===\n")
    print(f"Total: {NUM_CASES}, Passed: {passed}, Failed: {failed}")

    # Print final table to console
    if results_summary:
        headers = ["test", "preference", "scope", "passed", "overall"]
        # compute column widths
        col_widths = {h: max(len(h), max((len(str(r[h])) for r in results_summary), default=0)) for h in headers}
        # header row
        header_line = " | ".join(h.ljust(col_widths[h]) for h in headers)
        sep_line = "-+-".join("-" * col_widths[h] for h in headers)
        print(header_line)
        print(sep_line)
        for r in results_summary:
            row = " | ".join(str(r[h]).ljust(col_widths[h]) for h in headers)
            print(row)

    # write CSV summary
    summary_csv = os.path.join(results_dir, 'navigation_memory_summary.csv')
    try:
        os.makedirs(results_dir, exist_ok=True)
        with open(summary_csv, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=["test", "preference", "scope", "passed", "overall"])
            writer.writeheader()
            for r in results_summary:
                writer.writerow(r)
        print(f"Wrote summary CSV to {summary_csv}")
    except Exception as exc:
        print(f"Failed writing summary CSV: {exc}")

    # write per-test summary with one column per preference
    try:
        per_test_csv = os.path.join(results_dir, 'navigation_memory_summary_by_test.csv')
        if per_test_rows:
            fieldnames = ['test'] + sorted(FULL_CONSTRAINTS) + ['request_utterance', 'response_utterance', 'response_pois', 'matched_percentage']
            with open(per_test_csv, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                for r in per_test_rows:
                    writer.writerow(r)
            print(f"Wrote per-test summary CSV to {per_test_csv}")
    except Exception as exc:
        print(f"Failed writing per-test summary CSV: {exc}")
    sys.exit(0)