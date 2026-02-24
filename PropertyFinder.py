"""Property Finder using Steel, OpenAI, and Raindrop."""

import os
import re
import json
import html as html_lib
import argparse
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from urllib.parse import urljoin, urlparse, urlunparse, parse_qs

from dotenv import load_dotenv
load_dotenv()

from steel import Steel
import raindrop.analytics as raindrop
from raindrop_query import RaindropQuery
from openai import OpenAI

# Lazy-initialized clients
_query_client = None
_openai_client = None

LISTING_LINK_RE = re.compile(
    r'href=["\']([^"\']*(?:rooms|listing|property|hotel|apartment|home)[^"\']*)["\']',
    re.IGNORECASE,
)
LISTING_PATH_TOKENS = (
    "/hotel/",
    "/room/",
    "/rooms/",
    "/listing/",
    "/property/",
    "/apartment/",
    "/apartments/",
    "/home/",
    "/homes/",
    "/stay/",
    "/villa/",
    "/house/",
)
ASSET_PATH_SUFFIXES = (
    ".css",
    ".js",
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".gif",
    ".svg",
    ".ico",
    ".woff",
    ".woff2",
    ".ttf",
    ".map",
    ".xml",
    ".txt",
    ".json",
    ".pdf",
    ".mp4",
    ".webm",
)
CURRENCY_SYMBOLS = {
    "EUR": "\u20ac",
    "USD": "$",
    "GBP": "\u00a3",
    "HRK": "kn",
}

def require_env(keys: List[str]):
    missing = [key for key in keys if not os.getenv(key)]
    if missing:
        raise ValueError(f"Missing required environment variables: {', '.join(missing)}")


def normalize_url(url: str, base_url: str = "") -> str:
    value = (url or "").strip()
    if not value:
        return ""
    if value.startswith("//"):
        value = "https:" + value
    if base_url:
        value = urljoin(base_url, value)
    if value.startswith("/") and not base_url:
        return ""
    parsed = urlparse(value)
    if not parsed.scheme:
        parsed = urlparse("https://" + value)
    if not parsed.netloc:
        return ""
    return urlunparse((parsed.scheme.lower() or "https", parsed.netloc.lower(), parsed.path, parsed.params, parsed.query, ""))


def parse_number(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    cleaned = re.sub(r"[^\d,.\-]", "", text)
    if not cleaned:
        return None
    if "," in cleaned and "." in cleaned:
        if cleaned.rfind(",") > cleaned.rfind("."):
            cleaned = cleaned.replace(".", "").replace(",", ".")
        else:
            cleaned = cleaned.replace(",", "")
    elif "," in cleaned and "." not in cleaned:
        parts = cleaned.split(",")
        cleaned = "".join(parts[:-1]) + "." + parts[-1] if len(parts[-1]) in (1, 2) else "".join(parts)
    try:
        return float(cleaned)
    except ValueError:
        match = re.search(r"-?\d+(?:\.\d+)?", cleaned)
        return float(match.group(0)) if match else None

def get_query_client() -> RaindropQuery:
    global _query_client
    if _query_client is None:
        api_key = os.getenv("RAINDROP_QUERY_API_KEY")
        if not api_key:
            raise ValueError("RAINDROP_QUERY_API_KEY not set in environment")
        _query_client = RaindropQuery(api_key=api_key)
    return _query_client

def get_openai_client() -> OpenAI:
    global _openai_client
    if _openai_client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set in environment")
        _openai_client = OpenAI(api_key=api_key)
    return _openai_client


class PropertyFinder:

    def __init__(self, max_attempts: int = 2):
        self.session_id = f"search_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.user_id = os.getenv("RAINDROP_USER_ID", "local-cli")
        self.client = None
        self.session = None
        self.max_attempts = max(1, max_attempts)
        self.current_url = ""
        self.source_domain = ""
        self._session_scrape_param = None
        self.results: List[Dict] = []

    def _track(self, event: str, input_text: str, output_text: str, props: dict = None):
        raindrop.track_ai(
            user_id=self.user_id,
            event=event,
            input=input_text,
            output=output_text,
            properties={"session_id": self.session_id, **(props or {})},
        )

    def _signal(self, event_id: str, name: str, sentiment: str = "POSITIVE", props: dict = None):
        raindrop.track_signal(
            event_id=event_id,
            name=name,
            sentiment=sentiment,
            properties=props or {},
        )

    def _get_steel_client(self) -> Steel:
        if self.client is None:
            api_key = os.getenv("STEEL_API_KEY")
            if not api_key:
                raise ValueError("STEEL_API_KEY not set in environment")
            self.client = Steel(steel_api_key=api_key)
        return self.client

    def _scrape_html(self, url: str, delay: int = 3000):
        client = self._get_steel_client()
        final_url = normalize_url(url, self.current_url)
        if not final_url:
            raise ValueError(f"Invalid URL: {url}")
        kwargs = {"url": final_url, "format": ["html"], "delay": delay}
        if self.session:
            session_keys = ["session_id", "session"]
            if self._session_scrape_param:
                session_keys = [self._session_scrape_param] + [k for k in session_keys if k != self._session_scrape_param]
            for key in session_keys:
                try:
                    result = client.scrape(**kwargs, **{key: self.session.id})
                    self._session_scrape_param = key
                    return result
                except TypeError:
                    continue
                except Exception as exc:
                    msg = str(exc).lower()
                    if "unexpected keyword" in msg or "unexpected argument" in msg:
                        continue
                    raise
        return client.scrape(**kwargs)

    def start_session(self):
        t0 = datetime.now()
        self.session = self._get_steel_client().sessions.create()
        duration = (datetime.now() - t0).total_seconds()

        self._track(
            event="session_started",
            input_text="Create Steel browser session",
            output_text=f"Session {self.session.id} ready in {duration:.2f}s",
            props={
                "steel_session_id": self.session.id,
                "viewer_url": self.session.session_viewer_url,
                "duration_seconds": duration,
            },
        )
        print(f"Steel session: {self.session.id}")
        print(f"Watch live: {self.session.session_viewer_url}")

    def end_session(self):
        if self.session:
            self._get_steel_client().sessions.release(self.session.id)
            self._track(
                event="session_released",
                input_text=f"Release session {self.session.id}",
                output_text="Session released",
            )
            print("Session released")

    def scrape_url(self, url: str) -> str:
        interaction = raindrop.begin(
            user_id=self.user_id,
            event="page_fetch",
            input=f"Fetch URL: {url}",
            properties={"url": url},
        )

        try:
            print(f"Fetching: {url}")
            t0 = datetime.now()

            result = self._scrape_html(
                url=url,
                delay=3000,
            )
            content = result.content.html or ""

            duration = (datetime.now() - t0).total_seconds()

            print(f"Fetched {len(content)} chars in {duration:.2f}s")

            if duration > 8:
                self._signal(interaction.id, "slow_fetch", "NEGATIVE",
                             {"duration_seconds": duration})

            if len(content) < 500:
                self._signal(interaction.id, "thin_content", "NEGATIVE",
                             {"content_length": len(content)})

            interaction.set_properties({"duration_seconds": duration, "content_length": len(content)})
            interaction.finish(output=f"Fetched {len(content)} chars in {duration:.2f}s")

            return content

        except Exception as e:
            interaction.set_properties({"error": str(e)})
            interaction.finish(output=f"Fetch failed: {e}")
            self._signal(interaction.id, "fetch_failure", "NEGATIVE", {"error": str(e)})
            raise

    def parse_listings(self, html: str, user_prompt: str = None) -> List[Dict[str, Any]]:
        interaction = raindrop.begin(
            user_id=self.user_id,
            event="parse_listings",
            input=f"Parse listings from {len(html)} chars of HTML",
        )
        try:
            listings = self._extract_listings_with_fallbacks(html, user_prompt)
            valid = self._validate_and_dedupe(listings)
            self.results = valid

            sentiment = "POSITIVE" if valid else "NEGATIVE"
            signal = "results_found" if valid else "no_results"
            self._signal(interaction.id, signal, sentiment, {"count": len(valid)})

            interaction.finish(
                output=f"Parsed {len(valid)} valid listings",
                properties={"valid_count": len(valid), "raw_count": len(listings)},
            )

            return valid

        except Exception as e:
            interaction.finish(output=f"Parse failed: {e}")
            self._signal(interaction.id, "parse_failure", "NEGATIVE", {"error": str(e)})
            raise

    def _extract_listings_with_fallbacks(self, html: str, user_prompt: Optional[str]) -> List[Dict]:
        listings: List[Dict] = []

        if user_prompt:
            listings = self._extract_with_ai(html, user_prompt)
            if listings:
                self._track(
                    event="ai_parse_success",
                    input_text=f"AI extraction for: {user_prompt}",
                    output_text=f"{len(listings)} listings from AI",
                    props={"count": len(listings), "method": "ai"},
                )

        if len(listings) < 3:
            structured = self._parse_structured_scripts(html)
            if structured:
                self._track(
                    event="json_parse_success",
                    input_text="Parse structured scripts from HTML",
                    output_text=f"{len(structured)} listings from structured scripts",
                    props={"count": len(structured), "method": "structured_scripts"},
                )
                listings.extend(structured)

        if not listings:
            self._track(
                event="json_parse_empty",
                input_text="JSON-LD extraction",
                output_text="No listings found, trying regex fallback",
            )
            listings = self._regex_parse(html)

        return listings

    def _parse_structured_scripts(self, html: str) -> List[Dict]:
        script_matches = re.findall(
            r'<script[^>]+type="application/(ld\+json|json)"[^>]*>(.*?)</script>',
            html,
            re.DOTALL | re.IGNORECASE,
        )
        extracted_listings: List[Dict] = []
        for script_type, match in script_matches:
            data = self._parse_json_value(match)
            if data is None:
                continue

            if script_type.lower() == "ld+json":
                extracted = self._extract_from_json(data, max_depth=8, max_nodes=2500)
            else:
                lower = match.lower()
                if not any(token in lower for token in ["listing", "property", "itemlistelement", "offers", "price"]):
                    continue
                extracted = self._extract_from_json(data, max_depth=7, max_nodes=1200)
            extracted_listings.extend(extracted)
        return extracted_listings

    def _validate_and_dedupe(self, listings: List[Dict]) -> List[Dict]:
        valid: List[Dict] = []
        seen_keys = set()
        for raw_listing in listings:
            listing = self._normalize_listing(raw_listing)
            key = self._dedupe_key(listing)
            if key in seen_keys or not self._is_valid(listing):
                continue
            valid.append(listing)
            seen_keys.add(key)
        return valid

    def _prepare_content(self, html: str, max_chars: int = 180000) -> str:
        clean = re.sub(r'<(script|style|noscript)[^>]*>.*?</\1>', '', html, flags=re.DOTALL | re.IGNORECASE)
        clean = re.sub(r'<!--.*?-->', '', clean, flags=re.DOTALL)
        clean = re.sub(r'<svg[^>]*>.*?</svg>', '', clean, flags=re.DOTALL | re.IGNORECASE)
        return clean[:max_chars]

    def _extract_candidate_blocks(self, html: str, limit: int = 120) -> List[str]:
        blocks = []
        for match in LISTING_LINK_RE.finditer(html):
            url = normalize_url(html_lib.unescape(match.group(1)), self.current_url)
            if not url or not self._looks_like_listing_url(url):
                continue
            start = max(0, match.start() - 1400)
            end = min(len(html), match.end() + 2200)
            block = html[start:end]
            if not re.search(r"(?:\$|€|£|\bUSD\b|\bEUR\b|\bGBP\b|\bHRK\b|\bnight\b|\bper\s+night\b|\bprice\b)", block, re.IGNORECASE):
                continue
            blocks.append(block)
            if len(blocks) >= limit:
                break
        return list(dict.fromkeys(blocks))

    def _parse_json_value(self, raw: str) -> Any:
        if not raw:
            return None
        try:
            return json.loads(raw)
        except Exception:
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if not match:
                return None
            try:
                return json.loads(match.group(0))
            except Exception:
                return None

    def _parse_ai_payload(self, raw: str) -> List[Dict]:
        data = self._parse_json_value(raw)
        if data is None:
            return []
        if isinstance(data, dict):
            if isinstance(data.get("listings"), list):
                return data["listings"]
            if isinstance(data.get("results"), list):
                return data["results"]
            if isinstance(data.get("items"), list):
                return data["items"]
            return []
        return data if isinstance(data, list) else []

    def _canonicalize_url(self, raw_url: Any, base_url: str = "") -> Optional[str]:
        candidate = html_lib.unescape(str(raw_url or ""))
        if not candidate:
            return None
        url = normalize_url(candidate, base_url or self.current_url)
        if not url:
            return None
        parsed = urlparse(url)
        path = re.sub(r"/+$", "", parsed.path or "") or "/"
        return urlunparse((parsed.scheme.lower(), parsed.netloc.lower(), path, "", "", ""))

    def _is_generic_name(self, name: str) -> bool:
        value = (name or "").strip().lower()
        return bool(re.match(r"^(listing|property)\s+\d+$", value)) or value in {"unknown", "unknown property", "listing"}

    def _looks_like_listing_url(self, url: str) -> bool:
        if not url:
            return False
        parsed = urlparse(url)
        host = (parsed.netloc or "").lower()
        path = (parsed.path or "").lower()
        if not host:
            return False
        if self.source_domain and not host.endswith(self.source_domain):
            return False
        if path.endswith(ASSET_PATH_SUFFIXES):
            return False
        return any(token in path for token in LISTING_PATH_TOKENS)

    def _quality(self, listings: List[Dict]) -> Dict[str, float]:
        total = len(listings)
        if total == 0:
            return {"count": 0, "url_ratio": 0.0, "price_ratio": 0.0, "non_generic_ratio": 0.0, "good": False}
        with_url = sum(1 for x in listings if x.get("url") and self._looks_like_listing_url(x.get("url")))
        with_price = sum(1 for x in listings if x.get("price") is not None)
        non_generic = sum(1 for x in listings if not self._is_generic_name(x.get("name", "")))
        url_ratio = with_url / total
        price_ratio = with_price / total
        non_generic_ratio = non_generic / total
        return {"count": total, "url_ratio": round(url_ratio, 3), "price_ratio": round(price_ratio, 3), "non_generic_ratio": round(non_generic_ratio, 3), "good": total >= 5 and url_ratio >= 0.6 and non_generic_ratio >= 0.6 and price_ratio >= 0.2}

    def _query_events(self, query: str, search_in: str = "assistant_output", limit: int = 10, silent: bool = False):
        try:
            client = get_query_client()
            return client.events.search(
                query=query,
                mode="semantic",
                search_in=search_in,
                limit=limit,
            )
        except Exception as e:
            if not silent:
                print(f"Query failed: {e}")
            return None

    def _telemetry_hints(self, user_prompt: str) -> str:
        results = self._query_events(
            query=f"{self.source_domain} listing extraction url price rating {user_prompt}",
            search_in="assistant_output",
            limit=5,
            silent=True,
        )
        if not results:
            return ""
        items = results.data if hasattr(results, "data") else (results if isinstance(results, list) else [])
        hints = []
        for item in items:
            out = (getattr(item, "assistant_output", "") or "").replace("\n", " ").strip()
            if out:
                hints.append(out[:140])
        return "\n".join(hints[:4])

    def _build_ai_chunks(self, html: str) -> List[tuple]:
        cleaned = self._prepare_content(html)
        card_blocks = self._extract_candidate_blocks(html)
        script_blocks = re.findall(
            r'<script[^>]+type="application/(?:ld\+json|json)"[^>]*>(.*?)</script>',
            html,
            re.DOTALL | re.IGNORECASE,
        )
        script_blob = "\n\n".join(
            x[:8000] for x in script_blocks if any(t in x.lower() for t in ["listing", "property", "price", "offers"])
        )[:60000]
        chunks = []
        if card_blocks:
            chunks.append(("candidate_cards", "\n\n".join(card_blocks)[:100000]))
        if script_blob:
            chunks.append(("structured_json", script_blob))
        clean_chunks = [cleaned[:90000], cleaned[90000:180000]]
        chunks.extend((f"clean_html_{i+1}", chunk) for i, chunk in enumerate(clean_chunks) if chunk)
        if not chunks:
            chunks = [("raw_html", html[:90000])]
        return chunks

    def _build_ai_instructions(self, telemetry_hints: str, feedback: str) -> str:
        instructions = (
            "Extract property/accommodation listings as strict JSON object: "
            "{\"listings\":[{\"name\":\"\",\"location\":null,\"price\":null,\"currency\":null,\"url\":null,\"rating\":null,\"description\":null}]}. "
            "Extract as many real listing cards/results as possible. Keep price numeric. "
            "Do not return CSS/JS/image/static asset URLs; return only real listing detail URLs. "
            "For each listing, include card-level location and price if visible."
        )
        if telemetry_hints:
            instructions += f"\nPast telemetry hints:\n{telemetry_hints}"
        if feedback:
            instructions += f"\nPrevious attempt feedback:\n{feedback}"
        return instructions

    def _run_ai_iteration(self, client: OpenAI, user_prompt: str, source: str, content: str, instructions: str) -> List[Dict]:
        response = client.responses.create(
            model="gpt-5-mini",
            instructions=instructions,
            input=f"User intent: {user_prompt}\nSource: {source}\n\nContent:\n{content}\n\nReturn JSON only.",
            text={"format": {"type": "json_object"}},
        )
        parsed = self._parse_ai_payload(response.output_text or "")
        attempt_listings = [self._normalize_listing(item) for item in parsed if isinstance(item, dict)]
        return [x for x in attempt_listings if x.get("url") or not self._is_generic_name(x.get("name", ""))]

    def _feedback_from_quality(self, quality: Dict[str, float]) -> str:
        if quality["url_ratio"] < 0.6:
            return "Need more real listing URLs. Exclude static assets and non-listing links."
        if quality["non_generic_ratio"] < 0.6:
            return "Need real listing titles, not generic names."
        return "Need more listing prices."

    def _extract_with_ai(self, html: str, user_prompt: str) -> List[Dict]:
        interaction = raindrop.begin(
            user_id=self.user_id,
            event="ai_extraction",
            input=f"Extract listings using AI for: {user_prompt}",
        )

        try:
            client = get_openai_client()
            chunks = self._build_ai_chunks(html)
            telemetry_hints = self._telemetry_hints(user_prompt)
            combined: List[Dict] = []
            feedback = ""
            quality = self._quality(combined)
            attempts = min(self.max_attempts, len(chunks))

            for i in range(attempts):
                source, content = chunks[i]
                instructions = self._build_ai_instructions(telemetry_hints, feedback)
                attempt_listings = self._run_ai_iteration(client, user_prompt, source, content, instructions)
                combined = self._dedupe_listings(combined + attempt_listings)
                quality = self._quality(combined)

                self._track(
                    event="agentic_iteration",
                    input_text=f"AI attempt {i+1} ({source})",
                    output_text=f"{quality['count']} listings, url_ratio={quality['url_ratio']}, price_ratio={quality['price_ratio']}",
                    props={"attempt": i + 1, "source": source, **quality},
                )

                if quality["good"]:
                    break
                feedback = self._feedback_from_quality(quality)

            if not quality["good"]:
                self._track(
                    event="ai_extraction_low_quality",
                    input_text=f"Extract listings for: {user_prompt}",
                    output_text=f"Rejected AI output with url_ratio={quality['url_ratio']}, price_ratio={quality['price_ratio']}",
                    props={**quality, "method": "agentic_loop"},
                )
                interaction.finish(output=f"AI extraction low quality ({quality['count']} candidates), falling back", properties=quality)
                return []

            self._track(
                event="ai_extraction_success",
                input_text=f"Extract listings for: {user_prompt}",
                output_text=f"{len(combined)} listings extracted via AI loop",
                props={"count": len(combined), "model": "gpt-5-mini", "method": "agentic_loop", "attempts": attempts},
            )
            interaction.finish(output=f"AI extracted {len(combined)} listings", properties={"count": len(combined), "attempts": attempts})
            return combined

        except Exception as e:
            interaction.finish(output=f"AI extraction failed: {e}")
            self._signal(interaction.id, "ai_extraction_failure", "NEGATIVE", {"error": str(e)})
            self._track(event="ai_extraction_failed", input_text=f"Extract listings for: {user_prompt}", output_text=str(e))
            return []

    def _infer_location_from_url(self) -> Optional[str]:
        if not self.current_url:
            return None
        try:
            parsed = urlparse(self.current_url)
            query = parse_qs(parsed.query)
        except Exception:
            return None
        keys = ("location", "city", "q", "query", "where", "destination", "dest", "ss", "search")
        generic = {"hotel", "hotels", "apartment", "apartments", "property", "properties", "room", "rooms", "home", "homes", "rental", "rentals"}
        for key in keys:
            values = query.get(key) or []
            if not values:
                continue
            candidate = re.sub(r"\s+", " ", str(values[0])).strip(" ,-")
            candidate = re.sub(r"[^\w\s,.'-]", " ", candidate)
            candidate = re.sub(r"\s+", " ", candidate).strip(" ,-")
            low = candidate.lower()
            if not candidate or len(low) < 3 or low.isdigit() or low in generic:
                continue
            return candidate
        return None

    def _sanitize_location(self, location: Any) -> Optional[str]:
        if location is None:
            return None
        candidate = re.sub(r"\s+", " ", str(location)).strip(" ,-")
        if not candidate:
            return None
        low = candidate.lower()
        noise = {
            "quiet", "small", "big", "cozy", "cosy", "good", "great", "fast",
            "wifi", "wi fi", "internet", "budget", "cheap", "affordable",
            "family", "luxury", "central", "center", "centre", "near",
        }
        if low in noise:
            return None
        if re.fullmatch(r"\d+", low):
            return None
        return candidate

    def _heal_search_url_with_llm(self, url: str) -> str:
        if not url:
            return url
        parsed = urlparse(url)
        if not parsed.netloc:
            return url
        try:
            client = get_openai_client()
        except Exception:
            return url

        today = datetime.now().date()
        fallback_checkin = (today + timedelta(days=14)).isoformat()
        fallback_checkout = (today + timedelta(days=16)).isoformat()

        prompt = f"""
You are a URL healing assistant for property search websites.
Return a single JSON object only.
Schema:
  {{"healed_url":"...", "applied_parameters":["k=v", ...], "reason":"..."}}

Goal:
- Add missing query params only when required for property search pages to return results.
- Preserve scheme, host, and existing path.
- Do not change the host domain.
- Keep existing query params unless explicitly missing.

Context:
- Original URL: {url}
- Today (for relative date defaults): {today.isoformat()}
- Typical booking-style defaults if needed: checkin={fallback_checkin}, checkout={fallback_checkout}, group_adults=2, no_rooms=1, group_children=0

If no healing is needed or unsure, return the original URL unchanged and an empty applied_parameters list.
"""

        try:
            response = client.responses.create(
                model="gpt-5-mini",
                instructions="Return valid JSON only following the provided schema.",
                input=prompt,
                text={"format": {"type": "json_object"}},
            )
            data = self._parse_json_value(response.output_text or "")
            if not isinstance(data, dict):
                return url
            healed = (data.get("healed_url") or "").strip()
            if not healed:
                return url

            parsed_healed = urlparse(healed)
            if not parsed_healed.scheme or not parsed_healed.netloc:
                return url

            original_host = parsed.netloc.lower().replace("www.", "")
            healed_host = parsed_healed.netloc.lower().replace("www.", "")
            if healed_host == original_host or healed_host.endswith(f".{original_host}"):
                return healed
        except Exception:
            return url

        return url

    def _extract_from_json(self, data, max_depth: int = 8, max_nodes: int = 3000) -> List[Dict]:
        results = []
        state = {"nodes": 0}

        def walk(node: Any, depth: int):
            if depth > max_depth or state["nodes"] >= max_nodes:
                return
            state["nodes"] += 1
            if isinstance(node, dict):
                name = str(node.get("name") or node.get("title") or "").strip()
                price = node.get("price") or node.get("priceString") or node.get("priceValue")
                rating = node.get("rating") or node.get("starRating") or node.get("avgRating")
                location = node.get("location") or node.get("city") or node.get("neighborhood") or node.get("subtitle")
                if isinstance(location, dict):
                    location = location.get("name") or location.get("city") or location.get("addressLocality")
                url = node.get("url") or node.get("link") or node.get("@id")
                if name and (price is not None or rating is not None or url):
                    results.append({
                        "name": name,
                        "location": str(location).strip() if location else None,
                        "price": parse_number(price),
                        "currency": node.get("currency") or node.get("priceCurrency"),
                        "rating": parse_number(rating),
                        "url": url,
                    })
                for value in node.values():
                    walk(value, depth + 1)
            elif isinstance(node, list):
                for item in node[:300]:
                    walk(item, depth + 1)

        walk(data, 0)
        return results

    def _regex_parse(self, html: str) -> List[Dict]:
        listings = []
        seen = set()
        inferred_location = self._infer_location_from_url()
        for match in LISTING_LINK_RE.finditer(html):
            url = normalize_url(html_lib.unescape(match.group(1)), self.current_url)
            if not url or not self._looks_like_listing_url(url):
                continue
            canonical = self._canonicalize_url(url)
            if not canonical:
                continue
            parsed = urlparse(canonical)
            if canonical in seen:
                continue
            seen.add(canonical)
            start = max(0, match.start() - 1200)
            end = min(len(html), match.end() + 2200)
            snippet = html_lib.unescape(re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html[start:end])))
            price_match = re.search(
                r"(?:€|\$|£|USD|EUR|GBP|HRK)\s*([0-9][0-9.,]{1,8})|([0-9][0-9.,]{1,8})\s*(?:€|\$|£|USD|EUR|GBP|HRK)",
                snippet,
                re.IGNORECASE,
            )
            price = parse_number((price_match.group(1) or price_match.group(2)) if price_match else None)
            if price is not None and (price < 20 or price > 10000):
                price = None
            location = inferred_location
            path = parsed.path
            slug = path.strip("/").split("/")[-1].replace(".html", "")
            name = re.sub(r"[-_]+", " ", slug).strip().title() or f"Listing {len(listings) + 1}"
            listings.append({
                "name": name,
                "location": location,
                "price": price,
                "rating": None,
                "url": canonical,
            })
            if len(listings) >= 10:
                break
        self._track(
            event="regex_parse",
            input_text="Regex fallback parse",
            output_text=f"{len(listings)} listings via regex",
            props={"count": len(listings), "urls_found": len(seen)},
        )
        return listings

    def _normalize_listing(self, listing: Dict) -> Dict:
        name = str(listing.get("name") or listing.get("title") or "").strip()
        location = listing.get("location") or listing.get("city") or listing.get("subtitle")
        if isinstance(location, dict):
            location = location.get("name") or location.get("city")
        location = self._sanitize_location(location)
        raw_price = listing.get("price")
        if raw_price is None:
            raw_price = listing.get("price_per_night")
        price = parse_number(raw_price)
        rating = parse_number(listing.get("rating") or listing.get("score"))
        if rating is not None and rating > 5 and rating <= 10:
            rating = rating / 2.0
        if rating is not None and rating < 1:
            rating = None
        currency = listing.get("currency")
        if isinstance(currency, str):
            currency = currency.strip().upper() or None
        elif isinstance(raw_price, str):
            upper = raw_price.upper()
            if "€" in raw_price or "EUR" in upper:
                currency = "EUR"
            elif "$" in raw_price or "USD" in upper:
                currency = "USD"
            elif "£" in raw_price or "GBP" in upper:
                currency = "GBP"
        raw_url = listing.get("url") or listing.get("link")
        url = self._canonicalize_url(raw_url)
        if url and not self._looks_like_listing_url(url):
            url = None
        if (not name or self._is_generic_name(name)) and url:
            slug = urlparse(url).path.strip("/").split("/")[-1].replace(".html", "")
            guessed = re.sub(r"[-_]+", " ", slug).strip().title()
            if guessed:
                name = guessed
        if not location:
            location = self._infer_location_from_url()
        return {
            "name": name,
            "location": str(location).strip() if location else None,
            "price": price,
            "currency": currency,
            "rating": rating,
            "url": url or None,
            "description": listing.get("description"),
        }

    def _dedupe_key(self, listing: Dict):
        if listing.get("url"):
            canonical = self._canonicalize_url(listing["url"])
            if canonical:
                return ("url", canonical)
        return ("fallback", listing.get("name", "").lower(), (listing.get("location") or "").lower(), listing.get("price"))

    def _dedupe_listings(self, listings: List[Dict]) -> List[Dict]:
        seen = set()
        out = []
        for listing in listings:
            key = self._dedupe_key(listing)
            if key in seen:
                continue
            seen.add(key)
            out.append(listing)
        return out

    def _is_valid(self, listing: Dict) -> bool:
        if not listing.get("name"):
            return False
        if listing.get("url") and not self._looks_like_listing_url(listing.get("url")):
            return False
        price = listing.get("price")
        if price is not None and price <= 0:
            return False
        rating = listing.get("rating")
        if rating is not None and not (0 <= rating <= 5):
            return False
        if self._is_generic_name(listing.get("name", "")) and not listing.get("url"):
            return False
        return bool(listing.get("url") or listing.get("location") or price is not None or rating is not None)

    def save(self, filename: str = None):
        filename = filename or f"results_{self.session_id}.json"
        out = {
            "session_id": self.session_id,
            "search_date": datetime.now().isoformat(),
            "total": len(self.results),
            "results": self.results,
        }
        temp_file = f"{filename}.tmp"
        with open(temp_file, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        os.replace(temp_file, filename)
        self._track(event="results_saved", input_text=f"Save {len(self.results)} listings", output_text=f"Written to {filename}", props={"filename": filename, "count": len(self.results)})
        print(f"Saved to {filename}")

    def display(self, results: List[Dict]):
        print("\n" + "=" * 65)
        print("RESULTS")
        print("=" * 65)

        if not results:
            print("No results found.")
            return

        for i, r in enumerate(results, 1):
            price = r.get("price")
            currency = r.get("currency") or "EUR"
            if price is not None:
                symbol = CURRENCY_SYMBOLS.get(currency, currency)
                price_str = f"{symbol}{price:,}"
            else:
                price_str = "Price unknown"

            rating_str = f"{r['rating']}/5.0" if r.get("rating") else ""
            print(f"\n{i}. {r['name']}")
            print(f"   Location: {r.get('location') or 'Unknown'}")
            print(f"   Price: {price_str}")
            if rating_str:
                print(f"   Rating: {rating_str}")
            if r.get("url"):
                print(f"   URL: {r['url']}")

        top = results[0]
        print(f"\n{'=' * 65}")
        print(f"Top result: {top['name']}")
        print(f"   {top.get('location') or 'Unknown'}")
        print("=" * 65)

    def search_past_runs(self, query: str, limit: int = 10):
        print(f"\nSearching past runs for: \"{query}\"")
        return self._query_events(query=query, search_in="assistant_output", limit=limit, silent=False)

    def find_similar(self, description: str, limit: int = 10):
        print(f"\nFinding results similar to: \"{description}\"")
        return self._query_events(query=description, search_in="user_input", limit=limit, silent=False)

    def find_issues(self, limit: int = 10):
        print(f"\nFinding sessions with issues...")
        return self._query_events(
            query="slow fetch failure error timeout problem",
            search_in="assistant_output",
            limit=limit,
            silent=False,
        )

    def display_query_results(self, results, title: str = "Query Results"):
        print("\n" + "=" * 65)
        print(f"{title}")
        print("=" * 65)

        if hasattr(results, 'data'):
            items = results.data
        elif isinstance(results, list):
            items = results
        else:
            items = []

        if not items:
            print("No results found. (Events may take a few minutes to index)")
            return

        for i, r in enumerate(items, 1):
            event_name = getattr(r, 'event_name', 'unknown')
            user_input = getattr(r, 'user_input', '')[:80]
            assistant_output = getattr(r, 'assistant_output', '')[:80]
            timestamp = str(getattr(r, 'timestamp', ''))
            session = getattr(r, 'user_id', 'unknown')
            props = getattr(r, 'properties', {}) or {}
            relevance = getattr(r, 'relevance_score', 0)

            print(f"\n{i}. [{event_name}] {timestamp}")
            print(f"   Session: {session}")
            print(f"   Input: {user_input}")
            print(f"   Output: {assistant_output}")
            if props:
                print(f"   Props: {props}")
            print(f"   Relevance: {relevance:.2f}")

        print("\n" + "=" * 65)
    def run(self, url: str, prompt: str, location: str = None):
        print(f"\nStarting property finder run: {self.session_id}")
        print("Preparing search URL...")
        try:
            final_url = url.format(query=prompt, location=location or "")
        except KeyError:
            final_url = url
        final_url = normalize_url(final_url)
        if not final_url:
            raise ValueError("Invalid final URL")
        final_url = self._heal_search_url_with_llm(final_url)
        self.current_url = final_url
        self.source_domain = urlparse(final_url).netloc.lower().replace("www.", "")

        print("\n" + "=" * 65)
        print("PROPERTY FINDER")
        print("=" * 65)
        print(f"URL      : {final_url}")
        print(f"Prompt   : {prompt}")
        print(f"Session  : {self.session_id}")
        print("=" * 65)

        run_interaction = raindrop.begin(
            user_id=self.user_id,
            event="property_finder_run",
            input=f"Find properties at {final_url}",
            properties={"url": final_url, "prompt": prompt},
        )

        try:
            self.start_session()
            html = self.scrape_url(final_url)
            listings = self.parse_listings(html, user_prompt=prompt)
            self.save()

            top_name = listings[0]["name"] if listings else "none"
            run_interaction.finish(
                output=f"Found {len(listings)} results. Top: {top_name}",
                properties={"results_found": len(listings)},
            )
            self._signal(run_interaction.id, "task_success", "POSITIVE",
                         {"results_found": len(listings)})

            self.display(listings)
            return listings

        except Exception as e:
            run_interaction.finish(output=f"Failed: {e}")
            self._signal(run_interaction.id, "task_failure", "NEGATIVE", {"error": str(e)})
            print(f"\nFailed: {e}")
            raise

        finally:
            self.end_session()
            raindrop.flush()
            print(f"\nRaindrop session: {self.session_id}")
def main():
    parser = argparse.ArgumentParser(
        description="Property Finder - Search any website for properties/listings",
    )
    parser.add_argument("--url", help="URL to search")
    parser.add_argument("--prompt", help="Natural language description of what to find")
    parser.add_argument("--query", help="Search past runs semantically")
    parser.add_argument("--location", help="Location parameter for URL template", default="")
    parser.add_argument("--similar", help="Find similar past discoveries")
    parser.add_argument("--issues", action="store_true", help="Find sessions with problems")
    parser.add_argument("--max-attempts", type=int, default=2, help="Max AI extraction attempts in the agentic loop")

    args = parser.parse_args()

    if args.query and not args.url:
        require_env(["RAINDROP_QUERY_API_KEY"])
        agent = PropertyFinder(max_attempts=args.max_attempts)
        results = agent.search_past_runs(args.query)
        agent.display_query_results(results, f"Semantic Search: \"{args.query}\"")
        return

    if args.similar:
        require_env(["RAINDROP_QUERY_API_KEY"])
        agent = PropertyFinder(max_attempts=args.max_attempts)
        results = agent.find_similar(args.similar)
        agent.display_query_results(results, f"Similar Results: \"{args.similar}\"")
        return

    if args.issues:
        require_env(["RAINDROP_QUERY_API_KEY"])
        agent = PropertyFinder(max_attempts=args.max_attempts)
        results = agent.find_issues()
        agent.display_query_results(results, "Sessions with Issues")
        return

    if args.url and args.prompt:
        require_env(["STEEL_API_KEY", "OPENAI_API_KEY", "RAINDROP_WRITE_KEY", "RAINDROP_QUERY_API_KEY"])
        raindrop.init(os.getenv("RAINDROP_WRITE_KEY"))
        agent = PropertyFinder(max_attempts=args.max_attempts)
        results = agent.run(url=args.url, prompt=args.prompt, location=args.location)
        print(f"\nDone! Found {len(results)} results.")
        print(f"See results_{agent.session_id}.json for full data.")
        return

    parser.print_help()


if __name__ == "__main__":
    main()
