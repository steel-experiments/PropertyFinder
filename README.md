# Property Finder

A Steel-powered web automation agent that searches any website for properties/listings, with full Raindrop monitoring for observability.

## What It Does

Searches any property listing website and extracts:
- Property/Listing Name
- Location (City/Region)
- Price and Currency
- Rating (out of 5)
- URL to listing

Results are saved to `results_<session_id>.json`.

## Why This Demo?

This showcases realistic web automation challenges:

### Real-World Complexity
- **Any Website**: Works with vacation-rental sites, real-estate sites, booking platforms, and similar listing sites
- **Dynamic Content**: Handles JavaScript rendering via Steel
- **Anti-Bot Resilience**: Steel improves reliability on dynamic/protected sites
- **Intelligent Extraction**: Works across different site structures

### Monitoring Points
Every action is logged to Raindrop:
- Browser initialization timing
- Page load performance
- Listing extraction success/failure
- Data validation
- File save operations

### Signals for Alerts
- `task_success` / `task_failure` - Overall outcome
- `results_found` - Found results
- `no_results` - Empty search
- `slow_fetch` - Performance issues (>8s)
- `fetch_failure` - Content retrieval problems
- `thin_content` - Low content warning

## Setup

### 1. Install Dependencies

```bash
pip install steel-browser raindrop-ai raindrop-query python-dotenv openai
```

### 2. Get API Keys

**Steel**: https://steel.dev
Free tier: 100 browser hours/month

**Raindrop**: https://app.raindrop.ai
Sign up and get your API keys

### 3. Configure Environment

Copy the template and fill your keys:

```bash
cp .env.example .env
```

Then set values in `.env`:

```bash
STEEL_API_KEY=ste_your_steel_key
RAINDROP_WRITE_KEY=your_raindrop_write_key
RAINDROP_QUERY_API_KEY=your_raindrop_query_key
OPENAI_API_KEY=sk_your_openai_key
```

**Note**: Raindrop uses separate keys for writing (`RAINDROP_WRITE_KEY`) and querying (`RAINDROP_QUERY_API_KEY`). Get both from https://app.raindrop.ai

### 4. Run It!

```bash
python PropertyFinder.py --url "https://example-vacation-rental.com/search?location=Porec" --prompt "apartments in Porec for vacation rental"
```

## Usage

### Basic Usage

```bash
python PropertyFinder.py --url <url> --prompt "<what you're looking for>"

# Required Arguments:
#   --url       URL to search (any property listing site)
#   --prompt    Natural language description of what to find

# Optional Arguments:
#   --location    Location parameter for URL templates
#   --max-attempts  Max AI extraction attempts (default: 2)
```

### Examples

```bash
# Vacation rental search
python PropertyFinder.py --url "https://example-vacation-rental.com/search?location=Porec" --prompt "apartments in Porec"

# Real estate search
python PropertyFinder.py --url "https://example-real-estate.com/listings/split" --prompt "flats between 80 and 100 square meters"

# URL templates with placeholders
python PropertyFinder.py --url "https://example-vacation-rental.com/s/{location}/homes" --location "Colorado" --prompt "secluded mountain cabin"
```

### Semantic Search Mode

Search your past runs using natural language:

```bash
# Search past runs by meaning
python PropertyFinder.py --query "mountain cabin finds"

# Find similar discoveries
python PropertyFinder.py --similar "waterfront property with outdoor space"

# Find sessions with issues
python PropertyFinder.py --issues
```

## Example Output

```
PROPERTY FINDER
=====================================================================
URL      : https://example-vacation-rental.com/search?location=Porec
Prompt   : apartments in Porec
Session  : search_20260219_135720
=====================================================================

Steel session: 247f7e14-d470-4aae-b2b4-67f6668b2646
Watch live: https://app.steel.dev/sessions/247f7e14-d470-4aae-b2b4-67f6668b2646
Fetching: https://example-vacation-rental.com/search?location=Porec...
Fetched 1351946 chars in 5.58s
Saved to results_search_20260219_135720.json

=====================================================================
RESULTS
=====================================================================

1. Charming Apartment for Two
   Location: Poreč, Old Town
   Price: $286
   Rating: 5.0/5.0
   URL: /listing/1377529800182448220

2. Studio Apartment Near Beach
   Location: Poreč
   Price: $306
   Rating: 4.78/5.0
   URL: /listing/24710726

3. Room 2min from Beach
   Location: Poreč
   Price: $245
   Rating: 4.84/5.0
   URL: /listing/12921386

=====================================================================
Top result: Charming Apartment for Two
   Poreč, Old Town
=====================================================================

Done! Found 11 results.
See results_search_20260219_135720.json for full data.
```

## Output File

Results are saved to `results_<session_id>.json`:

```json
{
  "session_id": "search_20260219_135720",
  "search_date": "2026-02-19T13:57:20",
  "total": 11,
  "results": [
    {
      "name": "Charming Apartment for Two",
      "location": "Poreč",
      "price": 286,
      "currency": "USD",
      "rating": 5.0,
      "url": "/listing/1377529800182448220"
    }
  ]
}
```

## Expected Results by Site Type

| Site Type | Expected Listings | Notes |
|----------|-------------------|-------|
| Real estate listing sites | 10-15 | Good extraction for property sales |
| Vacation rental platforms | 10-11 | Vacation rentals with prices and ratings |
| Hotel booking platforms | 2-3 | Heavy JS rendering, limited results |
| Protected listing sites | 0 | Anti-bot protection |

## Raindrop Monitoring Dashboard

### Event Timeline

```
[10:30:01] session_started (steel_session_id: abc123)
[10:30:01] page_fetch (url: https://example-site.com/...)
[10:30:03] ai_extraction (count: 11)
[10:30:04] parse_listings (valid_count: 11)
[10:30:04] results_found (count: 11)
[10:30:04] results_saved (filename: results_search_20260219_135720.json)
[10:30:04] session_released
[10:30:04] property_finder_run (results_found: 11)
```

### Signal Alerts

```
results_found (count: 11)
task_success (results_found: 11)
```

### Queries You Can Run in Raindrop

- `event:page_fetch` - See all fetch operations
- `signal:slow_fetch` - Find performance issues
- `results_found > 0` - Successful searches
- `duration_seconds > 10` - Slow executions
- `session_id:search_20260219_135720` - View specific run

## Monitoring Benefits

### Without Raindrop
- Agent fails silently
- Don't know which step broke
- No performance tracking
- Hard to debug

### With Raindrop
- See exact failure point
- Full execution timeline
- Performance metrics
- Search by natural language
- Alert on anomalies
- Compare runs

## Troubleshooting

### No results found
- Try different prompt wording
- Check if the URL is correct
- Use `--issues` to find problematic sessions
- Watch the Steel session URL to see what's happening

### Fewer results than expected
- Some sites have anti-bot protection
- Check Raindrop traces with `--query "extraction listings"`
- Look at `raw_count` vs `valid_count` in traces

### Slow page loads
- Check Raindrop for `slow_fetch` signals
- Some sites can be slow with images loading
- Steel handles this automatically with timeouts

### Browser won't start
- Verify your Steel API key is correct
- Check you have browser hours remaining
- Look for `session_started` event in Raindrop

## Resources

- [Steel Documentation](https://docs.steel.dev)
- [Raindrop Documentation](https://docs.raindrop.ai)
- [Agent Skill Documentation](.agents/skills/propertyfinder/SKILL.md)
