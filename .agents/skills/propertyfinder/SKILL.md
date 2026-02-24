---
name: propertyfinder
description: Search and extract property listings from any website using Steel browser automation with AI-powered extraction. Use when the user wants to find apartments, houses, flats, or properties for sale/rent on property listing websites.
---

# PropertyFinder

A CLI tool that gets property listing websites and extracts listings using AI.

## Quick Start

```bash
python3 PropertyFinder.py --url "<listing_url>" --prompt "<what you're looking for>"
```

**Example:**
```bash
python3 PropertyFinder.py --url "https://example-vacation-rental.com/search?location=Porec" --prompt "apartments in Porec"
```

## Environment Setup

Before using PropertyFinder, ensure `.env` contains:

```env
STEEL_API_KEY=ste-...           # Cloud browser
RAINDROP_WRITE_KEY=...          # Analytics tracking
RAINDROP_QUERY_API_KEY=...      # Semantic search of past runs
OPENAI_API_KEY=sk-...           # AI-powered extraction
```

Install dependencies:
```bash
pip install openai steel raindrop-analytics raindrop-query python-dotenv
```

## CLI Flags

### Search Mode (Primary Usage)

| Flag | Required | Description |
|------|----------|-------------|
| `--url` | Yes | URL to search (any property listing site) |
| `--prompt` | Yes | Natural language description of what to find |
| `--location` | No | Location parameter for URL templates |
| `--keywords` | No | Optional hint keywords (comma-separated) |

**Examples:**
```bash
# Vacation rental search
python3 PropertyFinder.py --url "https://example-vacation-rental.com/search?location=Porec" --prompt "apartments for vacation rental"

# Real estate search
python3 PropertyFinder.py --url "https://example-real-estate.com/listings/split" --prompt "flats between 80 and 100 square meters"

# With custom hint keywords
python3 PropertyFinder.py --url "https://example-real-estate.com/houses/zagreb" --prompt "houses with outdoor space" --keywords "garden,backyard,terrace,patio"
```

### Semantic Search Mode

| Flag | Description |
|------|-------------|
| `--query <text>` | Search past runs semantically |
| `--similar <text>` | Find similar past discoveries |
| `--issues` | Find sessions with problems/failures |

**Examples:**
```bash
# Find past searches about cabins
python3 PropertyFinder.py --query "cabin mountain finds"

# Find sessions that had issues
python3 PropertyFinder.py --issues

# Find similar properties to what you've searched before
python3 PropertyFinder.py --similar "waterfront with outdoor space"
```

## How It Works

1. **Steel** launches a cloud browser, navigates to the URL, and gets rendered HTML
2. **AI** analyzes the HTML and extracts property listings (names, prices, locations, URLs, ratings)
3. **Raindrop** tracks everything for later semantic search

## Output

- **Console**: Extracted results
- **results.json**: Full data in JSON format

Each result contains:
- `name` - Property title
- `location` - City/area
- `price` - Numeric price
- `currency` - EUR, USD, GBP, HRK
- `rating` - 0-5 scale if available
- `url` - Link to listing

## Expected Results by Site Type

| Site Type | Expected Listings | Notes |
|----------|-------------------|-------|
| Real estate listing sites | 10-15 | Good extraction for property sales |
| Vacation rental platforms | 10-11 | Vacation rentals with prices and ratings |
| Hotel booking platforms | 2-3 | Heavy JS rendering, lower quality |
| Protected listing sites | 0 | Anti-bot protection |

## Troubleshooting

### Few or No Results

1. **Check traces**: `python3 PropertyFinder.py --issues`
2. **Search past runs**: `python3 PropertyFinder.py --query "extraction listings"`
3. **Watch the session**: Click the Steel session URL to see retrieval in real-time

### Common Issues

- **"No results found"**: Site may have anti-bot protection or heavy JS rendering
- **Wrong prices**: Some sites load prices dynamically
- **Missing names**: HTML structure may not match expected patterns

### Using Raindrop Traces

When debugging, check these metrics in traces:
- `raw_count` vs `valid_count` - if raw is high but valid is low, validation may be filtering results
- `prices_found` and `urls_found` - indicates HTML quality
- Recent traces take a few minutes to index

## Adding New CLI Flags

To extend the tool:

1. Edit `main()` function's argparse section in `PropertyFinder.py`:
```python
parser.add_argument("--newflag", help="Description of new flag")
```

2. Handle the flag after parsing:
```python
if args.newflag:
    # Handle new flag
```

3. Update `print_usage()` to document it

## Workflow Tips

1. Start with a descriptive `--prompt`
2. Check `results.json` for complete data
3. Use `--issues` to diagnose poor results
4. Use `--similar` to find matching past discoveries
5. Watch the Steel session URL to see retrieval progress
