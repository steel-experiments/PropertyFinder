# AGENTS.md

## Project
`PropertyFinder` is a CLI agent that uses Steel + OpenAI + Raindrop to search property/listing websites, extract listing cards, rank results, and save run outputs.

Main code path:
- `PropertyFinder.py`

## Skill Location
Project-specific skill for property listing automation:
- `.agents/skills/propertyfinder/SKILL.md`

When a task is about property search/extraction flows, read and follow that skill first.
If the skill references relative scripts/assets, resolve them relative to `.agents/skills/propertyfinder/`.

## Environment Setup
Copy `.env.example` to `.env` and fill keys:
- `STEEL_API_KEY`
- `OPENAI_API_KEY`
- `RAINDROP_WRITE_KEY`
- `RAINDROP_QUERY_API_KEY`

## Useful Commands
- Run finder:
`python PropertyFinder.py --url "<search-url>" --prompt "<natural-language-request>"`
- Query past runs:
`python PropertyFinder.py --query "<semantic query>"`
- Find similar:
`python PropertyFinder.py --similar "<description>"`
- Find issues:
`python PropertyFinder.py --issues`

## Engineering Guardrails
- Prefer generalizable logic over site-specific hardcoded branches.
- Keep CLI behavior and output schema stable unless the task explicitly changes them.
- Do not commit generated run artifacts (for example `results_search_*.json`).
