# Rate Limiting Feature

## Overview
The rate limiting feature allows you to control the number of API queries made per minute to avoid hitting API rate limits or to manage costs.

## Usage

Use the `-qpm` or `--queries-per-minute` option to set the maximum number of queries per minute:

```bash
# Limit to 10 queries per minute
python process.py -model gemini -qpm 10 PETs_Ukr.xlsx

# Limit to 60 queries per minute with other options
python process.py -model openai -qpm 60 -m 5 -n 50 PETs_Ukr.xlsx

# No rate limit (default behavior)
python process.py -model deepseek -m 5 -n 50 PETs_Ukr.xlsx
```

## How It Works

1. **Automatic Delay Calculation**: The system automatically calculates the required delay between requests:
   - For example, `-qpm 10` means 60 seconds / 10 queries = 6 seconds between requests
   - For example, `-qpm 60` means 60 seconds / 60 queries = 1 second between requests

2. **Sequential Processing**: When the rate limit is strict (< 10 queries per minute), the system automatically switches to sequential processing instead of parallel to ensure accurate rate limiting.

3. **Per-Request Enforcement**: Each API call waits for the appropriate amount of time since the last request to maintain the specified rate.

## Examples by Provider

### Gemini (Free Tier)
```bash
# Gemini free tier: 15 queries per minute
python process.py -model gemini -qpm 10 -m 5 -n 50 PETs_Ukr.xlsx
```

### OpenAI
```bash
# OpenAI Tier 1: 500 queries per minute (but you might want to be conservative)
python process.py -model openai -qpm 100 -m 10 -n 100 PETs_Ukr.xlsx
```

### DeepSeek
```bash
# DeepSeek rate limits vary by plan
python process.py -model deepseek -qpm 30 -m 5 -n 50 PETs_Ukr.xlsx
```

## Default Behavior

If you don't specify `-qpm`, the system uses default delays:
- **Gemini**: 1 second between requests (60 queries per minute)
- **OpenAI**: 0.1 seconds between requests (600 queries per minute)
- **DeepSeek**: 0.1 seconds between requests (600 queries per minute)

## Benefits

1. **Avoid API Errors**: Prevents hitting rate limit errors from the API providers
2. **Cost Control**: Helps manage API usage costs by limiting throughput
3. **Fair Usage**: Ensures you stay within free tier limits
4. **Predictable Runtime**: Makes execution time more predictable

## Calculating Processing Time

To estimate how long processing will take:

```
Time (minutes) = Total rows / Queries per minute
```

For example:
- 50 rows with `-qpm 10` = ~5 minutes
- 100 rows with `-qpm 60` = ~1.67 minutes
- 200 rows with `-qpm 30` = ~6.67 minutes
