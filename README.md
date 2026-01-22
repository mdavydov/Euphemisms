# Euphemism Analysis Tool

This tool processes XLSX files containing text data and uses Large Language Models (LLMs) to analyze euphemisms in the context of war-related language.

## Features

- **Multiple LLM Support**: OpenAI GPT, DeepSeek, and Google Gemini
- **Configurable Processing Limits**: Control rows per sheet and total rows processed
- **Statistics Mode**: Analyze file structure without making API calls
- **Automatic Output Naming**: Generate result files with experiment numbering
- **Parallel Processing**: Batch processing for efficiency
- **Rate Limiting**: Respects API rate limits for different providers

## Installation

1. Create a virtual environment using `uv`:
```bash
uv venv euph
source euph/bin/activate  # On Linux/Mac
```

2. Install dependencies:
```bash
uv pip install pandas openpyxl requests openai python-dotenv google-generativeai
```

3. Set up API keys in `.env` file:
```env
DEEPSEEK_API_KEY=your_deepseek_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
GEMINI_API_KEY=your_gemini_api_key_here
```

## Usage

### Command Line Options

```bash
python process.py [-h] [-m MAX_ROWS_PER_SHEET] [-n MAX_TOTAL_ROWS] 
                  [-model {deepseek,openai,gemini}] [-stat]
                  [--specific-model SPECIFIC_MODEL] [--output OUTPUT] 
                  [--verbose] input_file
```

### Parameters

- `input_file`: Path to the input XLSX file
- `-m, --max-rows-per-sheet`: Maximum rows per sheet to process (default: 5)
- `-n, --max-total-rows`: Maximum total rows to process across all sheets (default: 50)
- `-model, --model`: Model provider to use (`deepseek`, `openai`, `gemini`)
- `-stat, --statistics-only`: Only show file statistics without calling LLM APIs
- `--specific-model`: Specific model name to use (e.g., `gpt-4`, `deepseek-chat`, `gemini-1.5-pro`)
- `--output`: Custom output filename (auto-generated if not specified)
- `--verbose`: Enable verbose output for debugging

### Examples

#### Show File Statistics
```bash
python process.py -stat PETs_Ukr.xlsx
```

#### Process with OpenAI GPT-4o
```bash
python process.py -model openai -m 10 -n 100 PETs_Ukr.xlsx
```

#### Process with DeepSeek
```bash
python process.py -model deepseek -m 5 -n 50 PETs_Ukr.xlsx
```

#### Process with Gemini (default model)
```bash
python process.py -model gemini -m 8 -n 75 PETs_Ukr.xlsx
```

#### Process with Gemini 3 Pro Preview
```bash
python process.py -model gemini --specific-model gemini-3-pro-preview -m 8 -n 75 PETs_Ukr.xlsx
```

#### Use Custom Model and Output File
```bash
python process.py -model openai --specific-model gpt-4-turbo --output my_results.xlsx -m 3 -n 20 PETs_Ukr.xlsx
```

## Output Format

The tool generates Excel files with the naming convention:
```
Result-{model}-{method}-experiment{number}.xlsx
```

Where:
- `{model}`: Model provider and specific model name
- `{method}`: Processing method (e.g., `m5-n50` for 5 rows per sheet, 50 total)
- `{number}`: Auto-incrementing experiment number

Examples:
- `Result-openai-gpt-4o-m10-n100-experiment1.xlsx`
- `Result-deepseek-deepseek-chat-m5-n50-experiment1.xlsx`
- `Result-gemini-gemini-2-5-flash-lite-m8-n75-experiment1.xlsx`

## File Structure

```
├── process.py              # Main script
├── config.py               # Configuration and constants
├── llm_api.py             # LLM API clients
├── data_processor.py      # Excel file processing
├── statistics.py          # File analysis functionality
├── .env                   # API keys (create this file)
├── README.md              # This file
└── PETs_Ukr.xlsx         # Input data file
```

## Input File Format

The input XLSX file should have the following structure:
- Multiple sheets with different categories
- Each sheet should contain:
  - `text` column: Text to analyze
  - `label` column: Ground truth labels (0 or 1)
  - Other metadata columns (preserved in output)

## Output File Format

The output Excel file contains:
- Original columns from input
- `ai_label`: LLM prediction (0 or 1)
- `ai_reply`: Full LLM response text

## Processing Logic

1. **Text Cleaning**: Removes newlines and normalizes text
2. **Batch Processing**: Groups texts for efficient API calls
3. **Label Extraction**: Extracts binary labels from LLM responses
4. **Rate Limiting**: Adds delays between requests to respect API limits
5. **Error Handling**: Gracefully handles API failures and timeouts

## Vocabulary and Prompts

The tool includes a comprehensive vocabulary of Ukrainian war-related euphemisms and uses specialized prompts to guide the LLM analysis. The vocabulary covers terms like:

- Military terminology euphemisms
- Casualty-related terms
- Weapon and attack euphemisms
- Political and propaganda terms

## Supported Models

### OpenAI
- Default: `gpt-4o`
- API: OpenAI Chat Completions

### DeepSeek
- Default: `deepseek-chat`
- API: DeepSeek Chat Completions (OpenAI-compatible)

### Google Gemini
- Default: `gemini-2.5-flash-lite`
- Other options: `gemini-1.5-pro`, `gemini-3-pro-preview`, etc.
- API: Google Generative AI

## Error Handling

- Invalid API keys: Clear error messages with environment variable names
- Missing columns: Validation of required data structure
- API failures: Fallback to error labels with continued processing
- Rate limits: Built-in delays and retry mechanisms

## Troubleshooting

1. **API Key Issues**: Ensure `.env` file exists and contains valid API keys
2. **Model Not Found**: Check model names and API availability
3. **File Format**: Verify input XLSX has required columns (`text`, `label`)
4. **Rate Limits**: Reduce batch size or increase delays in `llm_api.py`

## Contributing

The code is organized into modules for easy maintenance:
- Add new LLM providers by extending the `LLMClient` class
- Modify processing logic in `DataProcessor` class
- Update vocabulary and prompts in `config.py`
- Add new statistics in `statistics.py`