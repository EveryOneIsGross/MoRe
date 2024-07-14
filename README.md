
# MoRe (Mixture 'o Rankers 🔥)

## Overview
MoRe analyzes JSONL (JSON Lines) files, compares specified fields within each JSON object, and calculates relevance scores using a combination of language models and traditional NLP techniques. It's designed to provide a robust, multi-faceted approach to assessing field relevance within structured data.

## Features
- JSONL structure analysis and visualization
- Configurable field comparison
- Multi-model relevance scoring using a diversity of local language models
- Traditional NLP scoring methods (cosine similarity, Jaccard similarity)
- Aggregated scoring to reduce bias and noise
- Detailed logging of all comparisons and scores
- Colorized console output for improved readability

## Requirements
- Python 3.6+
~~- OpenAI API access~~
- ollama
- Required Python packages: 
  - jsonlines
  - openai
  - numpy
  - scikit-learn
  - colorama

## Installation
1. Clone this repository
2. Install required packages:
   ```
   pip install jsonlines openai numpy scikit-learn colorama
   ```
3. Set up your OpenAI API credentials

## Usage
The script can be run in two modes:

1. JSONL Structure Analysis:
   ```
   python script_name.py your_input_file.jsonl
   ```

2. Full Analysis with Field Comparison and Ranking:
   ```
   python script_name.py your_input_file.jsonl --config your_config.json
   ```
   or
   ```
   python script_name.py --config your_config.json
   ```
   (if the input file is specified in the config)

## Configuration
The config.json file should contain:
- Input and output file paths
~~- OpenAI API settings~~
- Model configurations
- Field comparisons to perform
- Optional parameters like temperature layers

Example config.json structure:
```json
{
    "input_file": "qbism_insights.jsonl",
    "output_file": "output_ranked.jsonl",
    "comparisons": [
      {
        "field1_keys": ["conversations[1].value"],
        "field2_keys": ["conversations[2].value"]
      }
    ],
    "highlight_terms": ["QBism", "quantum", "Bayesianism"],
    "openai_base_url": "http://localhost:11434/v1",
    "openai_api_key": "ollama",
    "layers": 20,
    "models": [
      {
        "name": "internlm2:latest",
        "temperature": 0
      },
      {
        "name": "qwen2:latest",
        "temperature": 0
      },
      {
        "name": "llama3:text",
        "temperature": 0
      },
      {
        "name": "mistral:v0.3",
        "temperature": 0
      },
      {
        "name": "PHRASE-2:latest",
        "temperature": 0
      },
      {
        "name": "gemma2:latest",
        "temperature": 0
      }
    ]
}
```

## Output
The script produces two main outputs:
1. A new JSONL file with updated 'weight' fields based on relevance scores
2. A detailed JSON log file containing all comparison data and scores

## How It Works
1. The script reads the JSONL input file and optional configuration.
2. It analyzes and displays the structure of the JSONL data.
3. For each item in the JSONL file, it performs the specified field comparisons.
4. Each comparison involves:
   - Extracting and preparing field values
   - Calculating relevance scores using configured language models
   - Computing traditional NLP similarity scores
   - Aggregating all scores to produce a final relevance score
5. The script updates the 'weight' field of each JSONL item with its average relevance score.
6. Detailed logs are saved, and the updated JSONL is written to the output file.

## Customization
The script is designed to be flexible. You can easily modify the `rank_relevance` function to use different prompts or scoring methods, add new similarity metrics, or adjust the way scores are aggregated.

## Limitations
- Processing large JSONL files may be time-consuming
- The effectiveness of relevance scoring depends on the quality and relevance of the configured language models
