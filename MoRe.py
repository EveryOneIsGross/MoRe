"""
0.2v
"""

import argparse
import json
import jsonlines
from openai import OpenAI
from typing import List, Dict, Any
import os
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import sys
from colorama import init, Fore, Style

# Initialize colorama
init(autoreset=True)

def load_config(config_file: str) -> Dict:
    with open(config_file, 'r') as f:
        return json.load(f)

def load_jsonl(input_file: str) -> List[Dict]:
    with jsonlines.open(input_file) as reader:
        return list(reader)

def save_jsonl(output_file: str, data: List[Dict], log_data: Dict[str, Any]):
    with jsonlines.open(output_file, mode='w') as writer:
        for item, item_log in zip(data, log_data["items"]):
            # Use the overall_relevance_score from item_log
            avg_score = item_log.get("overall_relevance_score")
            
            if avg_score is not None:
                # Format the score as a float with two decimal places
                formatted_score = float(f"{avg_score:.2f}")
                
                # Update the nested 'weight' field if it exists, or add it if it doesn't
                if 'conversations' in item and isinstance(item['conversations'], list):
                    for conversation in item['conversations']:
                        if 'weight' in conversation:
                            conversation['weight'] = formatted_score
                            break
                    else:
                        # If no conversation had a 'weight' key, add it to the first conversation
                        if item['conversations']:
                            item['conversations'][0]['weight'] = formatted_score
                elif 'weight' in item:
                    # If 'weight' is at the top level, update it
                    item['weight'] = formatted_score
                else:
                    # If 'weight' doesn't exist anywhere, add it at the top level
                    item['weight'] = formatted_score
            
            # Write the (potentially updated) item to the JSONL file
            writer.write(item)

def save_log(log_data: Dict[str, Any], output_file: str):
    with open(output_file, 'w') as f:
        json.dump(log_data, f, indent=2)

def concatenate_list(lst, separator=" "):
    """Concatenate a list of items into a single string."""
    return separator.join(str(item) for item in lst if item is not None)

def get_nested_value(obj: Any, path: str, concatenate: bool = False) -> Any:
    """Retrieve a value from a nested dictionary using a dot-separated path."""
    keys = path.replace('[', '.').replace(']', '').split('.')
    for key in keys:
        if isinstance(obj, dict):
            obj = obj.get(key, {})
        elif isinstance(obj, list):
            try:
                obj = obj[int(key)]
            except (ValueError, IndexError):
                return None
        else:
            return None
    
    if concatenate and isinstance(obj, list):
        return concatenate_list(obj)
    return obj if obj != {} else None

def safe_string_conversion(value: Any) -> str:
    """Safely convert a value to a string, handling various types."""
    if isinstance(value, (str, int, float, bool)):
        return str(value)
    elif isinstance(value, (list, dict)):
        return json.dumps(value)
    elif value is None:
        return ""
    else:
        return str(type(value))

def compare_fields(field1: Any, field2: Any) -> tuple:
    """Prepare fields for comparison, converting to strings as necessary."""
    return safe_string_conversion(field1), safe_string_conversion(field2)

def analyze_jsonl_structure(data: Dict, max_depth: int = 10) -> Dict[str, Any]:
    def explore(obj, prefix='', depth=0):
        if depth > max_depth:
            return {'type': type(obj).__name__, 'depth_limit_reached': True}

        if isinstance(obj, dict):
            return {
                'type': 'dict',
                'keys': {
                    key: explore(value, f"{prefix}.{key}" if prefix else key, depth + 1)
                    for key, value in obj.items()
                }
            }
        elif isinstance(obj, list):
            if not obj:
                return {'type': 'list', 'empty': True}
            return {
                'type': 'list',
                'sample': [explore(item, f"{prefix}[{i}]", depth + 1) for i, item in enumerate(obj[:5])]
            }
        else:
            return {'type': type(obj).__name__, 'value': str(obj)[:50] + '...' if len(str(obj)) > 50 else str(obj)}

    return explore(data)

def highlight_terms(text: str, terms: List[str]) -> str:
    """Highlight specified terms in the text."""
    for term in terms:
        text = text.replace(term, f"{Fore.YELLOW}{term}{Style.RESET_ALL}")
    return text

def print_jsonl_guide(structure: Dict[str, Any], highlight_terms_list: List[str]):
    def print_structure(struct, path='', indent=''):
        if isinstance(struct, dict):
            if 'type' in struct:
                if struct['type'] == 'dict':
                    print(f"{indent}{Fore.CYAN}{path}{Style.RESET_ALL}: {Fore.GREEN}dict{Style.RESET_ALL}")
                    if 'keys' in struct:
                        for key, value in struct['keys'].items():
                            print_structure(value, f"{path}.{key}" if path else key, indent + "  ")
                elif struct['type'] == 'list':
                    print(f"{indent}{Fore.CYAN}{path}{Style.RESET_ALL}: {Fore.GREEN}list{Style.RESET_ALL}")
                    if 'sample' in struct:
                        for i, item in enumerate(struct['sample']):
                            print_structure(item, f"{path}[{i}]", indent + "  ")
                    elif 'empty' in struct and struct['empty']:
                        print(f"{indent}  {Fore.YELLOW}(empty list){Style.RESET_ALL}")
                else:
                    value = struct.get('value', 'N/A')
                    highlighted_value = highlight_terms(str(value), highlight_terms_list)
                    print(f"{indent}{Fore.CYAN}{path}{Style.RESET_ALL}: {Fore.MAGENTA}{struct['type']}{Style.RESET_ALL} = {highlighted_value}")
            elif 'depth_limit_reached' in struct:
                print(f"{indent}{Fore.CYAN}{path}{Style.RESET_ALL}: {Fore.RED}(max depth reached){Style.RESET_ALL}")
        else:
            print(f"{indent}{Fore.CYAN}{path}{Style.RESET_ALL}: {Fore.MAGENTA}{type(struct).__name__}{Style.RESET_ALL}")

    print(f"\n{Fore.YELLOW}JSONL Structure Guide:{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}----------------------{Style.RESET_ALL}")
    print("Here's the structure of the first item in your JSONL file:")
    print_structure(structure)

    print(f"\n{Fore.GREEN}Example usage in config.json:{Style.RESET_ALL}")
    print('''
{
  "comparisons": [
    {
      "field1_keys": ["conversations[0].value", "docs.semantic_results[0]"],
      "field2_keys": ["conversations[2].value", "docs.keyword_results[0]"]
    }
  ],
  "highlight_terms": ["important", "key", "relevant"]
}
''')
    print(f"{Fore.YELLOW}Note: Replace the example keys with the actual keys you want to compare.{Style.RESET_ALL}")

def rank_relevance(client: OpenAI, model: str, temp: float, field1: str, field2: str, max_retries: int = 6) -> float:
    def extract_score(data):
        if isinstance(data, dict):
            for key, value in data.items():
                if 'score' in key.lower():
                    return parse_score(value)
                elif isinstance(value, (dict, list)):
                    score = extract_score(value)
                    if score is not None:
                        return score
            # If no 'score' key is found, check if any value is a valid float
            for value in data.values():
                score = parse_score(value)
                if score is not None:
                    return score
        elif isinstance(data, list):
            for item in data:
                score = extract_score(item)
                if score is not None:
                    return score
        return parse_score(data)

    def parse_score(value):
        if isinstance(value, (int, float)):
            return float(value)
        elif isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                return None
        return None

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                temperature=temp,
                messages=[
                {"role": "system", "content": "You are a helpful assistant designed to analyze the relevance between two text fields. Your response must be a JSON object containing ONLY a 'score' key with a float value between 0 and 1, where 1 is highly relevant and 0 is not relevant at all. For example: {\"score\": 0.75}. Do not include any other keys or explanations in your response."},
                {"role": "user", "content": f"Analyze the relevance between the following two fields and provide a score between 0 and 1:\n\nField 1: {field1}\n\nField 2: {field2}\n\nRespond with a JSON object containing only a 'score' key and its float value."}
            ],
                response_format={"type": "json_object"}
            )
            
            content = response.choices[0].message.content
            
            # Try to parse as JSON first
            try:
                result = json.loads(content)
            except json.JSONDecodeError:
                # If JSON parsing fails, try to extract a float from the raw content
                result = content

            if not result:  # Handle empty responses
                raise ValueError("Empty response received")
            
            score = extract_score(result)
            
            if score is not None:
                return min(1.0, max(0.0, score))
            
            if attempt < max_retries - 1:
                print(f"{Fore.YELLOW}Warning: Attempt {attempt + 1} failed. Retrying...{Style.RESET_ALL}")
            else:
                print(f"{Fore.RED}Error: All {max_retries} attempts failed. Last response: {result}{Style.RESET_ALL}")
                return None

        except Exception as e:
            if attempt < max_retries - 1:
                print(f"{Fore.YELLOW}Error on attempt {attempt + 1}: {str(e)}. Retrying...{Style.RESET_ALL}")
            else:
                print(f"{Fore.RED}Error: All {max_retries} attempts failed. Last error: {str(e)}{Style.RESET_ALL}")
                return None

    return None
    
def cosine_sim(text1: str, text2: str) -> float:
    vectorizer = CountVectorizer().fit_transform([text1, text2])
    vectors = vectorizer.toarray()
    similarity = cosine_similarity(vectors)[0][1]
    return min(1.0, max(0.0, similarity))

def truncate_text(text: str, max_length: int = 50) -> str:
    """Truncate text to a maximum length, adding ellipsis if truncated."""
    return (text[:max_length] + '...') if len(text) > max_length else text

def main(input_file: str = None, config_file: str = None):
    config = None
    if config_file:
        try:
            config = load_config(config_file)
            input_file = input_file or config.get('input_file')
        except json.JSONDecodeError as e:
            print(f"{Fore.RED}Error: Unable to read the config file. Please ensure '{config_file}' is a valid JSON file.")
            print(f"Detailed error: {str(e)}{Style.RESET_ALL}")
            sys.exit(1)
    
    if not input_file:
        print(f"{Fore.RED}Error: No input file specified. Please provide an input file either as an argument or in the config file.{Style.RESET_ALL}")
        sys.exit(1)

    try:
        data = load_jsonl(input_file)
    except jsonlines.jsonlines.InvalidLineError as e:
        print(f"{Fore.RED}Error: Unable to read the JSONL file. Please ensure '{input_file}' is a valid JSONL file.")
        print(f"Detailed error: {str(e)}{Style.RESET_ALL}")
        sys.exit(1)

    # Analyze structure and print guide only once, before processing items
    structure = analyze_jsonl_structure(data[0])
    highlight_terms_list = config.get('highlight_terms', []) if config else []
    print_jsonl_guide(structure, highlight_terms_list)

    if config:
        output_file = config.get('output_file', f"{os.path.splitext(input_file)[0]}_ranked.jsonl")
        log_file = f"{os.path.splitext(output_file)[0]}_log.json"
        comparisons = config['comparisons']
        layers = config.get('layers', 1)

        client = OpenAI(
            base_url=config['openai_base_url'],
            api_key=config['openai_api_key'],
        )

        log_data = {
            "config": config,
            "items": []
        }

        for item_index, item in enumerate(data):
            print(f"\n{Fore.CYAN}Processing Item {item_index + 1}:{Style.RESET_ALL}")
            
            item_log = {
                "item_index": item_index,
                "comparisons": []
            }

            for comp_index, comparison in enumerate(comparisons):
                print(f"{Fore.BLUE}Comparison {comp_index + 1}:{Style.RESET_ALL}")
                field1_values = [get_nested_value(item, key, comparison.get('field1_concatenate', False)) 
                                for key in comparison['field1_keys']]
                field2_values = [get_nested_value(item, key, comparison.get('field2_concatenate', False)) 
                                for key in comparison['field2_keys']]
                
                field1 = concatenate_list(field1_values) if comparison.get('field1_concatenate', False) else field1_values[0]
                field2 = concatenate_list(field2_values) if comparison.get('field2_concatenate', False) else field2_values[0]

                field1, field2 = compare_fields(field1, field2)

                if not field1 or not field2:
                    print(f"{Fore.YELLOW}Warning: Could not find specified fields for this comparison.{Style.RESET_ALL}")
                    continue

                print(f"{Fore.GREEN}Field 1:{Style.RESET_ALL} {truncate_text(field1)}")
                print(f"{Fore.GREEN}Field 2:{Style.RESET_ALL} {truncate_text(field2)}")

                comp_log = {
                    "field1": field1,
                    "field2": field2,
                    "models": {}
                }

                llm_scores = []
                for model_config in config['models']:
                    model_name = model_config['name']
                    comp_log["models"][model_name] = {
                        "layers": []
                    }
                    base_temp = model_config['temperature']
                    for layer in range(layers):
                        temp = min(base_temp + (layer * 0.1), 2.0)
                        score = rank_relevance(
                            client,
                            model_name,
                            temp,
                            field1,
                            field2,
                            max_retries=3
                        )
                        if score is not None:
                            llm_scores.append(score)
                            comp_log["models"][model_name]["layers"].append({
                                "layer": layer,
                                "temperature": temp,
                                "score": score
                            })
                            print(f"{Fore.MAGENTA}{model_name}{Style.RESET_ALL} (Layer {layer}, Temp {temp:.2f}): Score = {Fore.YELLOW}{score:.2f}{Style.RESET_ALL}")
                        else:
                            print(f"{Fore.RED}Failed to get a valid score for {model_name} (Layer {layer}, Temp {temp:.2f}){Style.RESET_ALL}")

                try:
                    cosine_score = cosine_sim(field1, field2)
                    print(f"{Fore.BLUE}Cosine Similarity:{Style.RESET_ALL} {cosine_score:.2f}")
                    comp_log["cosine_similarity"] = cosine_score
                except Exception as e:
                    print(f"{Fore.RED}Error calculating NLP score: {str(e)}{Style.RESET_ALL}")
                    cosine_score = None
                
                # Combine all scores
                all_scores = [score for score in llm_scores + [cosine_score] if score is not None]
                
                if all_scores:
                    avg_score = np.mean(all_scores)
                    comp_log["overall_relevance_score"] = avg_score
                else:
                    avg_score = None
                    comp_log["overall_relevance_score"] = None
                
                print(f"{Fore.GREEN}Overall Relevance Score:{Style.RESET_ALL} {avg_score:.2f}" if avg_score is not None else f"{Fore.RED}Overall Relevance Score: N/A{Style.RESET_ALL}")

                # Add the comp_log to item_log["comparisons"]
                item_log["comparisons"].append(comp_log)

            # Calculate the average relevance score for this item
            item_relevance_scores = [comp["overall_relevance_score"] for comp in item_log["comparisons"] if comp["overall_relevance_score"] is not None]
            if item_relevance_scores:
                item_avg_score = np.mean(item_relevance_scores)
                item_log["overall_relevance_score"] = item_avg_score
            else:
                item_log["overall_relevance_score"] = None

            # After processing all comparisons for an item, append item_log to log_data["items"]
            log_data["items"].append(item_log)

        # After processing all items, save the JSONL and log files
        save_jsonl(output_file, data, log_data)
        save_log(log_data, log_file)

        print(f"\n{Fore.GREEN}Processing complete. Output saved to {output_file} and log saved to {log_file}{Style.RESET_ALL}")
    else:
        print(f"{Fore.YELLOW}No config file provided. JSONL structure analysis complete.{Style.RESET_ALL}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze JSONL structure and optionally rank relevance between multiple fields.")
    parser.add_argument("input_file", nargs="?", help="Input JSONL file")
    parser.add_argument("--config", help="JSON config file with settings for ranking")
    
    args = parser.parse_args()
    
    main(args.input_file, args.config)

print(f"\n{Fore.CYAN}Usage instructions:{Style.RESET_ALL}")
print("1. To analyze JSONL structure:")
print("   python script_name.py your_input_file.jsonl")
print("\n2. To analyze and perform ranking:")
print("   python script_name.py your_input_file.jsonl --config your_config.json")
print("   or")
print("   python script_name.py --config your_config.json (if input_file is specified in config)")
