import os
import pickle
import json
from typing import List, Dict
from gpt4all import GPT4All
from nomic import embed
import numpy as np
from openai import OpenAI
from colorama import init, Fore, Style
import argparse

init(autoreset=True)  # Initialize colorama

class ConversationalDocQAAgent:
    def __init__(self, model_name: str = "orca-mini-3b-gguf2-q4_0.gguf", chunk_size: int = 100, 
                 chunk_overlap: int = 20, history_window: int = 5, 
                 system_prompt_file: str = "system_prompt.md", max_doc_length: int = 512,
                 top_k: int = 3):
        try:
            self.llm = GPT4All(model_name)
        except ValueError as e:
            print(f"{Fore.RED}Error loading model: {e}")
            print(f"{Fore.YELLOW}Please make sure you have a valid model name or path.")
            raise
        self.max_doc_length = max_doc_length
        self.client = OpenAI(base_url='http://localhost:11434/v1', api_key='ollama')
        self.documents: List[Dict] = []
        self.document_source = "Unknown"
        self.embeddings: List[np.ndarray] = []
        self.full_content: str = ""
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k
        self.conversation_history: List[Dict] = []
        self.history_window = history_window
        self.system_prompt_template = self.load_system_prompt(system_prompt_file)
        self.query_expansion_data = []
        self.response_data = []
        self.query_generation_prompt = """
You are an AI assistant tasked with generating a comprehensive query for information retrieval. Your goal is to create a query that will extract the most salient knowledge from a given text while incorporating the user's initial question and the conversation history.

Consider the following:
1. The user's question and its intent
2. The conversation history provided
3. Potential related topics or subtopics that might provide valuable information

Generate a query that addresses these aspects and is likely to retrieve the most relevant information from the document.

Conversation History:
{conversation_history}

User's question: {user_question}

Generated query:
"""
        self.conversation_id = 0
        self.export_data = []


    def load_system_prompt(self, filename: str) -> str:
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            print(f"{Fore.RED}Error: System prompt file '{filename}' not found.")
            print(f"{Fore.YELLOW}Using default system prompt.")
            return """
You are an expert research assistant. Here is a document you will answer questions about:

Source: {source}

Document Content:
{document_content}

Procedure:

1. Identify Relevant Quotes:
   - Carefully review the document to find quotes most relevant to the question.
   - List the quotes in numbered order based on their relevance.
   - Ensure each quote is concise and to the point.

   If no relevant quotes are found, write "No relevant quotes" instead.

2. Answer the Question:
   - Begin your response with "Answer:".
   - Formulate the answer based on the information provided in the quotes, without quoting the content verbatim.
   - Reference the quotes by adding their bracketed numbers at the end of relevant sentences.

Response Format:

Quotes:
[1] "[Relevant quote from the document]"
[2] "[Relevant quote from the document]"

Answer:
[Answer to the question, with references to quotes by their bracketed numbers]

If the question cannot be answered by the document, state so.
"""

    def add_document(self, content: str, source: str):
        self.document_source = source
        self.full_content += content + "\n"
        words = content.split()
        chunks = []
        
        # Check if cached embeddings exist
        cache_file = f"{source}.pkl"
        if os.path.exists(cache_file):
            print(f"{Fore.GREEN}Loading cached embeddings for {source}")
            with open(cache_file, 'rb') as f:
                cached_data = pickle.load(f)
                self.documents.extend(cached_data['chunks'])
                self.embeddings.extend(cached_data['embeddings'])
            return

        print(f"{Fore.YELLOW}Generating embeddings for {source}")
        for i in range(0, len(words), self.chunk_size - self.chunk_overlap):
            chunk = ' '.join(words[i:i + self.chunk_size])
            start_position = i
            end_position = min(i + self.chunk_size, len(words))
            
            chunk_dict = {
                "content": chunk,
                "source": source,
                "start": start_position,
                "end": end_position
            }
            chunks.append(chunk_dict)
            
            embedding = embed.text([chunk], model="nomic-embed-text-v1.5", inference_mode="local")['embeddings'][0]
            self.embeddings.append(embedding)
        
        self.documents.extend(chunks)

        # Cache the embeddings
        with open(cache_file, 'wb') as f:
            pickle.dump({'chunks': chunks, 'embeddings': self.embeddings}, f)
        print(f"{Fore.GREEN}Cached embeddings for {source}")

    def generate_query(self, user_question: str) -> str:
        conversation_history_str = "\n".join([
            f"Q: {turn['content']}\nA: {turn['answer']}"
            for turn in self.conversation_history[-self.history_window:]
        ])
        
        query_prompt = self.query_generation_prompt.format(
            user_question=user_question,
            conversation_history=conversation_history_str
        )
        
        response = self.client.chat.completions.create(
            model="interstellarninja/hermes-2-pro-llama-3-8b:latest",
            messages=[
                {"role": "system", "content": "You are a query generation assistant."},
                {"role": "user", "content": query_prompt}
            ]
        )
        
        generated_query = response.choices[0].message.content.strip()
        
        self.add_to_query_expansion_export(user_question, generated_query, query_prompt)
        
        return generated_query

        
    def add_to_query_expansion_export(self, original_question: str, expanded_query: str, query_prompt: str):
        export_entry = {
            "conversations": [
                {"from": "system", "value": self.query_generation_prompt},
                {"from": "human", "value": query_prompt},
                {"from": "gpt", "value": expanded_query, "weight": 1}
            ],
            "docs": [],  # Query expansion doesn't use document chunks
            "id": self.conversation_id
        }
        self.query_expansion_data.append(export_entry)

    def search_documents(self, query: str) -> List[Dict]:
        query_embedding = embed.text([query], model="nomic-embed-text-v1.5", inference_mode="local")['embeddings'][0]
        similarities = [np.dot(query_embedding, doc_embedding) for doc_embedding in self.embeddings]
        top_indices = np.argsort(similarities)[-self.top_k:][::-1]
        return [
            {
                **self.documents[i],
                "relevance": round(similarities[i], 2)  # Add rounded relevance score
            } 
            for i in top_indices
        ]

    def format_chunks(self, chunks: List[Dict]) -> str:
        formatted_chunks = []
        for i, chunk in enumerate(chunks, start=1):
            formatted_chunk = f"\n[{i}] \"{chunk['content']}\" \n(Source: {chunk['source']}, Words: {chunk['start']}-{chunk['end']}, Relevance: {chunk['relevance']:.2f})\n"
            formatted_chunks.append(formatted_chunk)
        return "\n".join(formatted_chunks)

    def truncate_document(self, text: str, max_length: int = None) -> str:
        if max_length is None:
            max_length = self.max_doc_length
        if len(text) <= max_length:
            return text
        
        # Calculate the length of each part
        half_length = max_length // 2
        
        # Get the first and last parts
        first_part = text[:half_length]
        last_part = text[-half_length:]
        
        # Combine with an indicator of truncation
        return f"{first_part}\n\n... [middle part of document truncated] ...\n\n{last_part}"

    def answer_question(self, user_question: str) -> str:
        try:
            generated_query = self.generate_query(user_question)
            print(f"{Fore.CYAN}Generated Query: {generated_query}")
            
            relevant_chunks = self.search_documents(generated_query)
            formatted_chunks = self.format_chunks(relevant_chunks)
            
            truncated_content = self.truncate_document(self.full_content)
            system_prompt = self.system_prompt_template.format(
                source=self.document_source,
                document_content=truncated_content
            )
            
            conversation_history_str = "\n".join([
                f"Q: {turn['content']}\nA: {turn['answer']}"
                for turn in self.conversation_history[-self.history_window:]
            ])
            
            query_prompt = f"""Based on the following relevant chunks from the document and the conversation history, please answer the question. Follow the procedure and format specified in the system instructions.

Relevant Chunks:
{formatted_chunks}

Conversation History:
{conversation_history_str}

Question: {user_question}
Generated Query: {generated_query}"""

            print(f"\n{Fore.YELLOW}Query Agent System Prompt:")
            print(f"{Fore.WHITE}{system_prompt}\n")

            print(f"\n{Fore.YELLOW}Input Context:")
            print(f"{Fore.WHITE}{query_prompt}\n")

            response = self.client.chat.completions.create(
                model="interstellarninja/hermes-2-pro-llama-3-8b:latest",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query_prompt}
                ]
            )
            
            answer = response.choices[0].message.content
            
            self.conversation_history.append({"content": user_question, "answer": answer})
            self.conversation_history = self.conversation_history[-self.history_window:]
            
            self.add_to_response_export(user_question, answer, relevant_chunks, system_prompt, query_prompt, generated_query)
            
            return answer
        except Exception as e:
            return f"{Fore.RED}An error occurred while processing your question: {str(e)}"




    def add_to_response_export(self, question: str, answer: str, relevant_chunks: List[Dict], system_prompt: str, query_prompt: str, generated_query: str):
        export_entry = {
            "conversations": [
                {"from": "system", "value": system_prompt},
                {"from": "human", "value": query_prompt},
                {"from": "gpt", "value": answer, "weight": 1}
            ],
            "docs": [
                {
                    "content": chunk['content'],
                    "source": chunk['source'],
                    "start": chunk['start'],
                    "end": chunk['end'],
                    "relevance": chunk['relevance']
                } for chunk in relevant_chunks
            ],
            "id": self.conversation_id
        }
        self.response_data.append(export_entry)
        self.conversation_id += 1


    def export_to_jsonl(self, query_expansion_filename: str, response_filename: str):
        with open(query_expansion_filename, 'w', encoding='utf-8') as f:
            for entry in self.query_expansion_data:
                json.dump(entry, f, ensure_ascii=False)
                f.write('\n')
        print(f"{Fore.GREEN}Exported query expansion data to {query_expansion_filename}")

        with open(response_filename, 'w', encoding='utf-8') as f:
            for entry in self.response_data:
                json.dump(entry, f, ensure_ascii=False)
                f.write('\n')
        print(f"{Fore.GREEN}Exported response data to {response_filename}")

# Example usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Conversational Document Q&A Agent")
    parser.add_argument("--input", default="HERMES/content_only.txt", help="Path to the input document")
    parser.add_argument("--chunk-size", type=int, default=100, help="Chunk size for document processing")
    parser.add_argument("--chunk-overlap", type=int, default=20, help="Chunk overlap for document processing")
    parser.add_argument("--history-window", type=int, default=5, help="Number of conversation turns to keep in memory")
    parser.add_argument("--query-expansion-output", default="query_expansion.jsonl", help="Path to the query expansion output JSONL file")
    parser.add_argument("--response-output", default="response.jsonl", help="Path to the response output JSONL file")
    parser.add_argument("--system-prompt", default="system_prompt.md", help="Path to the system prompt markdown file")
    parser.add_argument("--max-doc-length", type=int, default=512, help="Maximum length of document content in system prompt")
    
    args = parser.parse_args()

    try:
        agent = ConversationalDocQAAgent(chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap, 
                                            history_window=args.history_window, system_prompt_file=args.system_prompt,
                                            max_doc_length=args.max_doc_length)

        # Add document
        try:
            with open(args.input, "r", encoding='utf-8') as f:
                content = f.read()
            agent.add_document(content, args.input)
            print(f"{Fore.GREEN}Successfully added document: {args.input}")
        except FileNotFoundError:
            print(f"{Fore.RED}Error: The specified file '{args.input}' was not found.")
            print(f"{Fore.YELLOW}Please make sure the file exists in the correct location.")
            exit(1)
        except Exception as e:
            print(f"{Fore.RED}An error occurred while adding the document: {str(e)}")
            exit(1)

        print(f"{Fore.GREEN}Welcome to the Conversational Document Q&A Agent!")
        print(f"{Fore.CYAN}You can start asking questions about the document. Type 'quit' to exit.")
        print()

        while True:
            user_question = input(f"{Fore.YELLOW}User: {Style.RESET_ALL}").strip()
            if user_question.lower() == 'quit':
                break

            answer = agent.answer_question(user_question)
            print(f"{Fore.MAGENTA}Assistant: {Style.RESET_ALL}{answer}")
            print()

        # Export conversation data
        agent.export_to_jsonl(args.query_expansion_output, args.response_output)
        print(f"{Fore.GREEN}Thank you for using the Conversational Document Q&A Agent. Goodbye!")

    except Exception as e:
        print(f"{Fore.RED}An unexpected error occurred: {str(e)}")
        print(f"{Fore.YELLOW}The application will now exit.")
