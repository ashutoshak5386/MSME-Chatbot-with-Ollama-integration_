import json
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_schemes_json(file_path):
    """Load and process a single JSON file."""
    try:
        if hasattr(file_path, 'read'):  # Handle file upload object
            data = json.load(file_path).get("data", {})
        else:  # Handle file path string
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f).get("data", {})
                
        fields = ["scheme_name", "objective", "benefits", "eligibility", 
                 "conditions", "description", "additional_info"]
        text = ""
        
        for field in fields:
            if field in data and data[field]:
                text += f"{field.replace('_',' ').capitalize()}: {data[field]}\n"
                
        if not text.strip():
            return []
            
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        return splitter.split_text(text)
        
    except json.JSONDecodeError:
        print(f"Invalid JSON in file: {file_path}")
        return []
    except Exception as e:
        print(f"Error processing file {file_path}: {str(e)}")
        return []

def load_all_schemes_from_folder(folder_path):
    """Load and process all JSON files in a folder."""
    all_chunks = []
    
    if not os.path.exists(folder_path):
        return all_chunks
        
    for filename in os.listdir(folder_path):
        if filename.endswith('.json'):
            file_path = os.path.join(folder_path, filename)
            chunks = load_schemes_json(file_path)
            if chunks:
                all_chunks.extend(chunks)
                
    return all_chunks