import json
from pathlib import Path
import sys

def downsize(input_path: Path, output_path: Path, percentage: float = 0.25):
    print(f"Reading {input_path}...")
    with open(input_path, "r") as f:
        data = json.load(f)
    
    original_count = len(data["data"])
    target_count = int(original_count * percentage)
    
    print(f"Original articles: {original_count}")
    print(f"Target articles: {target_count} ({percentage*100}%)")
    
    # Slice to keep the first N articles
    new_data = data["data"][:target_count]
    
    output_data = {
        "version": data.get("version", "1.1"),
        "data": new_data
    }
    
    print(f"Writing {output_path}...")
    with open(output_path, "w") as f:
        json.dump(output_data, f)
    
    print("Done!")

if __name__ == "__main__":
    # Adjust path to find data/documents relative to this script
    base_dir = Path(__file__).resolve().parent.parent / "data" / "documents"
    input_file = base_dir / "SQuAD-v1.1.json"
    output_file = base_dir / "SQuAD-small.json"
    
    if not input_file.exists():
        print(f"Error: {input_file} not found")
        sys.exit(1)
        
    downsize(input_file, output_file)
