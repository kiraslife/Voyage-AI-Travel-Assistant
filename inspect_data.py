import json
import os

# Define the path to our data folder
DATA_DIR = "data"

def inspect_json_file(filename):
    """
    Loads a JSON file, prints the total number of records,
    and displays the exact keys and values of the first record.
    """
    filepath = os.path.join(DATA_DIR, filename)
    
    # Always use try-except for file and API operations (Project Guideline!)
    try:
        with open(filepath, "r", encoding="utf-8") as file:
            data = json.load(file)
            
        print("=" * 50)
        print(f"FILE: {filename}")
        print(f"Total Records Found: {len(data)}")
        
        # Check if data is a list of dictionaries
        if isinstance(data, list) and len(data) > 0:
            first_item = data[0]
            print(f"Keys in {filename}: {list(first_item.keys())}")
            print("\nSample First Record:")
            # json.dumps with indent=2 prints the dictionary cleanly
            print(json.dumps(first_item, indent=2))
        elif isinstance(data, dict):
            # In case the JSON root is a dictionary instead of a list
            print(f"Root Keys in {filename}: {list(data.keys())}")
            
    except FileNotFoundError:
        print(f"Error: {filepath} not found. Check your folder path!")
    except json.JSONDecodeError:
        print(f"Error: {filepath} is not a valid JSON file.")
    except Exception as e:
        print(f"Unexpected error reading {filename}: {e}")

if __name__ == "__main__":
    # Inspect all three datasets required for our tools
    inspect_json_file("flights.json")
    inspect_json_file("hotels.json")
    inspect_json_file("places.json")