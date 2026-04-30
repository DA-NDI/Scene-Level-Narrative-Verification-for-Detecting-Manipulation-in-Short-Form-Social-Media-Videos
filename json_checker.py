import os
import json
import glob

# CONFIGURATION
VIDEO_FOLDER = "./videos" 

def check_linkage():
    # 1. Get all video files
    if not os.path.exists(VIDEO_FOLDER):
        print(f"❌ Error: Video folder '{VIDEO_FOLDER}' not found.")
        return
        
    print(f"Scanning {VIDEO_FOLDER}...")
    video_files = set(os.listdir(VIDEO_FOLDER))
    print(f"Found {len(video_files)} video files locally.")
    
    # 2. Find all JSON files in the current directory
    json_files = glob.glob("*.json")
    if not json_files:
        print("❌ Error: No .json files found in the current directory.")
        return

    print(f"Found {len(json_files)} JSON files: {json_files}")

    # 3. Iterate through each JSON file
    for json_file in json_files:
        print(f"\nProcessing metadata file: {json_file}...")
        
        try:
            with open(json_file, 'r') as f:
                # Handle both single-object list and JSONL (one object per line)
                try:
                    metadata = json.load(f)
                except json.JSONDecodeError:
                    f.seek(0)
                    metadata = [json.loads(line) for line in f]
        except Exception as e:
            print(f"⚠️ Could not read {json_file}: {e}")
            continue

        # Counters for this specific file
        matched_pub_id = 0
        matched_uuid = 0
        total_entries = len(metadata)

        # Check matches
        for entry in metadata:
            # Skip if it's not a video entry
            if 'publication_id' not in entry and 'id' not in entry:
                continue

            # Check 1: TikTok ID
            if 'publication_id' in entry:
                # Convert ID to string just in case
                if f"{entry['publication_id']}.mp4" in video_files:
                    matched_pub_id += 1
            
            # Check 2: UUID
            if 'id' in entry:
                if f"{entry['id']}.mp4" in video_files:
                    matched_uuid += 1

        print(f"  - Total Entries: {total_entries}")
        print(f"  - Matched (TikTok ID): {matched_pub_id}")
        print(f"  - Matched (UUID):      {matched_uuid}")

        if matched_pub_id > 0:
            print("  ✅ This file matches your videos using TikTok IDs.")
        elif matched_uuid > 0:
            print("  ✅ This file matches your videos using UUIDs.")
        else:
            print("  ⚠️ No matches found in this file (likely author/comment data).")

if __name__ == "__main__":
    check_linkage()
