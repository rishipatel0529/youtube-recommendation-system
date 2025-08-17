"""
backfill_tags.py
Backfill missing video tags in the database using YouTube API.
"""

from src import store, youtube
from src.utils import load_env

def main():
    # Fetch and update missing tags for videos in bulk.
    load_env()
    ids = store.list_ids_missing_tags(limit=1000)
    if not ids:
        print("No videos missing tags.")
        return
    tag_map = youtube.fetch_tags_for_ids(ids)
    store.update_tags_bulk(tag_map)
    print(f"Updated tags for {len(tag_map)} videos.")

if __name__ == "__main__":
    main()
