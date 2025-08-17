"""
backfill_category.py — Backfill missing YouTube video category IDs in the database.

- Finds videos in `videos` table with no category_id
- Fetches metadata from the YouTube API in batches
- Updates the database with the missing category_ids
"""

import sqlite3, itertools
from .utils import load_env
from . import store, youtube

def _missing_ids():
    # Return video IDs missing category_id from the database.
    con = store.get_conn()
    cur = con.cursor()
    cur.execute("SELECT id FROM videos WHERE category_id IS NULL OR category_id=''")
    ids = [r[0] for r in cur.fetchall()]
    con.close()
    return ids

def _chunks(seq, n):
    # Yield successive n-sized chunks from a sequence.
    it = iter(seq)
    while True:
        chunk = list(itertools.islice(it, n))
        if not chunk:
            return
        yield chunk

def main():
    load_env()
    store.init_db()
    ids = _missing_ids()
    if not ids:
        print("No missing category_id rows.")
        return
    print(f"Backfilling category_id for {len(ids)} videos...")
    for chunk in _chunks(ids, 50): # process in batches of 50
        items = youtube.video_details(chunk)
        store.upsert_videos(items)
    print("Done.")

if __name__ == "__main__":
    main()
