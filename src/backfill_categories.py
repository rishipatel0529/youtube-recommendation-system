import sqlite3, itertools
from .utils import load_env
from . import store, youtube

def _missing_ids():
    con = store.get_conn()
    cur = con.cursor()
    cur.execute("SELECT id FROM videos WHERE category_id IS NULL OR category_id=''")
    ids = [r[0] for r in cur.fetchall()]
    con.close()
    return ids

def _chunks(seq, n):
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
    for chunk in _chunks(ids, 50):
        items = youtube.video_details(chunk)
        store.upsert_videos(items)
    print("Done.")

if __name__ == "__main__":
    main()
