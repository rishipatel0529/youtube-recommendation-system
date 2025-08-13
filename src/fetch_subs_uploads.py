# src/fetch_subs_uploads.py
import time
from src.utils import load_env
from src import store, youtube

BATCH = 30      # channels per pass
PER_CH = 10     # videos per channel
SLEEP = 0.5     # polite pause

def main():
    load_env()          # ← load YOUTUBE_API_KEY, etc.
    store.init_db()     # ← ensure tables exist

    subs = store.get_subscriptions()
    if not subs:
        print("No subscriptions found. Run import_subscriptions first.")
        return

    print(f"Fetching uploads for {len(subs)} channels...")
    total = 0
    for i in range(0, len(subs), BATCH):
        chunk = subs[i:i+BATCH]
        all_items = []
        for s in chunk:
            try:
                vids = youtube.channel_uploads_videos(s["channel_id"], max_results=PER_CH)
                all_items.extend(vids)
            except Exception:
                pass
            time.sleep(SLEEP)
        if all_items:
            store.upsert_videos(all_items)
            total += len(all_items)
        print(f"Upserted ~{len(all_items)} from channels {i+1}-{i+len(chunk)}")
    print(f"Done. Total upserted: ~{total}")

if __name__ == "__main__":
    main()
