"""
categories.py — Map YouTube category IDs to human-readable labels.

- Provides a dictionary mapping of category_id → category name
- Utility function `label()` returns the readable name or "Unknown"
"""

# Mapping of YouTube video category IDs to names
ID_TO_NAME = {
    "1":"Film & Animation","2":"Autos & Vehicles","10":"Music","15":"Pets & Animals",
    "17":"Sports","18":"Short Movies","19":"Travel & Events","20":"Gaming","21":"Videoblogging",
    "22":"People & Blogs","23":"Comedy","24":"Entertainment","25":"News & Politics",
    "26":"Howto & Style","27":"Education","28":"Science & Technology","29":"Nonprofits & Activism",
}

def label(cat_id: str) -> str:
    # Return category name for a given ID, or 'Unknown' if not found.
    return ID_TO_NAME.get(cat_id or "", "Unknown")