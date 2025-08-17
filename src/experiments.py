"""
experiments.py — Utility for combining two ranked lists via team-draft interleaving.

Purpose:
- Interleave two candidate lists (A and B) in a balanced, fair way.
- Used for comparing ranking algorithms or mixing recommendation sources.

Returns:
- interleaved_ids: merged list of up to k items
- assignment_map: dict mapping each id → "A" or "B"
"""

from collections import deque

def team_draft_interleave(list_a, list_b, k=10):
    """
    Perform stable team-draft interleaving of two ranked lists.

    Args:
        list_a (list): First ranked list of IDs.
        list_b (list): Second ranked list of IDs.
        k (int): Max length of output list.

    Returns:
        tuple: (interleaved_ids, assignment_map)
    """
    A = deque([x for x in list_a if x])
    B = deque([x for x in list_b if x])
    out, owner = [], {}
    turn = "A" # start with A
    seen = set()
    while len(out) < k and (A or B):
        pick = None

        # take next from whichever side's turn, skipping duplicates
        if turn == "A" and A:
            while A and (A[0] in seen):
                A.popleft()
            if A:
                pick = A.popleft()
        elif turn == "B" and B:
            while B and (B[0] in seen):
                B.popleft()
            if B:
                pick = B.popleft()

        # add valid pick
        if pick and pick not in seen:
            out.append(pick)
            owner[pick] = turn
            seen.add(pick)

        # alternate turns
        turn = "B" if turn == "A" else "A"

        # if one side runs out, drain from the other until k
        if not A and B and len(out) < k:
            while B and len(out) < k:
                v = B.popleft()
                if v not in seen:
                    out.append(v); owner[v] = "B"; seen.add(v)
        if not B and A and len(out) < k:
            while A and len(out) < k:
                v = A.popleft()
                if v not in seen:
                    out.append(v); owner[v] = "A"; seen.add(v)

    return out[:k], owner
