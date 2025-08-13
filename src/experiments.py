# src/experiments.py
from collections import deque

def team_draft_interleave(list_a, list_b, k=10):
    """
    Returns (interleaved_ids, assignment_map) where assignment_map[id] ∈ {"A","B"}.
    Stable team-draft interleaving.
    """
    A = deque([x for x in list_a if x])  # ids only
    B = deque([x for x in list_b if x])
    out, owner = [], {}
    turn = "A"
    seen = set()
    while len(out) < k and (A or B):
        pick = None
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

        if pick and pick not in seen:
            out.append(pick)
            owner[pick] = turn
            seen.add(pick)

        # alternate
        turn = "B" if turn == "A" else "A"

        # if one side is empty, pull from the other
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
