"""
Single source of truth for progressive overload exercise count.

Fitness level -> completed session count -> exercise count per session:
  Beginner:     0-5 -> 4,  6-15 -> 5,  16+ -> 6
  Intermediate: 0-5 -> 6,  6-15 -> 7,  16+ -> 8
  Advanced:     0-5 -> 8,  6-15 -> 10, 16+ -> 12

Import this module wherever exercise count decisions are made.
Do NOT duplicate this logic in other files.
"""

# (min_sessions_inclusive, max_sessions_exclusive, count)
# max_sessions_exclusive=None means "16 and above" (no upper bound)
PROGRESSIVE_OVERLOAD_RANGES = {
    'beginner':     [(0, 6, 4),  (6, 16, 5),  (16, None, 6)],
    'intermediate': [(0, 6, 6),  (6, 16, 7),  (16, None, 8)],
    'advanced':     [(0, 6, 8),  (6, 16, 10), (16, None, 12)],
}

# Alternate spellings accepted by the ML service
_LEVEL_ALIASES = {
    'medium': 'intermediate',
    'expert': 'advanced',
}

# (min, max) exercise count bounds per level.
# Used as a reference by downstream validators (e.g. PHP range check).
# Update here when professor requirements change.
EXERCISE_COUNT_BOUNDS = {
    'beginner':     (4, 6),
    'intermediate': (6, 8),
    'advanced':     (8, 12),
}


def get_exercise_count(fitness_level: str, session_count: int = 0) -> int:
    """Return exercise count for a string fitness level and session count."""
    level = (fitness_level or 'beginner').lower()
    level = _LEVEL_ALIASES.get(level, level)
    ranges = PROGRESSIVE_OVERLOAD_RANGES.get(level, PROGRESSIVE_OVERLOAD_RANGES['beginner'])
    for low, high, count in ranges:
        if session_count >= low and (high is None or session_count < high):
            return count
    return 4  # hard fallback — should never be reached


def map_numeric_to_level(fitness_level_numeric: int) -> str:
    """Map numeric fitness level (1/3/5) to string level name."""
    if fitness_level_numeric <= 1:
        return 'beginner'
    elif fitness_level_numeric <= 3:
        return 'intermediate'
    return 'advanced'


def get_exercise_count_from_numeric(fitness_level_numeric: int, session_count: int = 0) -> int:
    """Return exercise count for a numeric fitness level (1/3/5) and session count."""
    return get_exercise_count(map_numeric_to_level(fitness_level_numeric), session_count)
