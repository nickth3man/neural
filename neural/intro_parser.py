"""Intro-window parsing for Tier 1 document-level metadata extraction.

Extracts deterministic metadata from the first N lines of a transcript:
- Host names from cast intros
- Guest names from explicit introductions
- Holiday/theme cues
- Episode type signals
- Sponsor mentions
- Topic rundown seeds
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from neural.corpus import TranscriptDocument, TranscriptLine
from neural.metadata_types import (
    ContentType,
    DocumentAnchor,
    EpisodeType,
)

# ============================================================================
# Configuration
# ============================================================================

# Intro window: first N seconds of transcript (roughly 3 minutes)
INTRO_WINDOW_SECONDS = 180

# Known show cast for roster priors (Gilbert Arenas show)
KNOWN_CAST_ALIASES: dict[str, str] = {
    "gilbert arenas": "Gilbert Arenas",
    "gilbert": "Gilbert Arenas",
    "gil": "Gilbert Arenas",
    "agent zero": "Gilbert Arenas",
    "swaggy p": "Nick Young",
    "nick young": "Nick Young",
    "swaggy": "Nick Young",
    "mr. b": "Brandon Jennings",
    "mr b": "Brandon Jennings",
    "brandon jennings": "Brandon Jennings",
    "bj": "Brandon Jennings",
    "josiah johnson": "Josiah Johnson",
    "josiah": "Josiah Johnson",
    "host": "Josiah Johnson",
}

# Cast intro patterns
CAST_INTRO_PATTERNS: list[re.Pattern[str]] = [
    re.compile(
        r"(?:we'?ve got|got|here with us is|joining us|in the (?:building|arena|house))\s+"
        r"(?P<name>[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2})",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:legend|nba champion|former|ncaa champion)\s+(?P<name>[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2})",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?P<name>[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2})\s+(?:is here|in the (?:building|arena|house))",
        re.IGNORECASE,
    ),
    re.compile(
        r"special guest.*?(?P<name>[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2})",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:this side of the couch|next to \w+).*?(?P<name>[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2})",
        re.IGNORECASE,
    ),
]

# Host intro patterns
HOST_INTRO_PATTERNS: list[re.Pattern[str]] = [
    re.compile(
        r"(?:i'?m|merely a host|your host)\s+(?P<name>[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2})",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?P<name>[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2})\s+(?:here|in the building)",
        re.IGNORECASE,
    ),
]

# Holiday/theme patterns
HOLIDAY_PATTERNS: dict[str, list[re.Pattern[str]]] = {
    "halloween": [
        re.compile(r"\bhalloween\b", re.IGNORECASE),
        re.compile(r"\bspooky\b", re.IGNORECASE),
        re.compile(r"\btrick.or.treat\b", re.IGNORECASE),
    ],
    "christmas": [
        re.compile(r"\bchristmas\b", re.IGNORECASE),
        re.compile(r"\bxmas\b", re.IGNORECASE),
        re.compile(r"\bholiday season\b", re.IGNORECASE),
    ],
    "thanksgiving": [
        re.compile(r"\bthanksgiving\b", re.IGNORECASE),
    ],
    "st_patricks": [
        re.compile(r"\bst\.?\s*patrick\b", re.IGNORECASE),
    ],
    "juneteenth": [
        re.compile(r"\bjuneteenth\b", re.IGNORECASE),
    ],
    "new_years": [
        re.compile(r"\bnew\s+year\b", re.IGNORECASE),
    ],
}

# Sponsor/ad-read patterns
SPONSOR_PATTERNS: list[re.Pattern[str]] = [
    re.compile(
        r"presented by\s+(?P<sponsor>[A-Za-z\s]+?)(?:\.|,|\s+Woo|\s+World)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:brought to you by|sponsored by|brought to you by|shout out to)\s+(?P<sponsor>[A-Za-z\s&']+)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:use )?promo code\s+\w+",
        re.IGNORECASE,
    ),
]

# Topic rundown patterns
TOPIC_RUNDOWN_PATTERNS: list[re.Pattern[str]] = [
    re.compile(
        r"(?:here'?s what'?s cracking|here'?s what'?s on tap|topics for today|today'?s (?:topics|lineup))",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:we'?re going to (?:talk about|get into|discuss))\s+(?P<topic>.{10,100}?)\.",
        re.IGNORECASE,
    ),
]

# Episode type signals
EPISODE_TYPE_SIGNALS: dict[EpisodeType, list[re.Pattern[str]]] = {
    EpisodeType.GUEST_INTERVIEW: [
        re.compile(r"\bspecial guest\b", re.IGNORECASE),
        re.compile(r"\bwelcome.*?to the arena\b", re.IGNORECASE),
        re.compile(r"\bfirst appearance\b", re.IGNORECASE),
    ],
    EpisodeType.RECAP: [
        re.compile(r"\brecap\b", re.IGNORECASE),
        re.compile(r"\bround up\b", re.IGNORECASE),
        re.compile(r"\bwrap up\b", re.IGNORECASE),
    ],
    EpisodeType.REACTION: [
        re.compile(r"\breact(?:s|ion|ing)\b", re.IGNORECASE),
        re.compile(r"\bbreak(?:s)?\s+down\b", re.IGNORECASE),
    ],
    EpisodeType.PREDICTION: [
        re.compile(r"\bpredict(?:s|ion|ing)\b", re.IGNORECASE),
        re.compile(r"\bpicks?\b", re.IGNORECASE),
    ],
    EpisodeType.TRADE_BREAKDOWN: [
        re.compile(r"\btrade\b", re.IGNORECASE),
        re.compile(r"\bdeal\b", re.IGNORECASE),
    ],
    EpisodeType.PLAYOFF_RECAP: [
        re.compile(r"\bplayoff\b", re.IGNORECASE),
        re.compile(r"\bfinals?\b", re.IGNORECASE),
    ],
    EpisodeType.LEAGUE_NEWS: [
        re.compile(r"\bleague news\b", re.IGNORECASE),
        re.compile(r"\bnba news\b", re.IGNORECASE),
    ],
    EpisodeType.SEASON_PREMIERE: [
        re.compile(r"\bseason (?:premiere|opener|kickoff|tip.?off)\b", re.IGNORECASE),
        re.compile(r"\bwelcome back\b", re.IGNORECASE),
    ],
}


# ============================================================================
# Intro Window Extraction
# ============================================================================


@dataclass(frozen=True)
class IntroWindow:
    """Parsed intro-window metadata with anchors."""

    lines: tuple[TranscriptLine, ...]
    text: str
    host_names: list[str]
    guest_names: list[str]
    holiday_theme: str
    episode_type: EpisodeType
    sponsors: list[str]
    topic_rundown: str
    content_type: ContentType
    episode_roster: list[str]

    # Anchors for traceability
    host_names_anchor: DocumentAnchor | None
    guest_names_anchor: DocumentAnchor | None
    holiday_theme_anchor: DocumentAnchor | None
    episode_type_anchor: DocumentAnchor | None
    topic_anchor: DocumentAnchor | None


def extract_intro_window(
    document: TranscriptDocument,
    *,
    window_seconds: int = INTRO_WINDOW_SECONDS,
) -> IntroWindow:
    """Extract intro-window metadata from a transcript document.

    Takes the first `window_seconds` of transcript lines and parses
    deterministic metadata from cast intros, ad reads, topic rundowns,
    and holiday/theme cues.
    """
    if not document.lines:
        return _empty_intro_window()

    # Find lines within intro window
    intro_lines = _get_intro_lines(document.lines, window_seconds)
    if not intro_lines:
        return _empty_intro_window()

    intro_text = " ".join(line.text for line in intro_lines)

    # Extract each field
    host_names, host_anchor = _extract_host_names(intro_lines, intro_text)
    guest_names, guest_anchor = _extract_guest_names(intro_lines, intro_text)
    holiday_theme, holiday_anchor = _extract_holiday_theme(intro_lines, intro_text)
    episode_type, type_anchor = _extract_episode_type(intro_lines, intro_text, guest_names)
    sponsors = _extract_sponsors(intro_text)
    topic_rundown, topic_anchor = _extract_topic_rundown(intro_lines, intro_text)
    content_type = _classify_intro_content(intro_text, guest_names)
    episode_roster = _build_episode_roster(host_names, guest_names, intro_text)

    return IntroWindow(
        lines=intro_lines,
        text=intro_text,
        host_names=host_names,
        guest_names=guest_names,
        holiday_theme=holiday_theme,
        episode_type=episode_type,
        sponsors=sponsors,
        topic_rundown=topic_rundown,
        content_type=content_type,
        episode_roster=episode_roster,
        host_names_anchor=host_anchor,
        guest_names_anchor=guest_anchor,
        holiday_theme_anchor=holiday_anchor,
        episode_type_anchor=type_anchor,
        topic_anchor=topic_anchor,
    )


def _get_intro_lines(
    lines: tuple[TranscriptLine, ...],
    window_seconds: int,
) -> tuple[TranscriptLine, ...]:
    """Get lines within the intro window time range."""
    if not lines:
        return ()

    # First line starts the window
    start_seconds = lines[0].start_seconds
    cutoff = start_seconds + window_seconds

    intro_lines = []
    for line in lines:
        if line.start_seconds > cutoff:
            break
        intro_lines.append(line)

    return tuple(intro_lines)


def _empty_intro_window() -> IntroWindow:
    """Return an empty intro window for transcripts with no parseable content."""
    return IntroWindow(
        lines=(),
        text="",
        host_names=[],
        guest_names=[],
        holiday_theme="",
        episode_type=EpisodeType.REACTION,
        sponsors=[],
        topic_rundown="",
        content_type=ContentType.COLD_OPEN_BANTER,
        episode_roster=[],
        host_names_anchor=None,
        guest_names_anchor=None,
        holiday_theme_anchor=None,
        episode_type_anchor=None,
        topic_anchor=None,
    )


def _extract_host_names(
    intro_lines: tuple[TranscriptLine, ...],
    intro_text: str,
) -> tuple[list[str], DocumentAnchor | None]:
    """Extract host names from intro window."""
    # Check known aliases first
    text_lower = intro_text.lower()
    for alias, canonical in KNOWN_CAST_ALIASES.items():
        if alias in text_lower and canonical in [
            "Josiah Johnson",
        ]:
            return [canonical], _find_anchor_for_text(intro_lines, alias)

    # Try pattern matching
    for pattern in HOST_INTRO_PATTERNS:
        match = pattern.search(intro_text)
        if match:
            name = match.group("name").strip()
            canonical = _resolve_canonical_name(name)
            if canonical:
                return [canonical], _find_anchor_for_text(intro_lines, name.lower())

    # Fallback: Josiah Johnson is the default host
    if "josiah" in text_lower or "host" in text_lower:
        return ["Josiah Johnson"], None

    return [], None


def _extract_guest_names(
    intro_lines: tuple[TranscriptLine, ...],
    intro_text: str,
) -> tuple[list[str], DocumentAnchor | None]:
    """Extract guest names from explicit introductions in the intro window."""
    guests: list[str] = []
    anchor: DocumentAnchor | None = None

    # Look for "special guest" introductions
    for line in intro_lines:
        line_lower = line.text.lower()
        if "special guest" in line_lower or "first appearance" in line_lower:
            # Try to extract the name
            for pattern in CAST_INTRO_PATTERNS:
                match = pattern.search(line.text)
                if match:
                    name = match.group("name").strip()
                    canonical = _resolve_canonical_name(name)
                    if canonical and canonical not in guests:
                        guests.append(canonical)
                        if anchor is None:
                            anchor = DocumentAnchor(
                                line_index=line.start_seconds,
                                timestamp=line.timestamp,
                                excerpt=line.text[:100],
                            )

    return guests, anchor


def _extract_holiday_theme(
    intro_lines: tuple[TranscriptLine, ...],
    intro_text: str,
) -> tuple[str, DocumentAnchor | None]:
    """Extract holiday/theme cues from the intro window."""
    for holiday, patterns in HOLIDAY_PATTERNS.items():
        for pattern in patterns:
            match = pattern.search(intro_text)
            if match:
                anchor_text = match.group(0)
                anchor = _find_anchor_for_text(intro_lines, anchor_text.lower())
                return holiday.replace("_", " "), anchor

    return "", None


def _extract_episode_type(
    intro_lines: tuple[TranscriptLine, ...],
    intro_text: str,
    guest_names: list[str],
) -> tuple[EpisodeType, DocumentAnchor | None]:
    """Classify episode type from intro signals."""
    # Guest presence implies guest interview
    if guest_names:
        return (
            EpisodeType.GUEST_INTERVIEW,
            _find_anchor_for_text(intro_lines, "special guest"),
        )

    # Try each episode type signal
    for ep_type, patterns in EPISODE_TYPE_SIGNALS.items():
        for pattern in patterns:
            match = pattern.search(intro_text)
            if match:
                anchor = _find_anchor_for_text(intro_lines, match.group(0).lower())
                return ep_type, anchor

    # Default to reaction
    return EpisodeType.REACTION, None


def _extract_sponsors(intro_text: str) -> list[str]:
    """Extract sponsor names from ad reads in the intro window."""
    sponsors: list[str] = []

    for pattern in SPONSOR_PATTERNS:
        for match in pattern.finditer(intro_text):
            if "sponsor" in match.groupdict():
                sponsor = match.group("sponsor").strip()
                if sponsor and sponsor not in sponsors:
                    sponsors.append(sponsor)

    # Known sponsors from transcript patterns
    known_sponsors = ["Underdog", "Wendy's", "Starbucks", "PC Financial", "Tia Rosa", "Adidas"]
    for sponsor in known_sponsors:
        if sponsor.lower() in intro_text.lower() and sponsor not in sponsors:
            sponsors.append(sponsor)

    return sponsors


def _extract_topic_rundown(
    intro_lines: tuple[TranscriptLine, ...],
    intro_text: str,
) -> tuple[str, DocumentAnchor | None]:
    """Extract topic seed from topic-rundown lines."""
    # Look for topic rundown patterns
    for pattern in TOPIC_RUNDOWN_PATTERNS:
        match = pattern.search(intro_text)
        if match:
            topic = match.groupdict().get("topic", "")
            anchor = _find_anchor_for_text(intro_lines, match.group(0).lower())
            return topic.strip(), anchor

    # Fallback: look for "here's what's cracking" line
    for line in intro_lines:
        if (
            "here's what's cracking" in line.text.lower()
            or "here's what's on tap" in line.text.lower()
        ):
            # Get the next few lines as topic seed
            idx = intro_lines.index(line)
            topic_lines = []
            for next_line in intro_lines[idx + 1 : idx + 4]:
                topic_lines.append(next_line.text)
            topic = " ".join(topic_lines).strip()
            if topic:
                anchor = DocumentAnchor(
                    line_index=line.start_seconds,
                    timestamp=line.timestamp,
                    excerpt=line.text[:100],
                )
                return topic[:200], anchor

    return "", None


def _classify_intro_content(
    intro_text: str,
    guest_names: list[str],
) -> ContentType:
    """Classify the primary content type of the intro window."""
    text_lower = intro_text.lower()

    if "special guest" in text_lower or guest_names:
        return ContentType.GUEST_INTERVIEW
    if any(s in text_lower for s in ["promo code", "download the app", "bonus"]):
        return ContentType.PROMO_READ
    if any(s in text_lower for s in ["ad", "sponsored", "brought to you"]):
        return ContentType.AD_READ
    if "here's what's cracking" in text_lower or "here's what's on tap" in text_lower:
        return ContentType.TOPIC_RUNDOWN
    if any(s in text_lower for s in ["welcome back", "good to see", "how's everybody"]):
        return ContentType.CAST_INTRO

    return ContentType.COLD_OPEN_BANTER


def _build_episode_roster(
    host_names: list[str],
    guest_names: list[str],
    intro_text: str,
) -> list[str]:
    """Build closed episode roster from intro mentions and known cast."""
    roster: list[str] = []

    # Add hosts
    roster.extend(host_names)

    # Add guests
    for guest in guest_names:
        if guest not in roster:
            roster.append(guest)

    # Scan intro for known cast aliases
    text_lower = intro_text.lower()
    for alias, canonical in KNOWN_CAST_ALIASES.items():
        if alias in text_lower and canonical not in roster:
            roster.append(canonical)

    # Ensure at least the core cast
    core_cast = ["Gilbert Arenas", "Nick Young", "Brandon Jennings"]
    for member in core_cast:
        if member not in roster:
            roster.append(member)

    return roster


def _resolve_canonical_name(name: str) -> str | None:
    """Resolve a name to its canonical form using known aliases."""
    name_lower = name.lower().strip()

    # Direct alias lookup
    if name_lower in KNOWN_CAST_ALIASES:
        return KNOWN_CAST_ALIASES[name_lower]

    # Check if it's already a canonical name
    canonical_names = set(KNOWN_CAST_ALIASES.values())
    if name in canonical_names:
        return name

    # Return as-is if it looks like a proper name
    parts = name.split()
    if len(parts) >= 2 and all(p[0].isupper() for p in parts if p):
        return name

    return None


def _find_anchor_for_text(
    intro_lines: tuple[TranscriptLine, ...],
    search_text: str,
) -> DocumentAnchor | None:
    """Find the anchor line containing the search text."""
    search_lower = search_text.lower()
    for line in intro_lines:
        if search_lower in line.text.lower():
            return DocumentAnchor(
                line_index=line.start_seconds,
                timestamp=line.timestamp,
                excerpt=line.text[:100],
            )
    return None
