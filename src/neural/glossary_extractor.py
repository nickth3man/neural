"""Document-level glossary extraction using OpenRouter LLM.

Runs a structured-output prompt on intro windows and manifest context
to populate deterministic or semi-deterministic document fields.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from neural.corpus import TranscriptDocument
from neural.intro_parser import IntroWindow, extract_intro_window
from neural.metadata_types import (
    ContentType,
    DocumentAnchor,
    DocumentMetadata,
    EpisodeManifestEntry,
    EpisodeType,
)
from neural.openrouter import OpenRouterError, complete_chat

# ============================================================================
# Configuration
# ============================================================================

DOCUMENT_EXTRACTION_SYSTEM_PROMPT = """\
You are a transcript metadata extractor for the "Gil's Arena" NBA podcast.

Your task is to extract structured metadata from the intro window of a transcript.

Rules:
1. Use ONLY information explicitly stated in the text.
2. For host names, match against the cast introductions (e.g., \
"we've got the legend Gilbert Arenas").
3. For guest names, only include people explicitly labeled as \
"special guest" or "first appearance."
4. For holiday themes, detect holiday or theme references \
(Halloween, Christmas, Juneteenth, etc.).
5. For episode type, classify based on the show's framing \
(recap, reaction, prediction, guest interview, etc.).
6. For topic, extract the primary topic from topic-rundown lines \
("here's what's cracking").
7. Build the episode roster from ALL speaker names mentioned in cast intros.
8. Return valid JSON matching the schema exactly. No markdown, no explanation.

Controlled vocabularies:
- episode_type: season_premiere, recap, reaction, prediction, \
guest_interview, trade_breakdown, playoff_recap, league_news
- content_type: ad_read, cold_open_banter, cast_intro, topic_rundown, \
main_discussion, guest_interview, promo_read, outro
- topic_category: team, player, league, event, business, \
social_justice, show_meta, sponsor, holiday
"""

DOCUMENT_EXTRACTION_USER_TEMPLATE = """\
Extract metadata from this Gil's Arena podcast transcript intro window.

TRANSCRIPT EXCERPT (first {line_count} lines, ~{seconds} seconds):
---
{intro_text}
---

EPISODE TITLE: {episode_title}
{manifest_context}

Return JSON with all required fields. Use empty arrays/strings where no data found.\
"""


# ============================================================================
# Extraction Functions
# ============================================================================


@dataclass(frozen=True)
class DocumentExtractionResult:
    """Result of document-level metadata extraction."""

    metadata: DocumentMetadata
    raw_response: str
    intro_window: IntroWindow
    used_llm: bool


def extract_document_metadata(
    document: TranscriptDocument,
    *,
    api_key: str | None = None,
    model: str | None = None,
    manifest_entry: EpisodeManifestEntry | None = None,
    skip_llm: bool = False,
) -> DocumentExtractionResult:
    """Extract document-level metadata from a transcript.

    Pipeline:
    1. Parse intro window (deterministic)
    2. Run LLM extraction on intro window (if not skipped)
    3. Merge deterministic and LLM results

    Args:
        document: Loaded transcript document.
        api_key: OpenRouter API key (reads from env if None).
        model: OpenRouter model override.
        manifest_entry: Optional manifest data for episode context.
        skip_llm: If True, skip LLM extraction (use deterministic only).

    Returns:
        DocumentExtractionResult with metadata, raw response, and intro window.
    """
    # Step 1: Parse intro window (deterministic)
    intro_window = extract_intro_window(document)

    # Step 2: LLM extraction (if not skipped)
    if skip_llm or not api_key:
        return DocumentExtractionResult(
            metadata=_build_metadata_from_intro(document, intro_window, manifest_entry),
            raw_response="",
            intro_window=intro_window,
            used_llm=False,
        )

    # Build manifest context
    manifest_context = ""
    if manifest_entry:
        manifest_context = f"""MANIFEST DATA:
- Title: {manifest_entry.title}
- Date: {manifest_entry.date}
- Description: {manifest_entry.description}
- URL Slug: {manifest_entry.url_slug}"""

    # Build the prompt
    user_message = DOCUMENT_EXTRACTION_USER_TEMPLATE.format(
        line_count=len(intro_window.lines),
        seconds=180,
        intro_text=intro_window.text[:2000],  # Limit intro text length
        episode_title=document.episode_title,
        manifest_context=manifest_context,
    )

    try:
        raw_response = complete_chat(
            messages=[
                {"role": "system", "content": DOCUMENT_EXTRACTION_SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            api_key=api_key,
            model=model,
            temperature=0.1,  # Low temperature for deterministic extraction
            timeout_seconds=60.0,
        )
    except OpenRouterError:
        # Fallback to deterministic extraction on LLM failure
        return DocumentExtractionResult(
            metadata=_build_metadata_from_intro(document, intro_window, manifest_entry),
            raw_response="",
            intro_window=intro_window,
            used_llm=False,
        )

    # Parse LLM response
    try:
        llm_data = _parse_llm_response(raw_response)
    except (json.JSONDecodeError, KeyError):
        # Fallback to deterministic extraction on parse failure
        return DocumentExtractionResult(
            metadata=_build_metadata_from_intro(document, intro_window, manifest_entry),
            raw_response=raw_response,
            intro_window=intro_window,
            used_llm=False,
        )

    # Merge deterministic and LLM results
    metadata = _merge_extraction_results(
        document=document,
        intro_window=intro_window,
        llm_data=llm_data,
        manifest_entry=manifest_entry,
    )

    return DocumentExtractionResult(
        metadata=metadata,
        raw_response=raw_response,
        intro_window=intro_window,
        used_llm=True,
    )


def _build_metadata_from_intro(
    document: TranscriptDocument,
    intro_window: IntroWindow,
    manifest_entry: EpisodeManifestEntry | None,
) -> DocumentMetadata:
    """Build document metadata from deterministic intro parsing only."""
    # Extract manifest data
    episode_date = ""
    episode_description = ""
    episode_url = ""
    url_slug = ""

    if manifest_entry:
        episode_date = manifest_entry.date
        episode_description = manifest_entry.description
        episode_url = manifest_entry.transcript_url
        url_slug = manifest_entry.url_slug

    return DocumentMetadata(
        show_title="Gil's Arena",
        episode_type=intro_window.episode_type,
        host_names=tuple(intro_window.host_names),
        guest_names=tuple(intro_window.guest_names),
        episode_date=episode_date,
        episode_description=episode_description,
        episode_url=episode_url,
        url_slug=url_slug,
        season=None,
        episode_number=None,
        holiday_theme=intro_window.holiday_theme,
        topic=intro_window.topic_rundown[:200] if intro_window.topic_rundown else "",
        subtopic="",
        content_type=intro_window.content_type,
        episode_type_anchor=intro_window.episode_type_anchor,
        host_names_anchor=intro_window.host_names_anchor,
        guest_names_anchor=intro_window.guest_names_anchor,
        holiday_theme_anchor=intro_window.holiday_theme_anchor,
        topic_anchor=intro_window.topic_anchor,
        episode_roster=tuple(intro_window.episode_roster),
    )


def _parse_llm_response(raw_response: str) -> dict:
    """Parse LLM response as JSON, stripping markdown fences if present."""
    text = raw_response.strip()

    # Strip markdown code fences
    if text.startswith("```"):
        lines = text.split("\n")
        # Find first and last fence
        start_idx = 1  # Skip first fence
        end_idx = len(lines)
        for i, line in enumerate(lines):
            if i > 0 and line.strip().startswith("```"):
                end_idx = i
                break
        text = "\n".join(lines[start_idx:end_idx])

    # Try to find JSON object in the text
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1:
        text = text[start : end + 1]

    return json.loads(text)


def _merge_extraction_results(
    document: TranscriptDocument,
    intro_window: IntroWindow,
    llm_data: dict,
    manifest_entry: EpisodeManifestEntry | None,
) -> DocumentMetadata:
    """Merge deterministic intro parsing with LLM extraction results."""
    # Prefer LLM values where they are non-empty, fallback to deterministic
    episode_type_str = llm_data.get("episode_type", "") or intro_window.episode_type.value
    episode_type = _parse_episode_type(episode_type_str)

    host_names = llm_data.get("host_names") or intro_window.host_names
    if not host_names:
        host_names = intro_window.host_names

    guest_names = llm_data.get("guest_names") or intro_window.guest_names
    if not guest_names:
        guest_names = intro_window.guest_names

    holiday_theme = llm_data.get("holiday_theme", "") or intro_window.holiday_theme
    topic = llm_data.get("topic", "") or intro_window.topic_rundown
    subtopic = llm_data.get("subtopic", "")

    content_type_str = llm_data.get("content_type", "") or intro_window.content_type.value
    content_type = _parse_content_type(content_type_str)

    # Episode roster from LLM or built from intro
    episode_roster = llm_data.get("episode_roster") or intro_window.episode_roster
    if not episode_roster:
        episode_roster = intro_window.episode_roster

    # Extract manifest data
    episode_date = ""
    episode_description = ""
    episode_url = ""
    url_slug = ""

    if manifest_entry:
        episode_date = manifest_entry.date
        episode_description = manifest_entry.description
        episode_url = manifest_entry.transcript_url
        url_slug = manifest_entry.url_slug

    # Build anchors
    episode_anchor = None
    if intro_window.lines:
        first_line = intro_window.lines[0]
        episode_anchor = DocumentAnchor(
            line_index=first_line.start_seconds,
            timestamp=first_line.timestamp,
            excerpt=first_line.text[:100],
        )

    return DocumentMetadata(
        show_title="Gil's Arena",
        episode_type=episode_type,
        host_names=tuple(host_names),
        guest_names=tuple(guest_names),
        episode_date=episode_date,
        episode_description=episode_description,
        episode_url=episode_url,
        url_slug=url_slug,
        season=None,
        episode_number=None,
        holiday_theme=holiday_theme,
        topic=topic[:200] if topic else "",
        subtopic=subtopic,
        content_type=content_type,
        episode_type_anchor=intro_window.episode_type_anchor or episode_anchor,
        host_names_anchor=intro_window.host_names_anchor or episode_anchor,
        guest_names_anchor=intro_window.guest_names_anchor,
        holiday_theme_anchor=intro_window.holiday_theme_anchor,
        topic_anchor=intro_window.topic_anchor or episode_anchor,
        episode_roster=tuple(episode_roster),
    )


def _parse_episode_type(value: str) -> EpisodeType:
    """Parse episode type string to enum, defaulting to REACTION."""
    try:
        return EpisodeType(value)
    except ValueError:
        return EpisodeType.REACTION


def _parse_content_type(value: str) -> ContentType:
    """Parse content type string to enum, defaulting to MAIN_DISCUSSION."""
    try:
        return ContentType(value)
    except ValueError:
        return ContentType.MAIN_DISCUSSION
