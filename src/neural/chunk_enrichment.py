"""Chunk-level enrichment with roster-constrained speaker attribution.

Runs structured-output prompts on chunk text plus compact document context
to populate chunk fields and their anchors.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from neural.chunking import TranscriptChunk
from neural.metadata_types import (
    ChunkAnchor,
    ChunkMetadata,
    ClaimType,
    ContentType,
    ConversationType,
    DocumentMetadata,
    EmotionalIntensity,
    EnrichedChunk,
    KeyMomentType,
    LlmMetadata,
    SourceConfidence,
    Stance,
    TopicCategory,
)
from neural.openrouter import OpenRouterError, complete_chat

# ============================================================================
# Configuration
# ============================================================================

CHUNK_ENRICHMENT_SYSTEM_PROMPT = """\
You are a transcript metadata enricher for the "Gil's Arena" NBA podcast.

Your task is to extract structured metadata from a single chunk of podcast transcript.

CRITICAL RULES:
1. For speaker attribution, you MUST choose from the closed episode roster provided.
   Do NOT invent speaker names. If you cannot determine the speaker, use the first roster member.
2. Use controlled vocabularies exactly as specified.
3. Provide evidence excerpts for speaker and topic attributions.
4. Be conservative: prefer "descriptive" stance and "medium" confidence unless clearly indicated.
5. For conversation type, classify the dialogue mode in this chunk.
6. For claim type, classify the primary statement being made.
7. Mentions: extract people, teams, and leagues explicitly named in the text.
8. topic_category: classify the primary domain (player, team, league, event, business, etc.).
9. controversy_signals: detect heated/controversial language (disagree, hate, terrible, wrong take, etc.).
10. key_moment_type: identify NBA moment type if applicable (trade, injury, breakout, award, historic, etc.).
11. emotional_intensity: rate energy level as low, medium, or high.

Controlled vocabularies:
- conversation_type: debate, analysis, reaction, banter, interview, news_roundup, promotion
- claim_type: fact, opinion, prediction, rumor, anecdote, promotion
- content_type: ad_read, cold_open_banter, cast_intro, topic_rundown,
  main_discussion, guest_interview, promo_read, outro
- stance: supportive, critical, mixed, skeptical, descriptive
- source_confidence: high, medium, low
- topic_category: team, player, league, event, business, social_justice, show_meta, sponsor, holiday
- key_moment_type: trade, injury, breakout, award, historic, suspension, contract, draft, retirement, game_result, season_preview, playoff, anthem
- emotional_intensity: low, medium, high
"""

CHUNK_ENRICHMENT_USER_TEMPLATE = """\
Enrich this chunk with metadata.

EPISODE CONTEXT:
- Episode: {episode_title}
- Episode Type: {episode_type}
- Holiday Theme: {holiday_theme}
- Episode Topic: {episode_topic}
- Episode Roster: {roster}

CHUNK #{chunk_index} ({start_ts} - {end_ts}):
---
{chunk_text}
---

Return JSON with all required fields. Use empty arrays/strings where no \
data found.
Speaker MUST be from the roster:
{roster_json}\
"""


# ============================================================================
# Enrichment Functions
# ============================================================================


@dataclass(frozen=True)
class ChunkEnrichmentResult:
    """Result of chunk-level enrichment."""

    enriched_chunk: EnrichedChunk
    raw_response: str
    used_llm: bool


def enrich_chunk(
    chunk: TranscriptChunk,
    doc_metadata: DocumentMetadata,
    chunk_index: int,
    *,
    api_key: str | None = None,
    model: str | None = None,
    skip_llm: bool = False,
) -> ChunkEnrichmentResult:
    """Enrich a single chunk with metadata.

    Pipeline:
    1. Build deterministic chunk metadata (position, mentions)
    2. Run LLM extraction on chunk text (if not skipped)
    3. Merge deterministic and LLM results into EnrichedChunk

    Args:
        chunk: Base transcript chunk.
        doc_metadata: Extracted document-level metadata.
        chunk_index: Index of this chunk in the episode.
        api_key: OpenRouter API key (reads from env if None).
        model: OpenRouter model override.
        skip_llm: If True, skip LLM extraction (use deterministic only).

    Returns:
        ChunkEnrichmentResult with enriched chunk and raw response.
    """
    # Step 1: Build deterministic chunk metadata
    chunk_meta = _build_deterministic_chunk_meta(chunk, chunk_index)

    # Step 2: LLM enrichment (if not skipped)
    if skip_llm or not api_key:
        enriched = _build_enriched_chunk(
            chunk=chunk,
            chunk_meta=chunk_meta,
            llm_meta=_default_llm_meta(),
            doc_metadata=doc_metadata,
        )
        return ChunkEnrichmentResult(
            enriched_chunk=enriched,
            raw_response="",
            used_llm=False,
        )

    # Build prompt
    roster = list(doc_metadata.episode_roster) or [
        "Gilbert Arenas",
        "Nick Young",
        "Brandon Jennings",
    ]
    roster_json = json.dumps(roster)

    user_message = CHUNK_ENRICHMENT_USER_TEMPLATE.format(
        episode_title=chunk.episode_title,
        episode_type=doc_metadata.episode_type.value,
        holiday_theme=doc_metadata.holiday_theme or "none",
        episode_topic=doc_metadata.topic or "general discussion",
        roster=", ".join(roster),
        chunk_index=chunk_index,
        start_ts=chunk.start_timestamp,
        end_ts=chunk.end_timestamp,
        chunk_text=chunk.chunk_text[:1500],  # Limit chunk text length
        roster_json=roster_json,
    )

    try:
        raw_response = complete_chat(
            messages=[
                {"role": "system", "content": CHUNK_ENRICHMENT_SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            api_key=api_key,
            model=model,
            temperature=0.1,
            timeout_seconds=60.0,
        )
    except OpenRouterError:
        enriched = _build_enriched_chunk(
            chunk=chunk,
            chunk_meta=chunk_meta,
            llm_meta=_default_llm_meta(),
            doc_metadata=doc_metadata,
        )
        return ChunkEnrichmentResult(
            enriched_chunk=enriched,
            raw_response="",
            used_llm=False,
        )

    # Parse LLM response
    try:
        llm_data = _parse_llm_response(raw_response)
    except (json.JSONDecodeError, KeyError):
        enriched = _build_enriched_chunk(
            chunk=chunk,
            chunk_meta=chunk_meta,
            llm_meta=_default_llm_meta(),
            doc_metadata=doc_metadata,
        )
        return ChunkEnrichmentResult(
            enriched_chunk=enriched,
            raw_response=raw_response,
            used_llm=False,
        )

    # Build LLM metadata from parsed response
    llm_meta = _build_llm_meta(llm_data, chunk, roster)

    enriched = _build_enriched_chunk(
        chunk=chunk,
        chunk_meta=chunk_meta,
        llm_meta=llm_meta,
        doc_metadata=doc_metadata,
    )

    return ChunkEnrichmentResult(
        enriched_chunk=enriched,
        raw_response=raw_response,
        used_llm=True,
    )


def enrich_chunks(
    chunks: list[TranscriptChunk],
    doc_metadata: DocumentMetadata,
    *,
    api_key: str | None = None,
    model: str | None = None,
    skip_llm: bool = False,
) -> list[ChunkEnrichmentResult]:
    """Enrich all chunks in a transcript.

    Args:
        chunks: List of transcript chunks.
        doc_metadata: Extracted document-level metadata.
        api_key: OpenRouter API key.
        model: OpenRouter model override.
        skip_llm: If True, skip LLM extraction.

    Returns:
        List of ChunkEnrichmentResult, one per chunk.
    """
    results = []
    for i, chunk in enumerate(chunks):
        result = enrich_chunk(
            chunk=chunk,
            doc_metadata=doc_metadata,
            chunk_index=i,
            api_key=api_key,
            model=model,
            skip_llm=skip_llm,
        )
        results.append(result)
    return results


# ============================================================================
# Internal Helpers
# ============================================================================


def _build_deterministic_chunk_meta(
    chunk: TranscriptChunk,
    chunk_index: int,
) -> ChunkMetadata:
    """Build deterministic chunk metadata from position and basic matching."""
    content_type = _classify_chunk_content_type(chunk, chunk_index)
    conversation_type = _classify_conversation_type(chunk.chunk_text)
    claim_type = _classify_claim_type(chunk.chunk_text)
    topic_category = _detect_topic_category(chunk.chunk_text)
    controversy_signals = _detect_controversy(chunk.chunk_text)
    key_moment_type = _detect_key_moment(chunk.chunk_text)
    emotional_intensity = _classify_emotional_intensity(chunk.chunk_text)

    return ChunkMetadata(
        chunk_index=chunk_index,
        adjacent_context_window="",
        mentioned_people=(),
        mentioned_teams=(),
        mentioned_leagues=(),
        conversation_type=conversation_type,
        claim_type=claim_type,
        content_type=content_type,
        topic_category=topic_category,
        controversy_signals=tuple(controversy_signals),
        key_moment_type=key_moment_type,
        emotional_intensity=emotional_intensity,
    )


def _classify_chunk_content_type(chunk: TranscriptChunk, chunk_index: int) -> ContentType:
    """Classify content type based on chunk position and text."""
    text_lower = chunk.chunk_text.lower()

    # First few chunks are likely intro/ad-read/cast-intro
    if chunk_index <= 1:
        if any(s in text_lower for s in ["promo code", "download", "bonus", "underdog"]):
            return ContentType.PROMO_READ
        if any(s in text_lower for s in ["sponsored", "brought to you", "ad"]):
            return ContentType.AD_READ
        return ContentType.COLD_OPEN_BANTER

    if chunk_index <= 3:
        if any(s in text_lower for s in ["welcome back", "here with us", "in the building"]):
            return ContentType.CAST_INTRO
        if any(s in text_lower for s in ["here's what's cracking", "topics"]):
            return ContentType.TOPIC_RUNDOWN

    # Last chunks are likely outro
    if "thanks for watching" in text_lower or "see you next" in text_lower:
        return ContentType.OUTRO

    return ContentType.MAIN_DISCUSSION


def _classify_conversation_type(text: str) -> ConversationType:
    """Classify conversation type from chunk text."""
    text_lower = text.lower()

    if any(w in text_lower for w in ["debate", "argue", "disagree", "versus", "vs"]):
        return ConversationType.DEBATE
    if any(w in text_lower for w in ["analyze", "breakdown", "statistics", "numbers"]):
        return ConversationType.ANALYSIS
    if any(w in text_lower for w in ["react", "wow", "oh my", "shocking"]):
        return ConversationType.REACTION
    if any(w in text_lower for w in ["interview", "ask", "tell us about"]):
        return ConversationType.INTERVIEW
    if any(w in text_lower for w in ["news", "report", "announced"]):
        return ConversationType.NEWS_ROUNDUP
    if any(w in text_lower for w in ["promo", "download", "code", "sponsor"]):
        return ConversationType.PROMOTION

    return ConversationType.BANTER


def _classify_claim_type(text: str) -> ClaimType:
    """Classify claim type from chunk text."""
    text_lower = text.lower()

    if any(w in text_lower for w in ["predict", "going to", "will be", "I think"]):
        return ClaimType.PREDICTION
    if any(w in text_lower for w in ["rumor", "reportedly", "sources say", "heard"]):
        return ClaimType.RUMOR
    if any(w in text_lower for w in ["story", "remember when", "one time", "back in"]):
        return ClaimType.ANECDOTE
    if any(w in text_lower for w in ["fact", "actually", "statistics", "data"]):
        return ClaimType.FACT

    return ClaimType.OPINION


def _detect_topic_category(text: str) -> TopicCategory:
    text_lower = text.lower()
    if any(w in text_lower for w in ["trade", "deal", "acquired", "traded for"]):
        return TopicCategory.TEAM
    if any(w in text_lower for w in ["player", "career", "stats", "game log"]):
        return TopicCategory.PLAYER
    if any(w in text_lower for w in ["league", "nba", "commissioner", " CBA"]):
        return TopicCategory.LEAGUE
    if any(w in text_lower for w in ["event", "all-star", "draft", "playoffs"]):
        return TopicCategory.EVENT
    if any(w in text_lower for w in ["contract", "salary", "money", "pay", "million"]):
        return TopicCategory.BUSINESS
    if any(w in text_lower for w in ["social justice", "community", "activism"]):
        return TopicCategory.SOCIAL_JUSTICE
    if any(w in text_lower for w in ["halloween", "christmas", "holiday", "juneteenth"]):
        return TopicCategory.HOLIDAY
    if any(w in text_lower for w in ["sponsor", "ad", "promo"]):
        return TopicCategory.SPONSOR
    return TopicCategory.PLAYER


def _detect_controversy(text: str) -> list[str]:
    text_lower = text.lower()
    found = []
    for signal in [
        "disagree",
        "wrong take",
        "overrated",
        "underrated",
        "hate",
        "stupid",
        "terrible",
        "awful",
        "embarrassing",
        "should be traded",
        "overpaid",
        "bust",
        "despise",
        "foolish",
    ]:
        if signal in text_lower:
            found.append(signal)
    return found[:5]


def _detect_key_moment(text: str) -> KeyMomentType | None:
    text_lower = text.lower()
    if any(w in text_lower for w in ["trade", "traded", "deal", "acquired"]):
        return KeyMomentType.TRADE
    if any(w in text_lower for w in ["injury", "hurt", "out for", "missed"]):
        return KeyMomentType.INJURY
    if any(w in text_lower for w in ["breakout", "emerged", "came into his own"]):
        return KeyMomentType.BREAKOUT
    if any(w in text_lower for w in ["mvp", "award", "all-nba", "rookie of the year"]):
        return KeyMomentType.AWARD
    if any(w in text_lower for w in ["historic", "record", "first time", "all-time"]):
        return KeyMomentType.HISTORIC
    if any(w in text_lower for w in ["suspended", "suspension"]):
        return KeyMomentType.SUSPENSION
    if any(w in text_lower for w in ["contract", "signed", "extension", "salary"]):
        return KeyMomentType.CONTRACT
    if any(w in text_lower for w in ["draft", "lottery", "pick"]):
        return KeyMomentType.DRAFT
    if any(w in text_lower for w in ["retire", "retirement", "legacy"]):
        return KeyMomentType.RETIREMENT
    if any(w in text_lower for w in ["playoff", "finals", "game 7", "elimination"]):
        return KeyMomentType.PLAYOFF
    if any(w in text_lower for w in ["score", "win", "loss", "final", "beat"]):
        return KeyMomentType.GAME_RESULT
    return None


def _classify_emotional_intensity(text: str) -> EmotionalIntensity:
    text_lower = text.lower()
    exclamations = text.count("!")
    caps_words = sum(1 for w in text.split() if w.isupper() and len(w) > 1)
    heated = any(
        w in text_lower for w in ["hate", "stupid", "terrible", "awful", "disgrace", "embarrassing"]
    )
    if exclamations >= 3 or caps_words >= 3 or heated:
        return EmotionalIntensity.HIGH
    if exclamations >= 1 or any(
        w in text_lower for w in ["wow", "shocking", "incredible", "crazy"]
    ):
        return EmotionalIntensity.MEDIUM
    return EmotionalIntensity.LOW


def _build_llm_meta(
    llm_data: dict,
    chunk: TranscriptChunk,
    roster: list[str],
) -> LlmMetadata:
    """Build LLM metadata from parsed response."""
    # Parse speaker, validate against roster
    speaker = llm_data.get("speaker", "")
    if speaker not in roster and roster:
        speaker = roster[0]  # Fallback to first roster member

    # Parse enums
    stance_str = llm_data.get("stance", "descriptive")
    stance = _parse_stance(stance_str)

    confidence_str = llm_data.get("source_confidence", "medium")
    confidence = _parse_confidence(confidence_str)

    # Build anchors
    speaker_anchor = None
    speaker_evidence = llm_data.get("speaker_evidence", "")
    if speaker_evidence:
        speaker_anchor = ChunkAnchor(
            chunk_index=0,  # Will be set by caller
            start_timestamp=chunk.start_timestamp,
            excerpt=speaker_evidence[:100],
        )

    topic_anchor = None
    topic_evidence = llm_data.get("topic_evidence", "")
    if topic_evidence:
        topic_anchor = ChunkAnchor(
            chunk_index=0,
            start_timestamp=chunk.start_timestamp,
            excerpt=topic_evidence[:100],
        )

    return LlmMetadata(
        speaker=speaker,
        topic=llm_data.get("topic", ""),
        subtopic=llm_data.get("subtopic", ""),
        sentiment=llm_data.get("sentiment", ""),
        stance=stance,
        source_confidence=confidence,
        speaker_anchor=speaker_anchor,
        topic_anchor=topic_anchor,
        sentiment_anchor=None,
    )


def _default_llm_meta() -> LlmMetadata:
    """Return default LLM metadata for when extraction is skipped."""
    return LlmMetadata(
        speaker="",
        topic="",
        subtopic="",
        sentiment="",
        stance=Stance.DESCRIPTIVE,
        source_confidence=SourceConfidence.LOW,
        speaker_anchor=None,
        topic_anchor=None,
        sentiment_anchor=None,
    )


def _build_enriched_chunk(
    chunk: TranscriptChunk,
    chunk_meta: ChunkMetadata,
    llm_meta: LlmMetadata,
    doc_metadata: DocumentMetadata,
) -> EnrichedChunk:
    """Build enriched chunk combining all metadata tiers."""
    return EnrichedChunk(
        # Base chunk fields
        episode_title=chunk.episode_title,
        source_file=chunk.source_file,
        start_timestamp=chunk.start_timestamp,
        end_timestamp=chunk.end_timestamp,
        start_seconds=chunk.start_seconds,
        end_seconds=chunk.end_seconds,
        chunk_text=chunk.chunk_text,
        line_count=chunk.line_count,
        # Tier 2
        chunk_meta=chunk_meta,
        # Tier 3
        llm_meta=llm_meta,
        # Inherited document fields
        show_title=doc_metadata.show_title,
        episode_type=doc_metadata.episode_type.value,
        host_names=doc_metadata.host_names,
        guest_names=doc_metadata.guest_names,
        episode_date=doc_metadata.episode_date,
        holiday_theme=doc_metadata.holiday_theme,
        episode_topic=doc_metadata.topic,
    )


def _parse_llm_response(raw_response: str) -> dict:
    """Parse LLM response as JSON, stripping markdown fences if present."""
    text = raw_response.strip()

    # Strip markdown code fences
    if text.startswith("```"):
        lines = text.split("\n")
        start_idx = 1
        end_idx = len(lines)
        for i, line in enumerate(lines):
            if i > 0 and line.strip().startswith("```"):
                end_idx = i
                break
        text = "\n".join(lines[start_idx:end_idx])

    # Find JSON object
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1:
        text = text[start : end + 1]

    return json.loads(text)


def _parse_stance(value: str) -> Stance:
    """Parse stance string to enum."""
    try:
        return Stance(value)
    except ValueError:
        return Stance.DESCRIPTIVE


def _parse_confidence(value: str) -> SourceConfidence:
    """Parse source confidence string to enum."""
    try:
        return SourceConfidence(value)
    except ValueError:
        return SourceConfidence.MEDIUM
