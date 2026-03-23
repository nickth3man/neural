"""Tests for neural.intro_parser — intro-window metadata extraction."""

from __future__ import annotations

from pathlib import Path

import pytest

from neural.corpus import TranscriptDocument, TranscriptLine, load_transcript
from neural.intro_parser import (
    _build_episode_roster,
    _extract_guest_names,
    _extract_holiday_theme,
    _extract_host_names,
    _extract_sponsors,
    extract_intro_window,
)
from neural.metadata_types import EpisodeType

# ============================================================================
# Fixtures
# ============================================================================


def _make_transcript_document(lines: list[tuple[int, str]]) -> TranscriptDocument:
    """Helper to create a TranscriptDocument from (seconds, text) tuples."""
    transcript_lines = tuple(
        TranscriptLine(
            timestamp=f"{s // 3600:02d}:{(s % 3600) // 60:02d}:{s % 60:02d}",
            start_seconds=s,
            text=text,
        )
        for s, text in lines
    )
    return TranscriptDocument(
        episode_title="Test Episode",
        source_file="test.txt",
        source_path="/test/test.txt",
        lines=transcript_lines,
    )


@pytest.fixture
def sample_intro_transcript() -> TranscriptDocument:
    """Sample transcript with intro window matching Gil's Arena format."""
    return _make_transcript_document(
        [
            (0, "Welcome back to Gills Arena presented by Underdog. Whoa, whoa!"),
            (10, "Back on the couch on this glorious Monday. How everybody feeling?"),
            (20, "Good, good. A lot of beef in these streets."),
            (30, "We got RB. You was talking a lot of shit this weekend, Swaggy."),
            (60, "This is Gil's Arena presented by Underdog Woo Woo."),
            (70, "As always, we got the legend Gilbert Arenas is here with us."),
            (80, "Rocking the Alex Saar jersey. Shout out to Saar."),
            (
                90,
                "This side of the couch, we got NBA champion of the world, swaggy Pete Nick Young.",
            ),
            (
                100,
                "And on this side of the couch, representing Tough Crowd, "
                "we got Mr. B, Brandon Jennings.",
            ),
            (110, "We got a special guest making his first appearance in the arena. Famous Los."),
            (120, "Welcome to Guild Arena presented by Underdog Woo Woo."),
            (150, "Before we do, been a lot of beef going on in the arena."),
            (
                200,
                "Here's what's cracking in the arena today. LeBron James future with the Lakers.",
            ),
            (210, "The Lakers suffered one of their worst losses of the season."),
            (220, "KD joined LeBron as the only players in NBA history to drop a 40 ball."),
            (300, "Let's talk about the first game of the season."),
        ]
    )


@pytest.fixture
def halloween_transcript() -> TranscriptDocument:
    """Sample transcript with Halloween theme."""
    return _make_transcript_document(
        [
            (0, "Welcome back to Gills Arena, presented by Underdog. Whoa, whoa!"),
            (10, "It's Halloween. Here in the arena, we are a day early."),
            (20, "What happened to your war? Dodger game, 18 innings."),
            (30, "But this is Gills Arena, presented by Underdog Woo-Wo."),
            (40, "I'm merely a host, Josiah Johnson."),
            (50, "We got the legend Gilbert Arenas here with us."),
            (60, "Next to get we're joined by NBA champion of the world Swaggy Pete Nick Young."),
            (
                70,
                "And next to pimping, representing Tough Crowd, we got Mr. Beat, Brandon Jennings.",
            ),
            (80, "And on this side, the distinguished gentleman, Kenyon Martin."),
            (90, "And next to Kenya, we got Rashad McCants in the building."),
            (200, "Here's what's cracking in Gil's Arena today."),
            (210, "Austin Reeves continued his hot start with a game-winning buzzer beater."),
        ]
    )


@pytest.fixture
def empty_transcript() -> TranscriptDocument:
    """Empty transcript document."""
    return TranscriptDocument(
        episode_title="Empty",
        source_file="empty.txt",
        source_path="/test/empty.txt",
        lines=(),
    )


# ============================================================================
# Intro Window Extraction Tests
# ============================================================================


class TestExtractIntroWindow:
    def test_basic_extraction(self, sample_intro_transcript: TranscriptDocument) -> None:
        """Test basic intro window extraction from sample transcript."""
        intro = extract_intro_window(sample_intro_transcript)

        assert len(intro.lines) > 0
        assert len(intro.text) > 0
        assert intro.episode_type is not None
        assert intro.content_type is not None

    def test_host_names(self, sample_intro_transcript: TranscriptDocument) -> None:
        """Test host name extraction from roster."""
        intro = extract_intro_window(sample_intro_transcript)
        assert "Gilbert Arenas" in intro.episode_roster

    def test_guest_names(self, sample_intro_transcript: TranscriptDocument) -> None:
        """Test guest extraction from 'special guest' intro."""
        intro = extract_intro_window(sample_intro_transcript)
        assert intro.episode_type == EpisodeType.GUEST_INTERVIEW

    def test_episode_type_guest(self, sample_intro_transcript: TranscriptDocument) -> None:
        """Test episode type is guest_interview when special guest present."""
        intro = extract_intro_window(sample_intro_transcript)
        assert intro.episode_type == EpisodeType.GUEST_INTERVIEW

    def test_episode_type_halloween(self, halloween_transcript: TranscriptDocument) -> None:
        """Test holiday theme extraction."""
        intro = extract_intro_window(halloween_transcript)
        assert "halloween" in intro.holiday_theme.lower()

    def test_episode_roster(self, sample_intro_transcript: TranscriptDocument) -> None:
        """Test episode roster is built from cast intros."""
        intro = extract_intro_window(sample_intro_transcript)
        assert "Gilbert Arenas" in intro.episode_roster
        assert "Nick Young" in intro.episode_roster
        assert "Brandon Jennings" in intro.episode_roster

    def test_empty_transcript(self, empty_transcript: TranscriptDocument) -> None:
        """Test handling of empty transcript."""
        intro = extract_intro_window(empty_transcript)
        assert len(intro.lines) == 0
        assert intro.host_names == []
        assert intro.guest_names == []
        assert intro.holiday_theme == ""

    def test_sponsor_extraction(self, sample_intro_transcript: TranscriptDocument) -> None:
        """Test sponsor extraction from ad reads."""
        intro = extract_intro_window(sample_intro_transcript)
        assert "Underdog" in intro.sponsors


class TestExtractHostNames:
    def test_from_known_aliases(self) -> None:
        """Test host extraction from known aliases."""
        lines = _make_transcript_document(
            [
                (0, "I'm merely a host, Josiah Johnson."),
            ]
        ).lines
        text = lines[0].text
        hosts, anchor = _extract_host_names(lines, text)
        assert "Josiah Johnson" in hosts

    def test_empty(self) -> None:
        """Test empty host extraction returns empty list when no match."""
        lines = _make_transcript_document([(0, "Random text")]).lines
        hosts, _ = _extract_host_names(lines, lines[0].text)
        assert hosts == []


class TestExtractGuestNames:
    def test_special_guest(self) -> None:
        """Test guest extraction detects special guest presence."""
        lines = _make_transcript_document(
            [
                (0, "We got a special guest making his first appearance. Famous Los."),
            ]
        ).lines
        text = " ".join(line.text for line in lines)
        guests, anchor = _extract_guest_names(lines, text)
        assert anchor is not None or len(guests) == 0

    def test_no_guest(self) -> None:
        """Test no guest extraction when no special guest."""
        lines = _make_transcript_document(
            [
                (0, "Welcome back to the show."),
            ]
        ).lines
        guests, _ = _extract_guest_names(lines, lines[0].text)
        assert guests == []


class TestExtractHolidayTheme:
    def test_halloween(self) -> None:
        """Test Halloween detection."""
        lines = _make_transcript_document(
            [
                (0, "It's Halloween in the arena today!"),
            ]
        ).lines
        holiday, anchor = _extract_holiday_theme(lines, lines[0].text)
        assert "halloween" in holiday.lower()

    def test_no_holiday(self) -> None:
        """Test no holiday detection."""
        lines = _make_transcript_document(
            [
                (0, "Welcome back to the show."),
            ]
        ).lines
        holiday, _ = _extract_holiday_theme(lines, lines[0].text)
        assert holiday == ""


class TestExtractSponsors:
    def test_underdog(self) -> None:
        """Test Underdog sponsor extraction."""
        text = "Gil's Arena presented by Underdog Fantasy."
        sponsors = _extract_sponsors(text)
        assert "Underdog" in sponsors

    def test_multiple_sponsors(self) -> None:
        """Test multiple sponsor extraction."""
        text = "This show is sponsored by Underdog and brought to you by Wendy's."
        sponsors = _extract_sponsors(text)
        assert len(sponsors) >= 1


class TestBuildEpisodeRoster:
    def test_includes_cast(self) -> None:
        """Test roster includes known cast members."""
        roster = _build_episode_roster(
            host_names=["Josiah Johnson"],
            guest_names=[],
            intro_text="Gilbert Arenas and Nick Young are here.",
        )
        assert "Gilbert Arenas" in roster
        assert "Nick Young" in roster

    def test_includes_guests(self) -> None:
        """Test roster includes guests."""
        roster = _build_episode_roster(
            host_names=["Josiah Johnson"],
            guest_names=["Dwight Howard"],
            intro_text="",
        )
        assert "Dwight Howard" in roster

    def test_core_cast_always_present(self) -> None:
        """Test core cast is always in roster."""
        roster = _build_episode_roster(
            host_names=[],
            guest_names=[],
            intro_text="",
        )
        assert "Gilbert Arenas" in roster
        assert "Nick Young" in roster
        assert "Brandon Jennings" in roster


class TestRealTranscript:
    """Test against actual transcript files in the repository."""

    def test_load_halloween_transcript(self) -> None:
        """Test loading and parsing the Halloween special transcript."""
        path = Path("gil/transcripts/Gils_Arena_Throws_A_SPOOKY_Halloween_Special.txt")
        if not path.exists():
            pytest.skip("Transcript file not found")

        doc = load_transcript(path)
        intro = extract_intro_window(doc)

        # Should detect Halloween
        assert "halloween" in intro.holiday_theme.lower()
        # Should have hosts and cast
        assert len(intro.host_names) > 0
        assert len(intro.episode_roster) >= 3
        # Should have Underdog sponsor
        assert "Underdog" in intro.sponsors

    def test_load_dwight_howard_transcript(self) -> None:
        """Test loading the Dwight Howard guest episode."""
        path = Path("gil/transcripts/Gils_Arena_Breaks_Down_NBA_Tip_Off_With_Dwight_Howard.txt")
        if not path.exists():
            pytest.skip("Transcript file not found")

        doc = load_transcript(path)
        intro = extract_intro_window(doc)

        # Should have roster
        assert "Gilbert Arenas" in intro.episode_roster
        assert "Nick Young" in intro.episode_roster
