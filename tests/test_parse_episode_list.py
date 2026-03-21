"""Tests for parse_episode_list."""

from scrape_podscripts import parse_episode_list

LISTING_HTML = """
<html><body>
<div class="listing-item">
  <h3><a href="/podcasts/gils-arena/ep-one">First Episode</a></h3>
  <span class="episode_date">Episode Date: Jan 1, 2024</span>
  <div class="geodir-category-content"><p>Short blurb here.</p></div>
</div>
<div class="listing-item">
  <h3><a href="/podcasts/gils-arena/gridiron-special">Gridiron</a></h3>
</div>
<div class="listing-item">
  <h3><a href="/podcasts/gils-arena/ep-two">Second Episode</a></h3>
  <span class="episode_date">Episode Date: Feb 2, 2024</span>
</div>
</body></html>
"""


def test_parse_episode_list_skips_gridiron_and_extracts_metadata() -> None:
    episodes = parse_episode_list(LISTING_HTML, page_num=1)
    assert len(episodes) == 2
    assert episodes[0]["title"] == "First Episode"
    assert episodes[0]["url_slug"] == "ep-one"
    assert episodes[0]["date"] == "Jan 1, 2024"
    assert episodes[0]["description"] == "Short blurb here."
    assert episodes[1]["title"] == "Second Episode"
    assert episodes[1]["url_slug"] == "ep-two"
    assert episodes[1]["date"] == "Feb 2, 2024"
    assert episodes[1]["description"] == ""


def test_parse_episode_list_empty_when_no_items() -> None:
    assert parse_episode_list("<html><body></body></html>", page_num=1) == []
