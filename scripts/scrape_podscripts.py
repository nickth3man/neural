#!/usr/bin/env python3
"""
PodScripts.co scraper for Gil's Arena podcast transcripts.

Respects robots.txt, rate limits requests, and stores transcripts as text files.
"""

import argparse
import html
import logging
import re
import sys
import time
import urllib.robotparser
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TypedDict

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm


# ============================================================================
# Type Definitions
# ============================================================================


class Episode(TypedDict):
    """Structured representation of an episode's metadata."""

    title: str
    url_slug: str
    date: str
    description: str


class ScrapeResult(TypedDict):
    """Result summary of a scraping run."""

    total_found: int
    skipped_existing: int
    succeeded: int
    failed: int
    failures: list[str]


# ============================================================================
# Configuration
# ============================================================================


@dataclass(frozen=True)
class ScraperConfig:
    """Immutable configuration for the scraper."""

    base_url: str = "https://podscripts.co"
    podcast_slug: str = "gils-arena"
    user_agent: str = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    rate_limit_seconds: int = 5
    max_retries: int = 3
    request_timeout: int = 30
    robots_timeout: int = 10
    total_pages: int = 22
    dry_run_episodes: int = 3
    failure_log_path: Path = field(default_factory=lambda: Path("data/scrape_failures.log"))
    default_transcripts_dir: Path = field(default_factory=lambda: Path("gil/transcripts"))

    @property
    def episode_list_url(self) -> str:
        """Generate the episode list URL from base components."""
        return f"{self.base_url}/podcasts/{self.podcast_slug}"

    @property
    def robots_url(self) -> str:
        """Generate the robots.txt URL."""
        return f"{self.base_url}/robots.txt"


# Module-level configuration instance
CONFIG = ScraperConfig()

# ============================================================================
# Logging Setup
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ============================================================================
# Exceptions
# ============================================================================


class RobotsBlockedError(Exception):
    """Raised when scraping is blocked by robots.txt."""

    pass


# ============================================================================
# Utility Functions
# ============================================================================


def check_robots_txt(session: requests.Session) -> bool:
    """
    Check robots.txt to ensure scraping is allowed using the standard library parser.

    Args:
        session: Active requests session for making HTTP calls.

    Returns:
        True if scraping is allowed, False otherwise.

    Raises:
        RobotsBlockedError: If scraping is explicitly blocked by robots.txt.
    """
    rp = urllib.robotparser.RobotFileParser()
    rp.set_url(CONFIG.robots_url)

    try:
        response = session.get(CONFIG.robots_url, timeout=CONFIG.robots_timeout)
        response.raise_for_status()
        rp.parse(response.text.splitlines())
        logger.debug("Successfully fetched and parsed robots.txt")
    except requests.RequestException as e:
        logger.warning(f"Could not fetch robots.txt: {e}. Proceeding with caution.")
        return True

    # Check if our user agent can fetch the podcast listing URL
    if not rp.can_fetch(CONFIG.user_agent, CONFIG.episode_list_url):
        logger.error("Scraping is blocked by robots.txt")
        return False

    logger.info("Scraping is allowed by robots.txt")
    return True


def get_existing_transcripts(output_dir: Path) -> set[str]:
    """
    Return transcript stems already present in the output directory.

    Args:
        output_dir: Directory to scan for existing transcripts.

    Returns:
        Set of filename stems (without .txt extension) for existing transcripts.
    """
    if not output_dir.exists():
        return set()

    return {path.stem for path in output_dir.glob("*.txt") if path.is_file()}


def fetch_with_retry(
    url: str,
    session: requests.Session,
    timeout: int | None = None,
) -> requests.Response | None:
    """
    Fetch URL with exponential backoff retry logic.

    Args:
        url: The URL to fetch.
        session: Active requests session for making HTTP calls.
        timeout: Request timeout in seconds. Uses config default if None.

    Returns:
        The response object if successful, None if all retries failed.
    """
    timeout = timeout or CONFIG.request_timeout

    for attempt in range(CONFIG.max_retries):
        try:
            response = session.get(url, timeout=timeout)
            response.raise_for_status()
            return response
        except requests.RequestException as e:
            wait_time = CONFIG.rate_limit_seconds * (2**attempt)
            logger.warning(
                f"Attempt {attempt + 1}/{CONFIG.max_retries} failed for {url}: {e}"
            )
            if attempt < CONFIG.max_retries - 1:
                logger.info(f"Waiting {wait_time}s before retry...")
                time.sleep(wait_time)
            else:
                log_failure(url, str(e))
                return None

    return None


def log_failure(url: str, error: str) -> None:
    """
    Log failed requests to failure log file.

    Args:
        url: The URL that failed.
        error: Error message describing the failure.
    """
    CONFIG.failure_log_path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().isoformat()

    with open(CONFIG.failure_log_path, "a", encoding="utf-8") as f:
        f.write(f"{timestamp} | {url} | {error}\n")


def sanitize_filename(title: str) -> str:
    """
    Convert episode title to a safe filename with underscores.

    Args:
        title: The episode title to sanitize.

    Returns:
        A filesystem-safe filename string.
    """
    # Remove characters that are problematic in filenames
    sanitized = re.sub(r'[<>:"/\\|?*\']', "", title)
    # Replace whitespace and hyphens with underscores
    sanitized = re.sub(r"[\s\-]+", "_", sanitized)
    # Collapse multiple underscores
    sanitized = re.sub(r"_+", "_", sanitized)
    # Strip leading/trailing underscores
    return sanitized.strip("_")


def transcript_output_path(episode_title: str, output_dir: Path) -> Path:
    """
    Build the output path for an episode transcript.

    Args:
        episode_title: Title of the episode.
        output_dir: Directory where transcripts are stored.

    Returns:
        Path object for the transcript file.
    """
    filename = sanitize_filename(episode_title)
    if not filename:
        filename = f"episode_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    return output_dir / f"{filename}.txt"


def normalize_line_text(text: str, separator: str = " ") -> str:
    """
    Normalize HTML transcript text while preserving the requested separator.

    Args:
        text: Raw text from HTML element.
        separator: Character(s) to replace whitespace with.

    Returns:
        Normalized text string.
    """
    normalized = html.unescape(text or "")
    normalized = normalized.replace("\xa0", " ")
    normalized = re.sub(r"\s+", separator, normalized)
    return normalized.strip()


def save_transcript_as_text(
    episode_title: str,
    transcript_lines: list[str],
    output_dir: Path,
) -> bool:
    """
    Save episode transcript as plain text matching the existing folder format.

    Args:
        episode_title: Title of the episode.
        transcript_lines: List of transcript text lines.
        output_dir: Directory to save the transcript in.

    Returns:
        True if save was successful, False otherwise.
    """
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        filepath = transcript_output_path(episode_title, output_dir)

        # Filter out empty lines
        lines = [line for line in transcript_lines if line]

        with open(filepath, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        logger.info(f"Saved transcript to {filepath}")
        return True

    except OSError as e:
        logger.error(f"Failed to save transcript as text: {e}")
        return False


def parse_episode_list(html_content: str, page_num: int) -> list[Episode]:
    """
    Parse episode list page and extract episode metadata.

    Args:
        html_content: HTML content of the episode list page.
        page_num: Page number for logging purposes.

    Returns:
        List of Episode dictionaries with metadata.
    """
    soup = BeautifulSoup(html_content, "html.parser")
    episodes: list[Episode] = []

    for item in soup.find_all("div", class_="listing-item"):
        try:
            # Extract title and link
            title_header = item.find("h3")
            title_link = title_header.find("a") if title_header else None
            if not title_link:
                continue

            title = title_link.get_text(strip=True)
            href = title_link.get("href", "")
            if not isinstance(href, str):
                continue

            # Extract slug from URL path
            parts = href.split("/")
            if len(parts) < 4 or not parts[-1]:
                continue
            url_slug = parts[-1]

            # Skip Gridiron episodes
            if "gridiron" in url_slug.lower():
                logger.debug(f"Skipping Gridiron episode: {url_slug}")
                continue

            # Extract date
            date_str = ""
            date_span = item.find("span", class_="episode_date")
            if date_span:
                date_text = date_span.get_text(strip=True)
                match = re.search(r"Episode Date:\s*(.+)", date_text)
                if match:
                    date_str = match.group(1).strip()

            # Extract description
            content_div = item.find("div", class_="geodir-category-content")
            desc_p = content_div.find("p") if content_div else None
            description = desc_p.get_text(strip=True) if desc_p else ""

            episodes.append(
                Episode(
                    title=title,
                    url_slug=url_slug,
                    date=date_str,
                    description=description,
                )
            )

        except Exception as e:
            logger.warning(f"Error parsing episode item on page {page_num}: {e}")
            continue

    logger.info(f"Found {len(episodes)} episodes on page {page_num}")
    return episodes


def parse_transcript_page(html_content: str) -> list[str]:
    """
    Parse transcript page and extract transcript lines.

    Args:
        html_content: HTML content of the transcript page.

    Returns:
        List of transcript text lines matching the existing plain text format.
    """
    soup = BeautifulSoup(html_content, "html.parser")
    transcript_lines: list[str] = []

    transcript_div = soup.find("div", class_="podcast-transcript")
    if not transcript_div:
        logger.warning("No transcript container found")
        return transcript_lines

    # Process different element types in the transcript
    for element in transcript_div.find_all(["div", "p", "h1", "h2", "h3", "h4"]):
        try:
            raw_classes = element.get("class")
            classes = set(raw_classes) if isinstance(raw_classes, list) else set()

            # Handle single-sentence elements (primary transcript content)
            if "single-sentence" in classes:
                line = normalize_line_text(element.get_text(strip=True))
                if line:
                    transcript_lines.append(line)
                continue

            # Handle chapter headings
            line = normalize_line_text(element.get_text(" ", strip=True))
            if line.lower().startswith("chapter "):
                transcript_lines.append(line)

        except Exception as e:
            logger.warning(f"Error parsing transcript element: {e}")
            continue

    logger.info(f"Extracted {len(transcript_lines)} transcript lines")
    return transcript_lines


def scrape_episode(
    episode: Episode,
    session: requests.Session,
    output_dir: Path,
) -> bool:
    """
    Scrape a single episode's transcript and store it as a text file.

    Args:
        episode: Episode metadata dictionary.
        session: Active requests session for making HTTP calls.
        output_dir: Directory to save the transcript in.

    Returns:
        True if scraping and saving succeeded, False otherwise.
    """
    transcript_url = f"{CONFIG.base_url}/podcasts/{CONFIG.podcast_slug}/{episode['url_slug']}"

    logger.info(f"Scraping episode: {episode['title']}")

    response = fetch_with_retry(transcript_url, session)
    if not response:
        return False

    transcript_lines = parse_transcript_page(response.text)

    if not transcript_lines:
        logger.warning(f"No transcript lines found for episode: {episode['url_slug']}")
        log_failure(transcript_url, "No transcript lines extracted")
        return False

    if save_transcript_as_text(episode["title"], transcript_lines, output_dir):
        logger.info(
            f"Saved {len(transcript_lines)} lines for episode: {episode['title']}"
        )
        return True

    log_failure(transcript_url, "Failed to write transcript file")
    return False


def run_scraper(
    dry_run: bool = False,
    resume: bool = False,
    start_page: int = 1,
    end_page: int | None = None,
    limit: int | None = None,
    output_dir: Path = CONFIG.default_transcripts_dir,
) -> ScrapeResult:
    """
    Main scraper orchestrator.

    Args:
        dry_run: If True, only scrape a small number of episodes for testing.
        resume: If True, skip episodes that already have transcript files.
        start_page: First page number to scrape (1-indexed).
        end_page: Last page number to scrape. Uses config default if None.
        limit: Maximum number of episodes to scrape. No limit if None.
        output_dir: Directory to save transcript files in.

    Returns:
        ScrapeResult dictionary with summary statistics.

    Raises:
        RobotsBlockedError: If scraping is blocked by robots.txt.
    """
    logger.info("Checking robots.txt...")

    with requests.Session() as session:
        session.headers.update({"User-Agent": CONFIG.user_agent})

        if not check_robots_txt(session):
            logger.error("Scraping blocked by robots.txt. Exiting.")
            raise RobotsBlockedError("Scraping blocked by robots.txt")

        # Determine which transcripts already exist
        existing_transcripts = get_existing_transcripts(output_dir) if resume else set()
        if resume and existing_transcripts:
            logger.info(
                f"Resume mode: skipping {len(existing_transcripts)} existing transcript files"
            )

        # Set page range
        if end_page is None:
            end_page = 1 if dry_run else CONFIG.total_pages

        # Validate page range
        if start_page < 1:
            logger.warning(f"Invalid start_page {start_page}, using 1")
            start_page = 1
        if end_page < start_page:
            logger.warning(f"end_page ({end_page}) < start_page ({start_page}), adjusting")
            end_page = start_page

        # Fetch episode list
        all_episodes: list[Episode] = []
        logger.info(f"Fetching episode list from pages {start_page} to {end_page}")

        for page_num in tqdm(
            range(start_page, end_page + 1),
            desc="Fetching episode lists",
            disable=dry_run,
        ):
            page_url = f"{CONFIG.episode_list_url}?page={page_num}"
            response = fetch_with_retry(page_url, session)

            if not response:
                logger.error(f"Failed to fetch page {page_num}")
                continue

            episodes = parse_episode_list(response.text, page_num)
            all_episodes.extend(episodes)

            # Rate limit between page fetches
            if page_num < end_page:
                time.sleep(CONFIG.rate_limit_seconds)

        logger.info(f"Found {len(all_episodes)} total episodes")

        # Apply dry run limit
        if dry_run:
            all_episodes = all_episodes[: CONFIG.dry_run_episodes]
            logger.info(f"Dry-run mode: limiting to {CONFIG.dry_run_episodes} episodes")

        # Filter out existing transcripts in resume mode
        episodes_to_scrape = [
            ep
            for ep in all_episodes
            if sanitize_filename(ep["title"]) not in existing_transcripts
        ]

        skipped_existing = len(all_episodes) - len(episodes_to_scrape)

        # Apply episode limit
        if limit is not None and len(episodes_to_scrape) > limit:
            episodes_to_scrape = episodes_to_scrape[:limit]
            logger.info(f"Limit applied: scraping at most {limit} episodes")

        if not episodes_to_scrape:
            logger.info("No new episodes to scrape")
            return ScrapeResult(
                total_found=len(all_episodes),
                skipped_existing=skipped_existing,
                succeeded=0,
                failed=0,
                failures=[],
            )

        logger.info(f"Scraping {len(episodes_to_scrape)} episodes")

        # Scrape episodes
        success_count = 0
        failure_count = 0
        failure_urls: list[str] = []

        for i, episode in enumerate(
            tqdm(episodes_to_scrape, desc="Scraping transcripts", disable=False)
        ):
            if scrape_episode(episode, session, output_dir):
                success_count += 1
            else:
                failure_count += 1
                failure_urls.append(episode["url_slug"])

            # Rate limit between episode scrapes
            if i < len(episodes_to_scrape) - 1:
                time.sleep(CONFIG.rate_limit_seconds)

        logger.info(
            f"Scraping complete: {success_count} succeeded, {failure_count} failed"
        )

        if failure_count > 0:
            logger.warning(f"Failures logged to {CONFIG.failure_log_path}")

        return ScrapeResult(
            total_found=len(all_episodes),
            skipped_existing=skipped_existing,
            succeeded=success_count,
            failed=failure_count,
            failures=failure_urls,
        )


def main() -> None:
    """Parse command-line arguments and run the scraper."""
    parser = argparse.ArgumentParser(
        description="Scrape Gil's Arena transcripts from PodScripts.co",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=f"Scrape only {CONFIG.dry_run_episodes} episodes for testing",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip episodes whose transcript files already exist in the output directory",
    )
    parser.add_argument(
        "--start-page",
        type=int,
        default=1,
        help="Starting page number (1-indexed)",
    )
    parser.add_argument(
        "--end-page",
        type=int,
        default=None,
        help="Ending page number (defaults to total pages, or 1 in dry-run)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of episodes to scrape (no limit by default)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(CONFIG.default_transcripts_dir),
        help="Directory to save transcript text files",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose debug logging",
    )

    args = parser.parse_args()

    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        result = run_scraper(
            dry_run=args.dry_run,
            resume=args.resume,
            start_page=args.start_page,
            end_page=args.end_page,
            limit=args.limit,
            output_dir=Path(args.output_dir),
        )

        # Print summary
        print(f"\n{'='*50}")
        print("Scraping Summary")
        print(f"{'='*50}")
        print(f"Total episodes found: {result['total_found']}")
        print(f"Skipped (existing):   {result['skipped_existing']}")
        print(f"Succeeded:            {result['succeeded']}")
        print(f"Failed:               {result['failed']}")
        if result["failures"]:
            print(f"Failed episodes:      {', '.join(result['failures'])}")
        print(f"{'='*50}")

        # Exit with error code if there were failures
        sys.exit(1 if result["failed"] > 0 else 0)

    except RobotsBlockedError:
        logger.error("Scraping blocked by robots.txt")
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Scraping interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()