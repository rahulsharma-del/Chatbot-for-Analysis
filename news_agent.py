import requests
import feedparser
from bs4 import BeautifulSoup
from typing import List, Dict

GOOGLE_NEWS_RSS = (
    "https://news.google.com/rss/search?q=Veoci%20healthcare&hl=en-US&gl=US&ceid=US:en"
)

def fetch_veoci_healthcare_news(limit: int = 5) -> List[Dict[str, str]]:
    """Fetch recent healthcare news articles related to Veoci.

    Parameters
    ----------
    limit: int
        Maximum number of articles to return.

    Returns
    -------
    List[Dict[str, str]]
        A list of dictionaries each containing title, link and summary.
    """
    feed = feedparser.parse(requests.get(GOOGLE_NEWS_RSS, timeout=10).text)
    articles = []
    for entry in feed.entries[:limit]:
        link = entry.link
        summary = _summarise_article(link)
        if not summary:
            summary = entry.get("summary", "")
        articles.append({"title": entry.title, "link": link, "summary": summary})
    return articles

def _summarise_article(url: str) -> str:
    """Fetch the article content and return the first couple of sentences."""
    try:
        resp = requests.get(url, timeout=10)
        soup = BeautifulSoup(resp.content, "html.parser")
        paragraphs = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
        text = " ".join(paragraphs)
        sentences = [s.strip() for s in text.split(".") if s.strip()]
        return ". ".join(sentences[:2]) + ("." if sentences else "")
    except Exception:
        return ""

if __name__ == "__main__":
    news_items = fetch_veoci_healthcare_news()
    for item in news_items:
        print(f"Title: {item['title']}")
        print(f"Summary: {item['summary']}")
        print(f"Link: {item['link']}\n")
