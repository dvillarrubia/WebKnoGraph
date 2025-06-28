from dataclasses import dataclass, field


@dataclass
class CrawlerConfig:
    # Adjusted paths to be relative to "/content/drive/My Drive/WebKnoGraph/"
    state_db_path: str = "/content/drive/My Drive/WebKnoGraph/data/crawler_state.db"
    parquet_path: str = "/content/drive/My Drive/WebKnoGraph/data/crawled_data_parquet/"
    min_request_delay: float = 1.0
    max_request_delay: float = 30.0
    max_pages_to_crawl: int = 700
    save_interval_pages: int = 2
    max_retries_request: int = 3
    max_redirects: int = 2
    request_timeout: int = 15
    allowed_path_segment: str = "/blog/"
    initial_start_url: str = "https://example.com/blog"
    user_agents: list[str] = field(
        default_factory=lambda: [
            # Chrome - Windows, Mac, Linux
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.5735.199 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.5993.117 Safari/537.36",
            # Firefox - Windows, Mac
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:125.0) Gecko/20100101 Firefox/125.0",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 12.6; rv:115.0) Gecko/20100101 Firefox/115.0",
            # Edge - Windows
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.5938.132 Safari/537.36 Edg/117.0.2045.60",
            # Mobile Browsers
            "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Mobile/15E148 Safari/604.1",
            "Mozilla/5.0 (Linux; Android 12; SM-G998B) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.5845.92 Mobile Safari/537.36",
            # Googlebot / SEO crawlers
            "Mozilla/5.0 (compatible; Googlebot/2.1; +http://www.google.com/bot.html)",
            "Mozilla/5.0 (compatible; Bingbot/2.0; +http://www.bing.com/bingbot.htm)",
        ]
    )
    allowed_query_params: list[str] = field(
        default_factory=lambda: ["page", "p", "q", "id", "post"]
    )
    base_domain: str = ""
