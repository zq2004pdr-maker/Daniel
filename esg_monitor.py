"""
ESG 뉴스 모니터링 시스템
- Google Alerts RSS 수집 → Gemini 분석 → Gmail HTML 이메일 발송
"""

import os
import re
import json
import smtplib
import hashlib
import logging
from datetime import datetime, timezone, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from urllib.parse import quote

import feedparser
import google.generativeai as genai
from dateutil import parser as dateutil_parser

from config import ALL_RSS_URLS, MIN_ARTICLES, PRIORITY_SCORE

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

KST = timezone(timedelta(hours=9))
NOW_KST = datetime.now(KST)
CUTOFF = NOW_KST - timedelta(hours=24)

GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
EMAIL_SENDER = os.environ["EMAIL_SENDER"]
EMAIL_RECEIVER = os.environ["EMAIL_RECEIVER"]
EMAIL_PASSWORD = os.environ["EMAIL_PASSWORD"]

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-2.5-flash")


# ─── 1. RSS 수집 ───────────────────────────────────────────────────────────────

def fetch_articles(urls: list[str]) -> list[dict]:
    seen_hashes: set[str] = set()
    articles: list[dict] = []

    for url in urls:
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries:
                title = _clean_html(getattr(entry, "title", "")).strip()
                link = getattr(entry, "link", "")
                published = getattr(entry, "published", "") or getattr(entry, "updated", "")

                if not title or not link:
                    continue

                pub_dt = _parse_date(published)
                if pub_dt is None or pub_dt < CUTOFF:
                    continue

                dedup_key = hashlib.md5(title.encode()).hexdigest()
                if dedup_key in seen_hashes:
                    continue
                seen_hashes.add(dedup_key)

                articles.append({
                    "title": title,
                    "link": link,
                    "published": pub_dt,
                    "source": feed.feed.get("title", url),
                    "summary": _clean_html(getattr(entry, "summary", "")),
                })
        except Exception as e:
            log.warning("RSS 파싱 실패 (%s): %s", url, e)

    log.info("수집된 기사 수: %d", len(articles))
    return articles


def _parse_date(raw: str) -> datetime | None:
    if not raw:
        return None
    try:
        dt = dateutil_parser.parse(raw)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(KST)
    except Exception:
        return None


def _clean_html(text: str) -> str:
    return re.sub(r"<[^>]+>", "", text).strip()


# ─── 2. Gemini 분석 ────────────────────────────────────────────────────────────

ANALYSIS_PROMPT = """당신은 삼성전자 ESG 전략팀 애널리스트입니다.
아래 뉴스 기사를 분석하여 반드시 JSON 형식으로만 응답하세요. 마크다운 코드블록 없이 순수 JSON만 출력하세요.

기사 제목: {title}
기사 요약: {summary}

JSON 형식:
{{
  "category": "환경(E)/사회(S)/지배구조(G)/규제/공급망/기후 중 하나",
  "score": 1에서 10 사이 정수 (삼성전자 ESG 전략 관련성과 중요도),
  "core_summary": ["핵심요약 첫 번째 문장", "핵심요약 두 번째 문장"],
  "samsung_implication": ["삼성전자 시사점 첫 번째 문장", "삼성전자 시사점 두 번째 문장"],
  "action_plan": "삼성전자 담당자가 취해야 할 액션플랜 한 문장"
}}

score 기준:
- 9-10: 즉각적 규제 대응 필요 또는 글로벌 ESG 기준 변경
- 7-8: 중요한 산업 트렌드 또는 경쟁사 동향
- 5-6: 일반적 ESG 뉴스
- 1-4: 관련성 낮음"""


def analyze_article(article: dict) -> dict | None:
    prompt = ANALYSIS_PROMPT.format(
        title=article["title"],
        summary=article["summary"][:800] if article["summary"] else "요약 없음",
    )
    try:
        response = model.generate_content(prompt)
        text = response.text.strip()
        text = re.sub(r"```(?:json)?", "", text).strip().rstrip("```").strip()
        result = json.loads(text)
        article.update(result)
        return article
    except json.JSONDecodeError as e:
        log.warning("JSON 파싱 실패 (%s): %s", article["title"][:40], e)
        return None
    except Exception as e:
        log.warning("Gemini API 오류 (%s): %s", article["title"][:40], e)
        return None


def analyze_all(articles: list[dict]) -> list[dict]:
    analyzed: list[dict] = []
    for i, art in enumerate(articles):
        log.info("분석 중 (%d/%d): %s", i + 1, len(articles), art["title"][:50])
        result = analyze_article(art)
        if result is not None:
            analyzed.append(result)

    priority = [a for a in analyzed if a.get("score", 0) >= PRIORITY_SCORE]
    others = [a for a in analyzed if a.get("score", 0) < PRIORITY_SCORE]

    priority.sort(key=lambda x: x.get("score", 0), reverse=True)
    others.sort(key=lambda x: x.get("score", 0), reverse=True)

    combined = priority + others
    final = combined[:max(MIN_ARTICLES, len(priority))]
    log.info("최종 기사 수: %d (우선순위: %d)", len(final), len(priority))
    return final


# ─── 3. HTML 이메일 생성 ───────────────────────────────────────────────────────

SCORE_COLORS = {
    "high": "#DC2626",    # 9-10: 빨강
    "mid": "#EA580C",     # 7-8: 주황
    "normal": "#2563EB",  # 6: 파랑
    "low": "#6B7280",     # ~5: 회색
}


def _score_color(score: int) -> str:
    if score >= 9:
        return SCORE_COLORS["high"]
    if score >= 7:
        return SCORE_COLORS["mid"]
    if score >= 6:
        return SCORE_COLORS["normal"]
    return SCORE_COLORS["low"]


def _google_search_url(title: str) -> str:
    return f"https://www.google.com/search?q={quote(title)}"


def build_html(articles: list[dict]) -> str:
    date_str = NOW_KST.strftime("%Y년 %m월 %d일")
    cards_html = ""

    for art in articles:
        score = art.get("score", 0)
        color = _score_color(score)
        category = art.get("category", "기타")
        core = art.get("core_summary", ["", ""])
        implication = art.get("samsung_implication", ["", ""])
        action = art.get("action_plan", "")
        search_url = _google_search_url(art["title"])
        pub_str = art["published"].strftime("%m/%d %H:%M") if art.get("published") else ""

        core_html = "".join(f"<li>{s}</li>" for s in core if s)
        impl_html = "".join(f"<li>{s}</li>" for s in implication if s)

        cards_html += f"""
        <div style="background:#fff;border-radius:12px;padding:20px 24px;margin-bottom:20px;
                    box-shadow:0 1px 4px rgba(0,0,0,0.08);border-left:5px solid {color};">
          <div style="display:flex;align-items:center;gap:10px;margin-bottom:10px;flex-wrap:wrap;">
            <span style="background:{color};color:#fff;font-weight:700;font-size:18px;
                         padding:4px 12px;border-radius:20px;">{score}</span>
            <span style="background:#F3F4F6;color:#374151;font-size:12px;
                         padding:3px 10px;border-radius:10px;">{category}</span>
            <span style="color:#9CA3AF;font-size:12px;margin-left:auto;">{pub_str} KST</span>
          </div>
          <h3 style="margin:0 0 12px;font-size:16px;line-height:1.5;">
            <a href="{search_url}" style="color:#111827;text-decoration:none;"
               target="_blank">{art['title']}</a>
          </h3>
          <div style="margin-bottom:10px;">
            <p style="font-weight:600;color:#374151;margin:0 0 4px;font-size:13px;">핵심 요약</p>
            <ul style="margin:0;padding-left:18px;color:#4B5563;font-size:13px;line-height:1.7;">
              {core_html}
            </ul>
          </div>
          <div style="margin-bottom:10px;">
            <p style="font-weight:600;color:#374151;margin:0 0 4px;font-size:13px;">삼성전자 시사점</p>
            <ul style="margin:0;padding-left:18px;color:#4B5563;font-size:13px;line-height:1.7;">
              {impl_html}
            </ul>
          </div>
          <div style="background:#F9FAFB;border-radius:8px;padding:10px 14px;">
            <p style="font-weight:600;color:#374151;margin:0 0 3px;font-size:12px;">액션플랜</p>
            <p style="color:#1D4ED8;margin:0;font-size:13px;font-weight:500;">{action}</p>
          </div>
        </div>"""

    legend_html = "".join([
        f'<span style="display:inline-flex;align-items:center;gap:4px;margin-right:14px;">'
        f'<span style="width:12px;height:12px;border-radius:50%;background:{c};display:inline-block;"></span>'
        f'<span style="font-size:12px;color:#6B7280;">{label}</span></span>'
        for label, c in [("9-10점", SCORE_COLORS["high"]), ("7-8점", SCORE_COLORS["mid"]),
                          ("6점", SCORE_COLORS["normal"]), ("~5점", SCORE_COLORS["low"])]
    ])

    return f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
</head>
<body style="margin:0;padding:0;background:#F3F4F6;font-family:'Apple SD Gothic Neo',
             'Noto Sans KR',Arial,sans-serif;">
  <div style="max-width:680px;margin:0 auto;padding:20px;">

    <!-- 헤더 -->
    <div style="background:linear-gradient(135deg,#1a7f37 0%,#166534 100%);
                border-radius:14px;padding:28px 30px;margin-bottom:20px;color:#fff;">
      <div style="font-size:12px;opacity:0.8;margin-bottom:4px;">삼성전자 ESG 전략팀</div>
      <h1 style="margin:0 0 6px;font-size:24px;font-weight:700;">ESG 뉴스 브리핑</h1>
      <div style="font-size:14px;opacity:0.9;">{date_str} &nbsp;·&nbsp; 총 {len(articles)}건</div>
    </div>

    <!-- 범례 -->
    <div style="background:#fff;border-radius:10px;padding:12px 18px;margin-bottom:20px;
                box-shadow:0 1px 3px rgba(0,0,0,0.06);">
      <span style="font-size:12px;color:#6B7280;margin-right:10px;">점수 범례:</span>
      {legend_html}
    </div>

    <!-- 기사 카드 -->
    {cards_html}

    <!-- 푸터 -->
    <div style="text-align:center;padding:20px 0;color:#9CA3AF;font-size:12px;">
      자동 생성 · ESG Monitor &nbsp;|&nbsp; Powered by Gemini AI<br>
      기사 제목 클릭 시 Google 검색으로 이동합니다.
    </div>
  </div>
</body>
</html>"""


# ─── 4. Gmail 발송 ─────────────────────────────────────────────────────────────

def send_email(html_body: str, article_count: int) -> None:
    subject = f"[ESG 브리핑] {NOW_KST.strftime('%Y.%m.%d')} · {article_count}건"

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = EMAIL_SENDER
    msg["To"] = EMAIL_RECEIVER
    msg.attach(MIMEText(html_body, "html", "utf-8"))

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(EMAIL_SENDER, EMAIL_PASSWORD)
        server.sendmail(EMAIL_SENDER, EMAIL_RECEIVER.split(","), msg.as_string())
    log.info("이메일 발송 완료: %s → %s (%d건)", EMAIL_SENDER, EMAIL_RECEIVER, article_count)


# ─── 5. 메인 ───────────────────────────────────────────────────────────────────

def main() -> None:
    log.info("=== ESG 모니터링 시작 (%s) ===", NOW_KST.strftime("%Y-%m-%d %H:%M KST"))

    articles = fetch_articles(ALL_RSS_URLS)
    if not articles:
        log.warning("수집된 기사가 없습니다. RSS URL을 확인하세요.")
        return

    analyzed = analyze_all(articles)
    if not analyzed:
        log.warning("분석된 기사가 없습니다.")
        return

    html = build_html(analyzed)
    send_email(html, len(analyzed))
    log.info("=== 완료 ===")


if __name__ == "__main__":
    main()
