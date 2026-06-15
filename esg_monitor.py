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
import difflib
from datetime import datetime, timezone, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import feedparser
from google import genai
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

gemini_client = genai.Client(api_key=GEMINI_API_KEY)


# ─── 1. RSS 수집 ───────────────────────────────────────────────────────────────

def fetch_articles(urls: list[str], cutoff: datetime = CUTOFF) -> list[dict]:
    seen_hashes: set[str] = set()
    seen_titles: list[str] = []
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
                if pub_dt is None or pub_dt < cutoff:
                    continue

                # 정확한 중복 제거
                dedup_key = hashlib.md5(title.encode()).hexdigest()
                if dedup_key in seen_hashes:
                    continue

                # 유사 제목 중복 제거 (72% 이상 유사하면 같은 이슈로 판단)
                if _is_similar_title(title, seen_titles):
                    log.debug("유사 기사 제외: %s", title[:50])
                    continue

                seen_hashes.add(dedup_key)
                seen_titles.append(title)

                articles.append({
                    "title": title,
                    "link": link,
                    "published": pub_dt,
                    "source": feed.feed.get("title", url),
                    "summary": _clean_html(getattr(entry, "summary", "")),
                })
        except Exception as e:
            log.warning("RSS 파싱 실패 (%s): %s", url, e)

    log.info("수집된 기사 수: %d (기준: 최근 %dh)", len(articles),
             round((NOW_KST - cutoff).total_seconds() / 3600))
    return articles


def _is_similar_title(title: str, seen: list[str], threshold: float = 0.72) -> bool:
    t = title.lower()
    for s in seen:
        if difflib.SequenceMatcher(None, t, s.lower()).ratio() >= threshold:
            return True
    return False


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
        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )
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
    "high":   "#DC2626",  # 9-10: 빨강
    "mid":    "#EA580C",  # 7-8: 주황
    "normal": "#2563EB",  # 6: 파랑
    "low":    "#6B7280",  # ~5: 회색
}
SCORE_LABELS = {
    "high":   "긴급",
    "mid":    "중요",
    "normal": "일반",
    "low":    "참고",
}


def _score_tier(score: int) -> str:
    if score >= 9: return "high"
    if score >= 7: return "mid"
    if score >= 6: return "normal"
    return "low"


def _bullet_rows(items: list[str], color: str) -> str:
    rows = []
    for text in items:
        if not text:
            continue
        rows.append(
            f'<div style="display:flex;align-items:flex-start;gap:8px;margin-bottom:6px;">'
            f'<span style="color:{color};font-size:9px;margin-top:5px;flex-shrink:0;">&#9632;</span>'
            f'<span style="color:#374151;font-size:13px;line-height:1.6;">{text}</span>'
            f'</div>'
        )
    return "".join(rows)


def build_html(articles: list[dict]) -> str:
    date_str = NOW_KST.strftime("%Y년 %m월 %d일")

    # 헤더 통계
    urgent  = sum(1 for a in articles if a.get("score", 0) >= 9)
    important = sum(1 for a in articles if 7 <= a.get("score", 0) < 9)
    normal  = sum(1 for a in articles if a.get("score", 0) < 7)

    stat_html = (
        f'<span style="margin-right:18px;font-size:13px;opacity:0.9;">'
        f'<span style="font-weight:700;font-size:18px;">{urgent}</span> 긴급</span>'
        f'<span style="margin-right:18px;font-size:13px;opacity:0.9;">'
        f'<span style="font-weight:700;font-size:18px;">{important}</span> 중요</span>'
        f'<span style="font-size:13px;opacity:0.9;">'
        f'<span style="font-weight:700;font-size:18px;">{normal}</span> 일반</span>'
    )

    # 기사 카드
    cards_html = ""
    for art in articles:
        score = art.get("score", 0)
        tier  = _score_tier(score)
        color = SCORE_COLORS[tier]
        label = SCORE_LABELS[tier]
        category   = art.get("category", "기타")
        core       = art.get("core_summary", [])
        implication = art.get("samsung_implication", [])
        action     = art.get("action_plan", "")
        article_url = art["link"]
        pub_str    = art["published"].strftime("%m/%d %H:%M") if art.get("published") else ""

        core_html = _bullet_rows(core, color)
        impl_html = _bullet_rows(implication, color)

        cards_html += f"""
    <div style="background:#ffffff;border-radius:16px;overflow:hidden;
                margin-bottom:16px;box-shadow:0 2px 12px rgba(0,0,0,0.07);">
      <!-- 점수 색상 상단 바 -->
      <div style="height:5px;background:{color};"></div>

      <div style="padding:20px 24px;">
        <!-- 상단 메타 -->
        <div style="display:flex;align-items:center;gap:10px;margin-bottom:14px;flex-wrap:wrap;">
          <!-- 점수 원형 뱃지 -->
          <div style="width:46px;height:46px;border-radius:50%;background:{color};
                      display:flex;align-items:center;justify-content:center;flex-shrink:0;">
            <span style="color:#fff;font-size:20px;font-weight:900;line-height:1;">{score}</span>
          </div>
          <div>
            <div style="margin-bottom:4px;">
              <span style="background:{color}1a;color:{color};font-size:11px;font-weight:700;
                           padding:3px 10px;border-radius:20px;margin-right:6px;">{label}</span>
              <span style="background:#F3F4F6;color:#6B7280;font-size:11px;
                           padding:3px 10px;border-radius:20px;">{category}</span>
            </div>
            <span style="color:#9CA3AF;font-size:11px;">{pub_str} KST</span>
          </div>
        </div>

        <!-- 제목 -->
        <h3 style="margin:0 0 16px;font-size:16px;font-weight:700;line-height:1.55;color:#111827;">
          <a href="{article_url}" style="color:#111827;text-decoration:none;" target="_blank">{art['title']}</a>
        </h3>

        <!-- 구분선 -->
        <div style="border-top:1px solid #F3F4F6;margin-bottom:14px;"></div>

        <!-- 핵심 요약 -->
        <div style="margin-bottom:12px;">
          <div style="font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:1.5px;
                      color:#9CA3AF;margin-bottom:8px;">핵심 요약</div>
          {core_html}
        </div>

        <!-- 삼성전자 시사점 -->
        <div style="margin-bottom:14px;">
          <div style="font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:1.5px;
                      color:#9CA3AF;margin-bottom:8px;">삼성전자 시사점</div>
          {impl_html}
        </div>

        <!-- 액션플랜 -->
        <div style="background:linear-gradient(135deg,#EFF6FF,#DBEAFE);border-radius:10px;
                    padding:13px 16px;border-left:3px solid #3B82F6;">
          <div style="font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:1.5px;
                      color:#3B82F6;margin-bottom:5px;">액션플랜</div>
          <div style="color:#1E40AF;font-size:13px;font-weight:600;line-height:1.6;">{action}</div>
        </div>
      </div>
    </div>"""

    return f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
</head>
<body style="margin:0;padding:0;background:#F0F2F5;font-family:'Apple SD Gothic Neo','Noto Sans KR',Arial,sans-serif;">
  <div style="max-width:680px;margin:0 auto;padding:24px 16px;">

    <!-- 헤더 -->
    <div style="background:linear-gradient(135deg,#0d5c2e 0%,#1a7f37 60%,#15803d 100%);
                border-radius:18px;padding:30px 32px;margin-bottom:16px;color:#fff;">
      <div style="font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:2px;
                  opacity:0.65;margin-bottom:10px;">삼성전자 ESG 전략팀</div>
      <h1 style="margin:0 0 6px;font-size:26px;font-weight:800;letter-spacing:-0.5px;">ESG 뉴스 브리핑</h1>
      <div style="font-size:14px;opacity:0.8;margin-bottom:20px;">{date_str} &nbsp;·&nbsp; 총 {len(articles)}건</div>
      <!-- 통계 요약 -->
      <div style="background:rgba(255,255,255,0.12);border-radius:12px;padding:14px 18px;
                  display:flex;align-items:center;">
        {stat_html}
      </div>
    </div>

    <!-- 점수 범례 -->
    <div style="background:#fff;border-radius:12px;padding:12px 20px;margin-bottom:20px;
                box-shadow:0 1px 4px rgba(0,0,0,0.06);display:flex;align-items:center;flex-wrap:wrap;gap:4px;">
      <span style="font-size:11px;color:#9CA3AF;font-weight:600;margin-right:8px;">점수 기준</span>
      {''.join(
          f'<span style="display:inline-flex;align-items:center;gap:5px;margin-right:14px;">'
          f'<span style="width:10px;height:10px;border-radius:50%;background:{SCORE_COLORS[t]};display:inline-block;"></span>'
          f'<span style="font-size:12px;color:#6B7280;">{lbl} ({rng})</span></span>'
          for t, lbl, rng in [
              ("high","긴급","9-10"), ("mid","중요","7-8"),
              ("normal","일반","6"), ("low","참고","~5"),
          ]
      )}
    </div>

    <!-- 기사 카드 -->
    {cards_html}

    <!-- 푸터 -->
    <div style="text-align:center;padding:24px 0 8px;color:#9CA3AF;font-size:11px;line-height:1.8;">
      자동 생성 &nbsp;·&nbsp; ESG Monitor &nbsp;·&nbsp; Powered by Gemini AI<br>
      기사 제목 클릭 시 원문으로 이동합니다.
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
    for extra_hours in (48, 72):
        if len(articles) >= MIN_ARTICLES:
            break
        log.info("기사 부족(%d개) → %dh 범위로 재수집", len(articles), extra_hours)
        articles = fetch_articles(ALL_RSS_URLS, NOW_KST - timedelta(hours=extra_hours))

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
