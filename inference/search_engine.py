# inference/search_engine.py
import socket
import time
import urllib.parse
import requests
from bs4 import BeautifulSoup
from stem import Signal
from stem.control import Controller

TOR_HOST = "127.0.0.1"
TOR_SOCKS_PORT = 9050
TOR_CONTROL_PORT = 9051

TOR_PROXIES = {
    "http": f"socks5h://{TOR_HOST}:{TOR_SOCKS_PORT}",
    "https": f"socks5h://{TOR_HOST}:{TOR_SOCKS_PORT}",
}

HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:128.0) Gecko/20100101 Firefox/128.0"
}

# عنوان محرك بحث Ahmia الرسمي على شبكة Tor + البديل عبر بوابة HTTPS الموجهة عبر Tor
AHMIA_ONION_SEARCH = "http://juhanurmihxlp77nkq76byazcldy2hlmovfu2epvl5ankdibsot4csyd.onion/search/?q="
AHMIA_CLEAR_SEARCH = "https://ahmia.fi/search/?q="


class LiveSearchEngine:
    def __init__(self, tor_password: str = ""):
        self.tor_password = tor_password

    def is_tor_running(self) -> bool:
        try:
            with socket.create_connection((TOR_HOST, TOR_SOCKS_PORT), timeout=3):
                return True
        except OSError:
            return False

    def renew_tor_identity(self) -> str:
        if not self.is_tor_running():
            return "خدمة Tor غير مشغّلة على المنفذ 9050."
        try:
            with Controller.from_port(port=TOR_CONTROL_PORT) as ctrl:
                ctrl.authenticate(password=self.tor_password)
                ctrl.signal(Signal.NEWNYM)
                time.sleep(3)
            return "تم تجديد دائرة Tor وتغيير عنوان IP بنجاح."
        except Exception as e:
            return f"تعذر تجديد الهوية عبر منفذ التحكم 9051 (Tor ما زال يعمل): {e}"

    def check_tor_ip(self) -> str:
        if not self.is_tor_running():
            return "Tor غير متصل. شغّله عبر: sudo systemctl start tor"
        try:
            r = requests.get("https://check.torproject.org/api/ip", proxies=TOR_PROXIES, timeout=20)
            return r.text.strip()
        except Exception as e:
            return f"فشل التحقق من اتصال Tor: {e}"

    def _clean_html(self, html: str, max_chars: int = 3500) -> str:
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "noscript", "svg"]):
            tag.decompose()
        lines = [line.strip() for line in soup.get_text(separator="\n").splitlines() if line.strip()]
        return "\n".join(lines)[:max_chars]

    def fetch_url(self, url: str, use_tor: bool = False, max_chars: int = 3500) -> str:
        if use_tor and not self.is_tor_running():
            return "[خطأ Tor]: خدمة Tor غير مشغّلة على 127.0.0.1:9050."
        try:
            r = requests.get(
                url,
                headers=HEADERS,
                proxies=TOR_PROXIES if use_tor else None,
                timeout=30 if use_tor else 20,
            )
            r.raise_for_status()
            return self._clean_html(r.text, max_chars=max_chars)
        except Exception as e:
            return f"[خطأ سحب الصفحة ({url})]: {e}"

    def search_surface_web(self, query: str, max_results: int = 5) -> str:
        """بحث حي في الإنترنت العادي (أو سحب رابط مباشر)"""
        q = query.strip()
        if q.startswith("http://") or q.startswith("https://"):
            return f"=== محتوى الرابط: {q} ===\n{self.fetch_url(q, use_tor=False)}"

        try:
            url = "https://html.duckduckgo.com/html/"
            r = requests.post(url, data={"q": q}, headers=HEADERS, timeout=20)
            r.raise_for_status()
            soup = BeautifulSoup(r.text, "html.parser")
            results = []

            for item in soup.select(".result")[:max_results]:
                title_el = item.select_one(".result__title a")
                snippet_el = item.select_one(".result__snippet")
                if not title_el:
                    continue
                title = title_el.get_text(strip=True)
                href = title_el.get("href", "")
                snippet = snippet_el.get_text(strip=True) if snippet_el else ""
                results.append(f"- العنوان: {title}\n  الرابط: {href}\n  الملخص: {snippet}")

            if not results:
                return f"لم يتم العثور على نتائج للبحث: {q}"
            return f"=== نتائج الإنترنت العادي ({q}) ===\n" + "\n".join(results)
        except Exception as e:
            return f"[خطأ في بحث الإنترنت العادي]: {e}"

    def search_dark_web(self, query: str, max_results: int = 5, fetch_top_onion: bool = False) -> str:
        """بحث حي في الشبكة المظلمة (.onion) عبر Tor SOCKS5"""
        if not self.is_tor_running():
            return "[خطأ Tor]: شغّل خدمة Tor أولاً (sudo systemctl start tor أو tor &) ثم أعد المحاولة."

        q = query.strip()
        # إذا أدخل المستخدم أو النموذج رابط .onion مباشرة
        if ".onion" in q:
            target = q if q.startswith("http") else f"http://{q}"
            return f"=== محتوى موقع Tor: {target} ===\n{self.fetch_url(target, use_tor=True)}"

        encoded_q = urllib.parse.quote_plus(q)
        html = None
        last_err = ""

        # المحاولة الأولى عبر عنوان Ahmia.onion ثم البديل عبر ahmia.fi موجّهاً داخل Tor
        for base_url in (AHMIA_ONION_SEARCH, AHMIA_CLEAR_SEARCH):
            try:
                r = requests.get(base_url + encoded_q, headers=HEADERS, proxies=TOR_PROXIES, timeout=35)
                if r.status_code == 200 and len(r.text) > 300:
                    html = r.text
                    break
            except Exception as e:
                last_err = str(e)

        if not html:
            return f"[خطأ في البحث عبر الشبكة المظلمة]: {last_err}"

        soup = BeautifulSoup(html, "html.parser")
        results = []
        top_onion_url = None

        for li in soup.select("li.result")[:max_results]:
            title_el = li.select_one("h4 a") or li.select_one("a")
            desc_el = li.select_one("p")
            cite_el = li.select_one("cite")
            if not title_el:
                continue
            title = title_el.get_text(strip=True)
            onion_link = cite_el.get_text(strip=True) if cite_el else title_el.get("href", "")
            desc = desc_el.get_text(strip=True) if desc_el else ""
            if not top_onion_url and ".onion" in onion_link:
                top_onion_url = onion_link if onion_link.startswith("http") else f"http://{onion_link}"
            results.append(f"- العنوان: {title}\n  الرابط: {onion_link}\n  الوصف: {desc}")

        if not results:
            return f"لم يتم العثور على نتائج في الشبكة المظلمة للبحث: {q}"

        output = f"=== نتائج الشبكة المظلمة عبر Tor ({q}) ===\n" + "\n".join(results)
        if fetch_top_onion and top_onion_url:
            page_text = self.fetch_url(top_onion_url, use_tor=True, max_chars=1800)
            output += f"\n\n=== مقتطف من النتيجة الأولى ({top_onion_url}) ===\n{page_text}"
        return output
