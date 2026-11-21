# data/harvester.py
import os
import time
import hashlib
import requests
from bs4 import BeautifulSoup
from stem import Signal
from stem.control import Controller

TOR_PROXIES = {
    "http": "socks5h://127.0.0.1:9050",
    "https": "socks5h://127.0.0.1:9050",
}

class AutonomousHarvester:
    def __init__(self, out_file="data/raw/corpus_all.txt", tor_control_port=9051, tor_password=""):
        os.makedirs(os.path.dirname(out_file), exist_ok=True)
        self.out_file = out_file
        self.seen_hashes = set()
        self.tor_control_port = tor_control_port
        self.tor_password = tor_password

    def renew_tor_circuit(self):
        try:
            with Controller.from_port(port=self.tor_control_port) as ctrl:
                ctrl.authenticate(password=self.tor_password)
                ctrl.signal(Signal.NEWNYM)
                time.sleep(3)
        except Exception as e:
            print(f"[!] تعذر تجديد دائرة Tor: {e}")

    def _save_clean_doc(self, text: str, env_tag: str = ""):
        clean = "\n".join(line.rstrip() for line in text.splitlines() if line.strip())
        if len(clean) < 150:
            return
        h = hashlib.sha256(clean.encode("utf-8")).hexdigest()
        if h in self.seen_hashes:
            return
        self.seen_hashes.add(h)
        with open(self.out_file, "a", encoding="utf-8") as f:
            prefix = f"{env_tag}\n" if env_tag else ""
            f.write(f"<|bos|>{prefix}{clean}<|eos|>\n")

    def scrape_url(self, url: str, use_tor: bool = False, env_tag: str = ""):
        headers = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) Gecko/20100101 Firefox/128.0"}
        try:
            proxies = TOR_PROXIES if use_tor else None
            r = requests.get(url, headers=headers, proxies=proxies, timeout=25)
            if r.status_code == 200:
                soup = BeautifulSoup(r.text, "html.parser")
                for tag in soup(["script", "style", "nav", "footer"]):
                    tag.decompose()
                self._save_clean_doc(soup.get_text(separator="\n"), env_tag=env_tag)
                print(f"[+] سحب ناجح ({'TOR' if use_tor else 'WEB'}): {url}")
        except Exception as e:
            print(f"[-] فشل السحب من {url}: {e}")

    def harvest_github_repo_files(self, repo_dir: str, env_tag: str = ""):
        """استخراج أكواد بايثون وباش وملفات Markdown من المستودعات المستنسخة محلياً"""
        exts = {".py", ".sh", ".bash", ".c", ".h", ".md", ".conf", ".yml", ".yaml"}
        for root, _, files in os.walk(repo_dir):
            for name in files:
                if os.path.splitext(name)[1].lower() in exts:
                    path = os.path.join(root, name)
                    try:
                        with open(path, "r", encoding="utf-8", errors="ignore") as f:
                            content = f.read()
                        doc = f"# FILE: {name}\n{content}"
                        self._save_clean_doc(doc, env_tag=env_tag)
                    except Exception:
                        pass

    def inject_specialized_env_knowledge(self):
        """حقن المعرفة التأسيسية الدقيقة لبيئات iSH و Segfault.net و Kali و Ubuntu"""
        seed_docs = [
            ("<|env_ish|>", """# iSH Shell (Alpine Linux x86 on iOS) Best Practices
- Package manager: apk add --no-cache python3 py3-pip git openssh-client tmux curl nmap
- Architecture constraint: 32-bit x86 usermode emulation on iOS; avoid heavy C-extension compilations (use pure Python sockets/asyncio/paramiko/requests instead of heavy compiled binaries).
- Background keepalive: cat /dev/location > /dev/null & (requires location permission in iOS).
- Default shell: /bin/ash (BusyBox); avoid bashisms like [[ ]] or arrays unless bash is explicitly installed."""),

            ("<|env_segfault|>", """# Segfault.net Disposable Root Server Operations
- Access: ssh -o SetEnv="SECRET=<your_secret>" root@segfault.net (or ssh root@8.segfault.net).
- Persistent encrypted storage: store all Python scripts, keys, and tools inside /sec (persists across container resets; root filesystem resets).
- Outbound network: all traffic is automatically routed through Tor/VPN exit; use proxychains4 or torsocks for explicit SOCKS routing.
- Session resilience: always wrap long scans or Python agents inside tmux or screen so SSH drops on mobile networks (Termius/iSH) never kill running tools."""),

            ("<|env_kali|>", """# Kali Linux Offensive & Defensive Automation
- Package manager: apt update && apt install -y python3-scapy python3-pwntools nmap masscan gobuster sqlmap metasploit-framework.
- Raw sockets & packet crafting: run Python scapy/raw-socket tools with sudo or CAP_NET_RAW (setcap cap_net_raw+ep $(readlink -f $(which python3)))."""),

            ("<|env_ubuntu|>", """# Ubuntu Server Security & Infrastructure Hardening
- Service & firewall automation: ufw, nftables, systemd timers, fail2ban, journalctl analysis via Python subprocess and systemd-python.
- Isolated execution: deploy Docker rootless containers for running untrusted security tools and self-testing AI-generated Python scripts.""")
        ]
        for tag, doc in seed_docs:
            self._save_clean_doc(doc, env_tag=tag)

if __name__ == "__main__":
    h = AutonomousHarvester()
    h.inject_specialized_env_knowledge()
    # مثال للزحف العادي والمظلم (أضف روابط .onion وقائمة المستودعات الخاصة بك)
    h.scrape_url("https://wiki.alpinelinux.org/wiki/Alpine_Package_Keeper", use_tor=False, env_tag="<|env_ish|>")
    # h.scrape_url("http://exampleonionaddress.onion/forum", use_tor=True, env_tag="<|tor_search|>")
