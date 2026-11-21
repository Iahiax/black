# agent/self_evolve.py
import os
import re
import json
import argparse
import subprocess
import requests
from bs4 import BeautifulSoup
from inference.cpu_runner import CPURunner

TOR_PROXIES = {
    "http": "socks5h://127.0.0.1:9050",
    "https": "socks5h://127.0.0.1:9050",
}

VALID_ENVS = {"ish", "segfault", "kali", "ubuntu", "python"}


class CyberCoreAutonomousAgent:
    def __init__(
        self,
        gguf_path: str = "export/cybercore-1.5b-q8_0.gguf",
        tok_path: str | None = None,
        replay_file: str = "data/raw/self_learned.jsonl",
        corpus_file: str = "data/raw/corpus_all.txt",
        quantize_int8: bool = True,
    ):
        self.runner = CPURunner(
            gguf_path=gguf_path,
            tok_path=tok_path,
            quantize_int8=quantize_int8,
        )
        self.replay_file = replay_file
        self.corpus_file = corpus_file
        os.makedirs(os.path.dirname(replay_file) or ".", exist_ok=True)
        os.makedirs(os.path.dirname(corpus_file) or ".", exist_ok=True)

    def search_surface_or_darkweb(self, url: str, darknet: bool = False) -> str:
        """سحب وتنظيف المحتوى من الإنترنت العادي أو الشبكة المظلمة عبر Tor"""
        headers = {
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) Gecko/20100101 Firefox/128.0"
        }
        try:
            r = requests.get(
                url,
                headers=headers,
                proxies=TOR_PROXIES if darknet else None,
                timeout=25,
            )
            r.raise_for_status()
            soup = BeautifulSoup(r.text, "html.parser")
            for tag in soup(["script", "style", "nav", "footer"]):
                tag.decompose()
            clean_text = "\n".join(
                line.strip() for line in soup.get_text(separator="\n").splitlines() if line.strip()
            )
            return clean_text[:4000]
        except Exception as e:
            return f"FETCH_ERROR: {e}"

    def execute_in_environment(
        self,
        env: str,
        code_or_cmd: str,
        is_python: bool = False,
        segfault_secret: str = "",
        timeout: int = 60,
    ) -> tuple[int, str]:
        """تنفيذ السكربت أو الأمر في البيئة المحددة بدقة"""
        env = env.lower().strip()

        if env == "ish":
            # محاكاة مطابقة لبيئة iSH (Alpine Linux 32-bit x86 + ash shell)
            shell_cmd = f"python3 - <<'EOF'\n{code_or_cmd}\nEOF" if is_python else code_or_cmd
            cmd = [
                "docker", "run", "--rm", "--platform", "linux/386",
                "alpine:latest", "ash", "-c", shell_cmd,
            ]
        elif env == "kali":
            shell_cmd = f"python3 - <<'EOF'\n{code_or_cmd}\nEOF" if is_python else code_or_cmd
            cmd = ["docker", "run", "--rm", "kalilinux/kali-rolling", "bash", "-c", shell_cmd]
        elif env == "ubuntu":
            shell_cmd = f"python3 - <<'EOF'\n{code_or_cmd}\nEOF" if is_python else code_or_cmd
            cmd = ["docker", "run", "--rm", "ubuntu:24.04", "bash", "-c", shell_cmd]
        elif env == "segfault":
            # تنفيذ مباشر على خادم Segfault.net عبر SSH
            remote_cmd = f"python3 - <<'EOF'\n{code_or_cmd}\nEOF" if is_python else code_or_cmd
            cmd = [
                "ssh", "-o", "StrictHostKeyChecking=no",
                "-o", "ConnectTimeout=15",
            ]
            if segfault_secret:
                cmd.extend(["-o", f"SetEnv=SECRET={segfault_secret}"])
            cmd.extend(["root@segfault.net", remote_cmd])
        else:
            cmd = ["python3", "-c", code_or_cmd]

        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
            output = (proc.stdout + "\n" + proc.stderr).strip()
            return proc.returncode, output
        except subprocess.TimeoutExpired:
            return 124, f"EXECUTION_TIMEOUT: تجاوز التنفيذ مهلة {timeout} ثانية."
        except FileNotFoundError as e:
            return 127, f"MISSING_BINARY: {e}"
        except Exception as e:
            return 1, f"EXECUTION_ERROR: {e}"

    def _extract_code(self, text: str) -> tuple[str, bool]:
        """استخراج الكود من الرد وتحديد ما إذا كان بايثون أو أوامر Shell"""
        match = re.search(r"```([a-zA-Z0-9_-]*)\n(.*?)```", text, re.DOTALL)
        if match:
            lang = match.group(1).strip().lower()
            code = match.group(2).strip()
            is_python = lang in {"python", "py", "python3"}
            return code, is_python
        return text.strip(), False

    def _handle_optional_search(self, text: str) -> str | None:
        """التقاط طلبات البحث التلقائي في الإنترنت العادي أو المظلم إذا أصدرها النموذج"""
        web_match = re.search(r"<\|web_search\|>\s*(https?://\S+)", text)
        if web_match:
            url = web_match.group(1).strip()
            print(f"[*] الوكيل يبحث في الإنترنت العادي: {url}")
            return self.search_surface_or_darkweb(url, darknet=False)

        tor_match = re.search(r"<\|tor_search\|>\s*(https?://\S+)", text)
        if tor_match:
            url = tor_match.group(1).strip()
            print(f"[*] الوكيل يبحث عبر شبكة Tor: {url}")
            return self.search_surface_or_darkweb(url, darknet=True)

        return None

    def _commit_to_self_learning_memory(
        self, env_tag: str, task: str, solution: str, output: str
    ):
        """حفظ الحل الناجح في ذاكرة التجارب ودمجه تلقائياً في ملف التدريب المستمر"""
        record = {
            "env": env_tag,
            "task": task,
            "verified_solution": solution,
            "execution_output": output[:500],
        }
        with open(self.replay_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        # إضافة النسخة المُحقَّقة إلى corpus_all.txt لتُستخدم في دورة التدريب القادمة
        train_sample = (
            f"<|bos|><|sys|>أنت CyberCore-Zero 1.5B خبير الأمن السيبراني والبايثون.<|eos|>\n"
            f"<|usr|>{env_tag} {task}<|eos|>\n"
            f"<|ast|>{solution}<|eos|>\n"
        )
        with open(self.corpus_file, "a", encoding="utf-8") as f:
            f.write(train_sample)

    def solve_and_learn(
        self,
        task: str,
        target_env: str = "kali",
        max_retries: int = 5,
        segfault_secret: str = "",
        stream: bool = True,
    ):
        """حلقة التوليد -> البحث (إن لزم) -> الاختبار الفعلي -> التصحيح الذاتي -> التعلم"""
        target_env = target_env if target_env in VALID_ENVS else "ubuntu"
        env_tag = f"<|env_{target_env}|>" if target_env != "python" else "<|py_exec|>"

        history = (
            f"<|bos|><|sys|>أنت CyberCore-Zero 1.5B. أنشئ كوداً نظيفاً وقابلاً للتنفيذ مباشرة لبيئة {target_env}.<|eos|>\n"
            f"<|usr|>{env_tag} {task}<|eos|>\n<|ast|>"
        )

        for attempt in range(1, max_retries + 1):
            print(f"\n--- [المحاولة {attempt}/{max_retries} | البيئة: {target_env}] ---")
            solution = self.runner.generate_text(
                history,
                max_new_tokens=768,
                temperature=0.2,
                stream=stream,
            ).strip()

            # 1. التحقق إن كان النموذج طلب بحثاً في الويب أو Tor قبل التنفيذ
            search_result = self._handle_optional_search(solution)
            if search_result is not None:
                history += (
                    f"{solution}<|eos|>\n"
                    f"<|obs|>نتائج البحث:\n{search_result[:2000]}\nأكمل كتابة الكود النهائي الآن.<|eos|>\n<|ast|>"
                )
                continue

            # 2. استخراج الكود وتنفيذه في البيئة المستهدفة
            code_to_run, is_python = self._extract_code(solution)
            if not code_to_run:
                history += f"{solution}<|eos|>\n<|obs|>لم يتم العثور على كود قابل للتنفيذ. اكتب الكود داخل بلوك ```.<|eos|>\n<|ast|>"
                continue

            rc, output = self.execute_in_environment(
                env=target_env,
                code_or_cmd=code_to_run,
                is_python=is_python,
                segfault_secret=segfault_secret,
            )

            if rc == 0:
                print(f"\n[+] نجح التنفيذ في المحاولة {attempt} على بيئة {target_env}!")
                print(f"[+] المخرجات:\n{output[:1000]}")
                self._commit_to_self_learning_memory(env_tag, task, solution, output)
                print(f"[+] تم تسجيل الحل في ذاكرة التعلم الذاتي ({self.replay_file}).")
                return solution, output

            print(f"\n[!] فشل التنفيذ (Exit {rc}) — جارٍ تغذية الخطأ للنموذج للتصحيح الذاتي...")
            history += (
                f"{solution}<|eos|>\n"
                f"<|obs|>فشل التنفيذ برمز خطأ ({rc}):\n{output[:1500]}\n"
                f"أصلح الكود ليعمل بنجاح على بيئة {target_env}.<|eos|>\n<|ast|>"
            )

        return None, "فشل الوكيل بعد استنفاد جميع المحاولات."


def main():
    parser = argparse.ArgumentParser(description="CyberCore-Zero 1.5B Autonomous Self-Evolving Agent (GGUF CPU)")
    parser.add_argument("--gguf", type=str, default="export/cybercore-1.5b-q8_0.gguf")
    parser.add_argument("--tok", type=str, default=None)
    parser.add_argument("--env", type=str, default="ubuntu", choices=sorted(VALID_ENVS))
    parser.add_argument("--task", type=str, default="")
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--segfault_secret", type=str, default=os.environ.get("SEGFAULT_SECRET", ""))
    parser.add_argument("--no_int8", action="store_true")
    args = parser.parse_args()

    agent = CyberCoreAutonomousAgent(
        gguf_path=args.gguf,
        tok_path=args.tok,
        quantize_int8=not args.no_int8,
    )

    if args.task:
        agent.solve_and_learn(
            task=args.task,
            target_env=args.env,
            max_retries=args.retries,
            segfault_secret=args.segfault_secret,
        )
        return

    # الوضع التفاعلي إذا لم يتم تمرير --task
    print("\n=== CyberCore-Zero 1.5B Self-Evolving Agent (اكتب exit للخروج أو /env لتغيير البيئة) ===")
    current_env = args.env
    while True:
        task = input(f"\n[Agent:{current_env}] المهمة> ").strip()
        if not task:
            continue
        if task.lower() in {"exit", "quit"}:
            break
        if task.startswith("/env "):
            new_env = task.split(" ", 1)[1].strip().lower()
            if new_env in VALID_ENVS:
                current_env = new_env
                print(f"[*] تم التبديل إلى بيئة: {current_env}")
            else:
                print(f"[!] بيئة غير صالحة. المتاح: {', '.join(sorted(VALID_ENVS))}")
            continue

        agent.solve_and_learn(
            task=task,
            target_env=current_env,
            max_retries=args.retries,
            segfault_secret=args.segfault_secret,
        )


if __name__ == "__main__":
    main()
