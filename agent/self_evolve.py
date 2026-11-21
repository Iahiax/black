# agent/self_evolve.py
import os
import json
import subprocess
import requests
from inference.cpu_runner import CPURunner

TOR_PROXIES = {"http": "socks5h://127.0.0.1:9050", "https": "socks5h://127.0.0.1:9050"}

class CyberCoreAutonomousAgent:
    def __init__(
        self,
        ckpt_path="checkpoints/cybercore_inference.pt",
        tok_path="tokenizer/out/tokenizer.json",
        replay_file="data/raw/self_learned.jsonl"
    ):
        self.runner = CPURunner(ckpt_path=ckpt_path, tok_path=tok_path, quantize_int8=True)
        self.replay_file = replay_file
        os.makedirs(os.path.dirname(replay_file), exist_ok=True)

    def search_surface_or_darkweb(self, url: str, darknet: bool = False) -> str:
        try:
            r = requests.get(url, proxies=TOR_PROXIES if darknet else None, timeout=20)
            return r.text[:4000]
        except Exception as e:
            return f"ERROR: {e}"

    def execute_in_environment(self, env: str, code_or_cmd: str, segfault_secret: str = "") -> tuple[int, str]:
        if env == "ish":
            cmd = ["docker", "run", "--rm", "--platform", "linux/386", "alpine:latest", "ash", "-c", code_or_cmd]
        elif env == "kali":
            cmd = ["docker", "run", "--rm", "kalilinux/kali-rolling", "bash", "-c", code_or_cmd]
        elif env == "ubuntu":
            cmd = ["docker", "run", "--rm", "ubuntu:24.04", "bash", "-c", code_or_cmd]
        elif env == "segfault":
            cmd = [
                "ssh", "-o", "StrictHostKeyChecking=no",
                "-o", f"SetEnv=SECRET={segfault_secret}",
                "root@segfault.net", code_or_cmd
            ]
        else:
            cmd = ["python3", "-c", code_or_cmd]

        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        return proc.returncode, (proc.stdout + "\n" + proc.stderr).strip()

    def solve_and_learn(self, task: str, target_env: str = "kali", max_retries: int = 5):
        env_tag = f"<|env_{target_env}|>"
        history = (
            f"<|bos|><|sys|>أنت CyberCore-Zero. أنشئ كوداً قابلاً للتنفيذ مباشرة لبيئة {target_env}.<|eos|>\n"
            f"<|usr|>{env_tag} {task}<|eos|>\n<|ast|>"
        )

        for attempt in range(1, max_retries + 1):
            solution = self.runner.generate_text(history, max_new_tokens=768, temperature=0.2).strip()
            code_to_run = self._extract_code(solution)
            rc, output = self.execute_in_environment(target_env, code_to_run)

            if rc == 0:
                print(f"[+] نجح التنفيذ في المحاولة {attempt} على بيئة {target_env}!")
                self._commit_to_self_learning_memory(env_tag, task, solution, output)
                return solution, output

            print(f"[!] خطأ في المحاولة {attempt} (Exit {rc}) — جارٍ التصحيح الذاتي...")
            history += f"{solution}<|eos|>\n<|obs|>فشل التنفيذ:\n{output[:1500]}\nأصلح الكود.<|eos|>\n<|ast|>"

        return None, "فشل بعد استنفاد المحاولات."

    def _extract_code(self, text: str) -> str:
        if "```" in text:
            blocks = text.split("```")
            if len(blocks) >= 2:
                code = blocks[1]
                lines = code.splitlines()
                return "\n".join(lines[1:]) if lines and lines[0].strip() in ("python", "bash", "sh", "ash") else code
        return text

    def _commit_to_self_learning_memory(self, env_tag: str, task: str, solution: str, output: str):
        record = {"env": env_tag, "task": task, "verified_solution": solution, "execution_output": output[:500]}
        with open(self.replay_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    agent = CyberCoreAutonomousAgent()
    agent.solve_and_learn("اكتب سكربت بايثون يفحص المنافذ الشائعة محلياً بدون مكتبات خارجية.", target_env="ubuntu")
