"""Terminal demo: synthesize a dual-Franka peg-transfer data script with DeepSeek.

The demo is intentionally terminal-first for screen recording:
1. Print the user's natural-language data request.
2. Print the atomic skill spec sent to the model.
3. Ask DeepSeek to generate a task script in the same style as the current thin script.
4. Save and display the generated play_once / scene-generation code.
5. Optionally run the generated script to write HDF5 and MP4 data.

Set the API key outside this file:

    export DEEPSEEK_API_KEY="..."
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import textwrap
import time
import urllib.error
import urllib.request
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_SCRIPT = PROJECT_ROOT / "scripts" / "synthetic_factory_dual_franka_peg_transfer_thin.py"
SKILL_SPEC = PROJECT_ROOT / "prompts" / "factory_dual_franka_peg_transfer_atomic_skills.md"
GENERATED_DIR = PROJECT_ROOT / "scripts"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "llm_demo" / "factory_dual_franka_peg_transfer"
DEFAULT_REQUEST = (
    "帮我合成一段具身数据，内容是两个franka机械臂相对而立于桌面上，桌面中间有一个hole，"
    "franka1手持peg，将peg装配进hole中。另一个机械臂随后将这个peg取出来。"
)


def print_block(title: str, body: str):
    line = "=" * 88
    print(f"\n{line}\n{title}\n{line}", flush=True)
    print(body.rstrip(), flush=True)


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def extract_section(source: str, start_pattern: str, end_pattern: str | None = None) -> str:
    start = re.search(start_pattern, source, flags=re.MULTILINE)
    if not start:
        return ""
    if end_pattern is None:
        return source[start.start() :]
    end = re.search(end_pattern, source[start.end() :], flags=re.MULTILINE)
    if not end:
        return source[start.start() :]
    return source[start.start() : start.end() + end.start()]


def build_prompt(user_request: str, skill_spec: str, template_script: str) -> list[dict[str, str]]:
    system = """You are a senior robotics code-generation assistant.
Generate a complete Python script for InternUtopia/IsaacLab synthetic data.
You must preserve the structure and imports of the provided template script.
Only output one Python code block. Do not include explanations outside code.
The script must implement the requested sequence using the documented atomic skills."""
    user = f"""Natural-language task request:
{user_request}

Atomic skill specification:
{skill_spec}

Template script to follow exactly in style and runtime setup:
```python
{template_script}
```

Generation requirements:
- Keep the AppLauncher, parser, scene registration, Hydra main, and output saving structure.
- Use the dual-Franka peg-transfer scene: register_dual_franka_factory_env(), DUAL_TASK_ID.
- The generated play_once must: reset scene, prepare robot2 home hold, insert peg with franka1,
  release/lift franka1, return franka1 home, grasp peg with franka2, extract peg, save HDF5/MP4.
- Output a full runnable Python file."""
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def call_deepseek(messages: list[dict[str, str]], model: str, timeout: int) -> str:
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY is not set. Export it before running this demo.")

    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.1,
        "max_tokens": 12000,
    }
    request = urllib.request.Request(
        "https://api.deepseek.com/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"DeepSeek HTTP {exc.code}: {detail}") from exc
    return data["choices"][0]["message"]["content"]


def ensure_project_root_import_path(code: str) -> str:
    needle = "_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), \"..\"))\n"
    addition = needle + "if _PROJECT_ROOT not in sys.path:\n    sys.path.insert(0, _PROJECT_ROOT)\n"
    if "sys.path.insert(0, _PROJECT_ROOT)" in code:
        return code
    return code.replace(needle, addition, 1)


def extract_python_code(model_text: str) -> str:
    fenced = re.search(r"```(?:python)?\s*(.*?)```", model_text, flags=re.DOTALL)
    code = fenced.group(1).strip() if fenced else model_text.strip()
    return ensure_project_root_import_path(code) + "\n"


def fallback_generated_script() -> str:
    """Use the current known-good thin task as an offline demo fallback."""
    source = read_text(TEMPLATE_SCRIPT)
    banner = (
        '"""LLM-demo generated dual-Franka peg transfer task.\n\n'
        "This file follows scripts/synthetic_factory_dual_franka_peg_transfer_thin.py.\n"
        '"""\n'
    )
    return re.sub(r'""".*?"""', banner.rstrip(), source, count=1, flags=re.DOTALL)


def parse_rollout_info(output: str) -> dict | None:
    decoder = json.JSONDecoder()
    for match in reversed(list(re.finditer(r"\{", output))):
        try:
            obj, _ = decoder.raw_decode(output[match.start() :])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and isinstance(obj.get("info"), dict):
            return obj["info"]
    return None


def run_generated_script(script_path: Path, output_dir: Path, extra_args: list[str]) -> int:
    runtime_python = str(PROJECT_ROOT / ".venv" / "bin" / "python")
    if not Path(runtime_python).exists():
        runtime_python = sys.executable
    cmd = [
        runtime_python,
        str(script_path),
        "--num_envs",
        "1",
        "--device",
        "cuda:0",
        "--headless",
        "--disable_fabric",
        "--record_camera",
        "franka2_d455",
        "--video_frame_repeat",
        "1",
        "--output_dir",
        str(output_dir),
        "--wrist_camera_rotate_zyx",
        "0",
        "90",
        "180",
        *extra_args,
    ]
    shell_cmd = "source /data/user/isaacsim/setup_conda_env.sh && " + " ".join(shlex.quote(part) for part in cmd)
    print_block("5. 自动运行生成脚本", shell_cmd)

    proc = subprocess.Popen(
        ["/bin/bash", "-lc", shell_cmd],
        cwd=str(PROJECT_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    output_parts = []
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="", flush=True)
        output_parts.append(line)
    return_code = proc.wait()

    info = parse_rollout_info("".join(output_parts))
    if info is not None:
        success = bool(info.get("success"))
        dataset = info.get("dataset", "<missing>")
        video = info.get("video", "<missing>")
        status = "合成成功" if success and return_code == 0 else "合成失败"
        print_block(
            "6. 合成结果",
            f"status:  {status}\n"
            f"success: {success}\n"
            f"hdf5:    {dataset}\n"
            f"mp4:     {video}",
        )
    else:
        status = "合成成功" if return_code == 0 else "合成失败"
        print_block(
            "6. 合成结果",
            f"status:     {status}\n"
            f"returncode: {return_code}\n"
            f"output_dir:  {output_dir}\n"
            "未能从子进程输出中解析 JSON 结果，请检查上方日志和输出目录。",
        )
    return return_code


def main():
    parser = argparse.ArgumentParser(description="DeepSeek terminal demo for synthetic peg-transfer data generation.")
    parser.add_argument("--request", type=str, default=DEFAULT_REQUEST, help="Natural-language data request.")
    parser.add_argument("--model", type=str, default="deepseek-chat", help="DeepSeek model name.")
    parser.add_argument("--timeout", type=int, default=180, help="DeepSeek API timeout in seconds.")
    parser.add_argument("--offline", action="store_true", help="Skip DeepSeek and use the current template as fallback.")
    parser.add_argument("--run", action="store_true", help="Run the generated script after writing it.")
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Generated data output directory.")
    parser.add_argument("--generated_dir", type=Path, default=GENERATED_DIR, help="Directory for generated Python scripts.")
    parser.add_argument("script_args", nargs=argparse.REMAINDER, help="Extra args passed to the generated task after --.")
    args = parser.parse_args()

    template_script = read_text(TEMPLATE_SCRIPT)
    skill_spec = read_text(SKILL_SPEC)

    print_block("1. 用户输入的数据描述", args.request)
    print_block("2. 输入大模型的原子技能描述", skill_spec)

    if args.offline:
        model_text = "```python\n" + fallback_generated_script() + "\n```"
        print_block("DeepSeek 调用", "offline 模式：使用当前 thin task 模板生成演示脚本。")
    else:
        messages = build_prompt(args.request, skill_spec, template_script)
        print_block("DeepSeek 调用", f"model={args.model}\napi_key=从环境变量 DEEPSEEK_API_KEY 读取，不在终端打印")
        model_text = call_deepseek(messages, model=args.model, timeout=args.timeout)

    generated_code = extract_python_code(model_text)
    args.generated_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    generated_path = args.generated_dir / f"llm_generated_dual_franka_peg_transfer_{timestamp}.py"
    generated_path.write_text(generated_code, encoding="utf-8")

    play_once = extract_section(generated_code, r"^    def play_once\(self\):", r"^    def check_success")
    scene_code = extract_section(
        generated_code,
        r"^register_dual_franka_factory_env\(\)",
        r"^class DualFrankaPegTransferTask",
    )
    main_code = extract_section(generated_code, r"^@hydra_task_config\(DUAL_TASK_ID", r"^if __name__ == .__main__.")

    shown = "\n\n".join(
        part
        for part in (
            "# Scene registration / scene configuration entry\n" + scene_code.strip(),
            "# play_once generated by the model\n" + play_once.strip(),
            "# Hydra main scene setup\n" + main_code.strip(),
        )
        if part.strip()
    )
    print_block("3. 大模型生成的 play_once 和场景生成代码", shown or generated_code[:6000])
    print_block(
        "4. 生成脚本和输出目录",
        textwrap.dedent(
            f"""
            generated_script: {generated_path}
            output_dir:       {args.output_dir}
            """
        ).strip(),
    )

    if args.run:
        extra = args.script_args
        if extra and extra[0] == "--":
            extra = extra[1:]
        raise SystemExit(run_generated_script(generated_path, args.output_dir, extra))

    print("\n未加 --run，所以只生成脚本不启动仿真。录屏时要自动合成数据就加 --run。", flush=True)


if __name__ == "__main__":
    main()
