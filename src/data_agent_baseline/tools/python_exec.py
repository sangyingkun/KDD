from __future__ import annotations

import contextlib
import io
import json
import multiprocessing
import os
import shutil
import sys
import tempfile
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Force UTF-8 encoding for all subprocess stdout/stderr
_SUB_STDOUT_ENCODING = "utf-8"
_SUB_STDERR_ENCODING = "utf-8"


@contextlib.contextmanager
def _capture_process_streams(stdout_path: Path, stderr_path: Path):
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    saved_stdout_fd = os.dup(1)
    saved_stderr_fd = os.dup(2)

    with stdout_path.open("w+b") as stdout_file, stderr_path.open("w+b") as stderr_file:
        try:
            if original_stdout is not None:
                original_stdout.flush()
            if original_stderr is not None:
                original_stderr.flush()

            os.dup2(stdout_file.fileno(), 1)
            os.dup2(stderr_file.fileno(), 2)

            sys.stdout = io.TextIOWrapper(
                os.fdopen(os.dup(1), "wb"),
                encoding=_SUB_STDOUT_ENCODING,
                errors="replace",
                line_buffering=True,
                write_through=True,
            )
            sys.stderr = io.TextIOWrapper(
                os.fdopen(os.dup(2), "wb"),
                encoding=_SUB_STDERR_ENCODING,
                errors="replace",
                line_buffering=True,
                write_through=True,
            )
            yield
        finally:
            if sys.stdout is not None:
                sys.stdout.flush()
            if sys.stderr is not None:
                sys.stderr.flush()

            if sys.stdout is not original_stdout:
                sys.stdout.close()
            if sys.stderr is not original_stderr:
                sys.stderr.close()

            sys.stdout = original_stdout
            sys.stderr = original_stderr
            os.dup2(saved_stdout_fd, 1)
            os.dup2(saved_stderr_fd, 2)
            os.close(saved_stdout_fd)
            os.close(saved_stderr_fd)


def _read_captured_stream(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def _execute_code_in_namespace(
    *,
    namespace: dict[str, Any],
    context_root: str,
    code: str,
    stdout_path: Path,
    stderr_path: Path,
) -> dict[str, Any]:
    try:
        os.chdir(context_root)
        with _capture_process_streams(stdout_path, stderr_path):
            exec(code, namespace, namespace)
        return {"success": True}
    except BaseException as exc:  # noqa: BLE001
        return {
            "success": False,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }


def _run_python_code_once(
    context_root: str,
    code: str,
    stdout_path: str,
    stderr_path: str,
    result_path: str,
) -> None:
    namespace: dict[str, Any] = {
        "__builtins__": __builtins__,
        "__name__": "__main__",
        "context_root": context_root,
        "Path": Path,
    }
    result = _execute_code_in_namespace(
        namespace=namespace,
        context_root=context_root,
        code=code,
        stdout_path=Path(stdout_path),
        stderr_path=Path(stderr_path),
    )
    Path(result_path).write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")


def execute_python_code(context_root: Path, code: str, *, timeout_seconds: int = 30) -> dict[str, Any]:
    resolved_context_root = context_root.resolve()
    runtime_root = resolved_context_root / ".data_agent_runtime"
    runtime_root.mkdir(parents=True, exist_ok=True)
    temp_dir = runtime_root / f"python_once_{uuid.uuid4().hex}"
    temp_dir.mkdir(parents=True, exist_ok=True)
    try:
        stdout_path = temp_dir / "stdout.txt"
        stderr_path = temp_dir / "stderr.txt"
        result_path = temp_dir / "result.json"
        stdout_path.write_text("")
        stderr_path.write_text("")

        process = multiprocessing.Process(
            target=_run_python_code_once,
            args=(
                resolved_context_root.as_posix(),
                code,
                stdout_path.as_posix(),
                stderr_path.as_posix(),
                result_path.as_posix(),
            ),
        )
        process.start()
        process.join(timeout_seconds)

        if process.is_alive():
            process.terminate()
            process.join()
            return {
                "success": False,
                "output": _read_captured_stream(stdout_path),
                "stderr": _read_captured_stream(stderr_path),
                "error": f"Python execution timed out after {timeout_seconds} seconds.",
                "stateful_session": False,
            }

        if not result_path.exists():
            return {
                "success": False,
                "output": _read_captured_stream(stdout_path),
                "stderr": _read_captured_stream(stderr_path),
                "error": "Python execution exited without returning a result.",
                "stateful_session": False,
            }

        result = json.loads(result_path.read_text(encoding="utf-8"))
        result["output"] = _read_captured_stream(stdout_path)
        result["stderr"] = _read_captured_stream(stderr_path)
        result["stateful_session"] = False
        return result
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def _python_session_worker(context_root: str, session_dir: str) -> None:
    namespace: dict[str, Any] = {
        "__builtins__": __builtins__,
        "__name__": "__main__",
        "context_root": context_root,
        "Path": Path,
    }
    os.chdir(context_root)
    session_root = Path(session_dir)

    while True:
        if (session_root / "shutdown.flag").exists():
            return

        request_paths = sorted(session_root.glob("request_*.json"))
        if not request_paths:
            time.sleep(0.05)
            continue

        request_path = request_paths[0]
        try:
            payload = json.loads(request_path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            request_path.unlink(missing_ok=True)
            continue
        request_path.unlink(missing_ok=True)

        request_id = payload.get("request_id")
        response_path = session_root / f"response_{request_id}.json"
        result = _execute_code_in_namespace(
            namespace=namespace,
            context_root=context_root,
            code=str(payload.get("code", "")),
            stdout_path=Path(str(payload["stdout_path"])),
            stderr_path=Path(str(payload["stderr_path"])),
        )
        response_path.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")


@dataclass(slots=True)
class PythonSession:
    context_root: Path
    _process: multiprocessing.Process | None = field(init=False, default=None)
    _lock: threading.Lock = field(init=False, default_factory=threading.Lock)
    _request_counter: int = field(init=False, default=0)
    _temp_dir: Path = field(init=False)

    def __post_init__(self) -> None:
        self.context_root = self.context_root.resolve()
        runtime_root = self.context_root / ".data_agent_runtime"
        runtime_root.mkdir(parents=True, exist_ok=True)
        self._temp_dir = runtime_root / f"python_session_{uuid.uuid4().hex}"
        self._temp_dir.mkdir(parents=True, exist_ok=True)

    def _ensure_started(self) -> None:
        if not self._temp_dir.exists():
            runtime_root = self.context_root / ".data_agent_runtime"
            runtime_root.mkdir(parents=True, exist_ok=True)
            self._temp_dir = runtime_root / f"python_session_{uuid.uuid4().hex}"
            self._temp_dir.mkdir(parents=True, exist_ok=True)
        if self._process is not None and self._process.is_alive():
            return
        self._process = multiprocessing.Process(
            target=_python_session_worker,
            args=(
                self.context_root.as_posix(),
                self._temp_dir.as_posix(),
            ),
        )
        self._process.start()

    def execute(self, code: str, *, timeout_seconds: int = 30) -> dict[str, Any]:
        with self._lock:
            self._ensure_started()
            self._request_counter += 1
            request_id = self._request_counter
            stdout_path = self._temp_dir / f"stdout_{request_id}.txt"
            stderr_path = self._temp_dir / f"stderr_{request_id}.txt"
            request_path = self._temp_dir / f"request_{request_id}.json"
            response_path = self._temp_dir / f"response_{request_id}.json"
            stdout_path.write_text("")
            stderr_path.write_text("")
            request_path.write_text(
                json.dumps(
                    {
                        "request_id": request_id,
                        "code": code,
                        "stdout_path": stdout_path.as_posix(),
                        "stderr_path": stderr_path.as_posix(),
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            deadline = time.time() + timeout_seconds
            while time.time() < deadline:
                if response_path.exists():
                    result = json.loads(response_path.read_text(encoding="utf-8"))
                    response_path.unlink(missing_ok=True)
                    break
                if self._process is None or not self._process.is_alive():
                    result = {
                        "success": False,
                        "error": "Python session exited unexpectedly.",
                        "traceback": "",
                    }
                    break
                time.sleep(0.05)
            else:
                self.close(force=True)
                return {
                    "success": False,
                    "output": _read_captured_stream(stdout_path),
                    "stderr": _read_captured_stream(stderr_path),
                    "error": (
                        f"Python session execution timed out after {timeout_seconds} seconds. "
                        "The session was reset."
                    ),
                    "stateful_session": True,
                    "session_reset": True,
                }

            payload = dict(result)
            payload["output"] = _read_captured_stream(stdout_path)
            payload["stderr"] = _read_captured_stream(stderr_path)
            payload["stateful_session"] = True
            payload.setdefault("session_reset", False)
            return payload

    def close(self, *, force: bool = False) -> None:
        with self._lock:
            process = self._process
            if process is not None and process.is_alive():
                if force:
                    process.terminate()
                    process.join(timeout=1.0)
                    if process.is_alive():
                            process.kill()
                            process.join()
                else:
                    (self._temp_dir / "shutdown.flag").write_text("shutdown", encoding="utf-8")
                    process.join(timeout=1.0)
                    if process.is_alive():
                        process.terminate()
                        process.join(timeout=1.0)
                        if process.is_alive():
                            process.kill()
                            process.join()
            self._process = None
            if self._temp_dir.exists():
                shutil.rmtree(self._temp_dir, ignore_errors=True)
