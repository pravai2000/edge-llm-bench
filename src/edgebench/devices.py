"""Device abstraction: the same benchmark code runs locally (Mac) or over SSH (Jetson)."""
import os
import shlex
import subprocess
from typing import List, Optional


class Device:
    def __init__(self, name: str, kind: str = "local", host: Optional[str] = None,
                 user: Optional[str] = None):
        if kind not in ("local", "ssh"):
            raise ValueError(f"unknown device kind: {kind}")
        if kind == "ssh" and not host:
            raise ValueError("ssh device needs a host")
        self.name, self.kind, self.host, self.user = name, kind, host, user

    @property
    def target(self) -> str:
        return f"{self.user}@{self.host}" if self.user else str(self.host)

    def expand(self, path: str) -> str:
        """~ is expanded locally for local devices; left for the remote shell over SSH."""
        return os.path.expanduser(path) if self.kind == "local" else path

    def wrap(self, cmd: List[str]) -> List[str]:
        if self.kind == "local":
            return cmd
        # Keep a leading ~ expandable by the remote shell: quote everything else.
        quoted = " ".join(c if c.startswith("~/") and " " not in c else shlex.quote(c) for c in cmd)
        return ["ssh", "-o", "BatchMode=yes", self.target, quoted]

    def run(self, cmd: List[str], timeout: Optional[float] = None, check: bool = True):
        return subprocess.run(self.wrap(cmd), capture_output=True, text=True,
                              timeout=timeout, check=check)

    def run_shell(self, command: str, timeout: Optional[float] = None, check: bool = True):
        """Run a shell string (used for power-mode commands like 'sudo nvpmodel -m 0 && ...')."""
        if self.kind == "local":
            return subprocess.run(command, shell=True, capture_output=True, text=True,
                                  timeout=timeout, check=check)
        return subprocess.run(["ssh", "-o", "BatchMode=yes", self.target, command],
                              capture_output=True, text=True, timeout=timeout, check=check)

    def popen(self, cmd: List[str]) -> subprocess.Popen:
        return subprocess.Popen(self.wrap(cmd), stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, text=True)
