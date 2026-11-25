"""Power logging. Two sources: Jetson `tegrastats` and macOS `powermetrics` (needs sudo)."""
import os
import re
import statistics
import threading
import time
from dataclasses import dataclass
from typing import List, Optional

# Jetson Orin Nano prints total module input power as "VDD_IN 5123mW/5123mW" (instant/average).
# Older boards use POM_5V_IN. The rail name can differ by JetPack release: check raw tegrastats.
_TEGRA_PATTERNS = [re.compile(r"VDD_IN\s+(\d+)mW"), re.compile(r"POM_5V_IN\s+(\d+)/")]
# Apple Silicon powermetrics: "Combined Power (CPU + GPU + ANE): 1234 mW"
_MAC_RE = re.compile(r"Combined Power \(CPU \+ GPU \+ ANE\):\s*(\d+)\s*mW")
_MAC_CPU_RE = re.compile(r"^CPU Power:\s*(\d+)\s*mW", re.M)
# powermetrics prints "GPU Power" more than once per sample (two sections); the values are close, so average them.
_MAC_GPU_RE = re.compile(r"^GPU Power:\s*(\d+)\s*mW", re.M)


def parse_tegrastats(text: str) -> List[float]:
    watts = []
    for line in text.splitlines():
        for pat in _TEGRA_PATTERNS:
            m = pat.search(line)
            if m:
                watts.append(int(m.group(1)) / 1000.0)
                break
    return watts


def parse_powermetrics(text: str) -> List[float]:
    return [int(m.group(1)) / 1000.0 for m in _MAC_RE.finditer(text)]


def parse_powermetrics_split(text: str):
    """Return (cpu_watts, gpu_watts) lists, one value per sample (GPU lines averaged within a sample)."""
    cpu = [int(x) / 1000.0 for x in _MAC_CPU_RE.findall(text)]
    gpu = [int(x) / 1000.0 for x in _MAC_GPU_RE.findall(text)]
    return cpu, gpu


@dataclass
class PowerStats:
    mean_w: Optional[float]
    max_w: Optional[float]
    samples: int
    error: Optional[str] = None
    mean_cpu_w: Optional[float] = None
    mean_gpu_w: Optional[float] = None
    window: str = "full"      # "timed" when samples were restricted to the benchmark's timed section


def summarize(watts: List[float]) -> PowerStats:
    if not watts:
        return PowerStats(None, None, 0)
    return PowerStats(round(statistics.fmean(watts), 3), max(watts), len(watts))


class PowerLogger:
    """Start before a benchmark, stop after; returns PowerStats for that window.

    tegrastats: a streaming subprocess that is terminated at stop().
    powermetrics: a background thread that repeatedly runs ONE bounded sample (`-n 1`). Each call exits
    by itself, so output is flushed (powermetrics buffers when piped) and no root process has to be
    signalled. Failures (for example sudo needing a password) are captured in PowerStats.error.
    """

    def __init__(self, device, source: str, interval_ms: int = 500):
        self.device, self.source, self.interval_ms = device, source, interval_ms
        self.proc = None
        self._thread = None
        self._stop_evt = threading.Event()
        self._chunks: List[tuple] = []   # (unix time the sample finished, raw text)
        self._error: Optional[str] = None

    def _command(self):
        if self.source == "tegrastats":
            return ["tegrastats", "--interval", str(self.interval_ms)]
        if self.source == "powermetrics":
            cmd = ["/usr/bin/powermetrics", "--samplers", "cpu_power,gpu_power",
                   "-i", str(self.interval_ms), "-n", "1"]
            if self.device.kind == "local" and os.geteuid() != 0:
                cmd = ["sudo", "-n"] + cmd   # -n: fail instead of prompting; see README
            return cmd
        raise ValueError(f"unknown power source: {self.source}")

    def _sample_loop(self):
        cmd = self._command()
        while not self._stop_evt.is_set():
            try:
                r = self.device.run(cmd, timeout=30, check=False)
            except Exception as e:  # timeout, missing binary, ssh failure
                self._error = f"{type(e).__name__}: {e}"
                return
            if r.returncode != 0:
                self._error = ((r.stderr or r.stdout or "").strip() or f"exit {r.returncode}")[:300]
                return
            self._chunks.append((time.time(), r.stdout))

    def start(self):
        if self.source == "none":
            return
        if self.source == "powermetrics":
            self._thread = threading.Thread(target=self._sample_loop, daemon=True)
            self._thread.start()
        else:
            self.proc = self.device.popen(self._command())

    def stop(self, window=None) -> PowerStats:
        """window=(t_start, t_end) as unix times restricts powermetrics samples to that interval (used for
        runtimes that report their own timed section, so startup and model load do not dilute the mean)."""
        if self.source == "none":
            return PowerStats(None, None, 0)
        if self.source == "powermetrics":
            self._stop_evt.set()
            if self._thread:
                self._thread.join(timeout=35)
            chunks, used = self._chunks, "full"
            if window:
                lo, hi = window[0], window[1] + self.interval_ms / 1000.0 + 0.2
                inside = [c for c in self._chunks if lo <= c[0] <= hi]
                if inside:
                    chunks, used = inside, "timed"
            text = "\n".join(t for _, t in chunks)
            stats = summarize(parse_powermetrics(text))
            stats.window = used
            cpu, gpu = parse_powermetrics_split(text)
            if cpu:
                stats.mean_cpu_w = round(statistics.fmean(cpu), 3)
            if gpu:
                stats.mean_gpu_w = round(statistics.fmean(gpu), 3)
        else:
            if self.proc is None:
                return PowerStats(None, None, 0)
            self.proc.terminate()
            try:
                out, err = self.proc.communicate(timeout=10)
            except Exception:
                self.proc.kill()
                out, err = self.proc.communicate()
            stats = summarize(parse_tegrastats(out or ""))
            if not stats.samples and err:
                self._error = err.strip()[:300]
        if not stats.samples and not self._error:
            self._error = "no power samples parsed; check the tool's raw output format"
        stats.error = self._error
        return stats
