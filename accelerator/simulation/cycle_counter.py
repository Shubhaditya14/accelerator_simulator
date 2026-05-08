"""Cycle counter utilities for simulated accelerator operations."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class OperationLog:
    """Record of a single operation's cycle count."""

    name: str
    cycles: int
    input_shape: tuple[int, ...]
    output_shape: tuple[int, ...]


class CycleCounter:
    """Accumulate cycle counts across accelerator operations."""

    def __init__(self) -> None:
        self.logs: list[OperationLog] = []

    def reset(self) -> None:
        """Clear all logged operations."""
        self.logs = []

    def log_operation(
        self,
        name: str,
        cycles: int,
        input_shape: tuple[int, ...],
        output_shape: tuple[int, ...],
    ) -> None:
        """Record a named operation and its cycle count."""
        self.logs.append(OperationLog(name, int(cycles), input_shape, output_shape))

    def total_cycles(self) -> int:
        """Return total cycles across all operations."""
        return sum(log.cycles for log in self.logs)

    def summary(self) -> None:
        """Print a summary table of cycles per operation."""
        total = self.total_cycles()
        print("Operation | Cycles | % of total")
        print("-" * 40)
        for log in self.logs:
            pct = 0.0 if total == 0 else 100.0 * log.cycles / total
            print(f"{log.name:25s} | {log.cycles:8d} | {pct:6.2f}%")
        print("-" * 40)
        print(f"Total cycles: {total}")

    def estimate_latency_ms(self, clock_freq_mhz: float = 12.0) -> float:
        """Estimate latency in milliseconds for a given clock frequency."""
        total = self.total_cycles()
        return total / (clock_freq_mhz * 1e6) * 1000.0
