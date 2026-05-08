"""Processing element (PE) for the systolic array."""


class ProcessingElement:
    """Single MAC unit with accumulator and cycle tracking."""

    def __init__(self) -> None:
        """Initialize accumulator and cycle counter."""
        self.accumulator = 0
        self.cycles = 0

    def tick(self, a_in: int, b_in: int) -> tuple[int, int]:
        """Run one cycle: MAC and pass-through inputs to neighbors."""
        product = int(a_in) * int(b_in)
        self.accumulator += int(product)
        self.cycles += 1
        return int(a_in), int(b_in)

    def reset(self) -> None:
        """Reset the accumulator and cycle counter."""
        self.accumulator = 0
        self.cycles = 0
