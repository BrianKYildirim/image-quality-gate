import time
from types import TracebackType
from typing import NamedTuple


class Stage(NamedTuple):
    name: str
    offset_ns: int  # start of the stage, relative to the timer's creation
    duration_ns: int


class StageTimer:
    """
    Per-request stage timings on time.perf_counter_ns (monotonic, high resolution).

        with timer.stage("decode"):
            ...

    Stages run one after another (they don't nest). A stage is recorded only if
    its block completes without raising. The timer is its own context manager,
    so timing a stage allocates nothing but the recorded tuple.
    """

    __slots__ = ("origin_ns", "_records", "_name", "_start")

    def __init__(self) -> None:
        self.origin_ns = time.perf_counter_ns()
        self._records: list[tuple[str, int, int]] = []
        self._name: str | None = None
        self._start = 0

    def stage(self, name: str) -> "StageTimer":
        if self._name is not None:
            raise RuntimeError(f"stage {name!r} started inside {self._name!r}")
        self._name = name
        return self

    def __enter__(self) -> None:
        self._start = time.perf_counter_ns()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        end = time.perf_counter_ns()
        if exc_type is None:
            self._records.append((self._name or "", self._start, end))
        self._name = None

    @property
    def stages(self) -> list[Stage]:
        origin = self.origin_ns
        return [
            Stage(name, start - origin, end - start)
            for name, start, end in self._records
        ]

    def elapsed_ns(self) -> int:
        return time.perf_counter_ns() - self.origin_ns
