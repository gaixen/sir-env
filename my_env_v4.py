from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MyEnvV4Action:
    message: str


@dataclass
class MyEnvV4Observation:
    echoed_message: str


@dataclass
class MyEnvV4Result:
    observation: MyEnvV4Observation
    reward: float
    done: bool


class MyEnvV4Env:
    """Compatibility adapter used by organizer sample `inference.py`.

    This mirrors the expected async API:
    - `await MyEnvV4Env.from_docker_image(...)`
    - `await env.reset()`
    - `await env.step(MyEnvV4Action(...))`
    - `await env.close()`

    Behavior follows the sample environment contract:
    - Observation echoes the last message
    - Reward is proportional to message length: `0.1 * len(message)`
    """

    def __init__(self, image_name: str | None = None):
        self.image_name = image_name
        self._closed = False
        self._step_count = 0
        self._max_steps = 10_000  # practically unbounded for sample driver
        self._last_echo = ""

    @classmethod
    async def from_docker_image(cls, image_name: str | None = None) -> "MyEnvV4Env":
        return cls(image_name=image_name)

    async def reset(self) -> MyEnvV4Result:
        self._ensure_open()
        self._step_count = 0
        self._last_echo = ""
        return MyEnvV4Result(
            observation=MyEnvV4Observation(echoed_message=self._last_echo),
            reward=0.0,
            done=False,
        )

    async def step(self, action: MyEnvV4Action) -> MyEnvV4Result:
        self._ensure_open()
        self._step_count += 1

        msg = action.message if action and action.message is not None else ""
        self._last_echo = str(msg)
        reward = float(len(self._last_echo) * 0.1)
        done = self._step_count >= self._max_steps

        return MyEnvV4Result(
            observation=MyEnvV4Observation(echoed_message=self._last_echo),
            reward=reward,
            done=done,
        )

    async def close(self) -> None:
        self._closed = True

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("Environment is closed")
