from collections import defaultdict
from typing import Any, ClassVar, Literal


class Logger:
    _level: ClassVar[int] = 0
    _collect: ClassVar[bool] = False
    _num_indent: ClassVar[int] = 0

    indent_str: ClassVar[str] = "| "
    sink: ClassVar[defaultdict[str, list[Any]]] = defaultdict(list)

    @classmethod
    def activate(cls, level: int = 1, collect: bool = False, clear: bool = True) -> None:
        if level < 0:
            raise ValueError

        cls._level = level
        cls._collect = collect
        if clear:
            cls.clear()

    @classmethod
    def deactivate(cls) -> None:
        cls._level = 0
        cls._collect = False

    @classmethod
    def is_active(cls, level: int = 1) -> bool:
        return cls._collect or level <= cls._level

    @classmethod
    def log(cls, message: str, level: int = 1, indent: Literal["in", "out"] | None = None) -> None:
        if indent == "out":
            cls._num_indent = max(0, cls._num_indent - 1)

        if level <= cls._level:
            print(f"{cls.indent_str * cls._num_indent}{message}")

        if indent == "in":
            cls._num_indent += 1

    @classmethod
    def dump(cls, key: str, value: Any) -> None:
        if cls._collect:
            cls.sink[key].append(value)

    @classmethod
    def clear(cls) -> None:
        cls.sink.clear()
        cls._num_indent = 0
