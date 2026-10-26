"""Persistent registry of known people (name <-> numeric face ID), stored as CSV."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Person:
    id: int
    name: str


class PeopleRegistry:
    FIELDS = ("Name", "ID")

    def __init__(self, path: Path):
        self.path = Path(path)

    def all(self) -> list[Person]:
        if not self.path.exists():
            return []
        with self.path.open(encoding="utf-8-sig", newline="") as f:
            return [Person(int(r["ID"]), r["Name"]) for r in csv.DictReader(f)]

    def get(self, person_id: int) -> Person | None:
        return next((p for p in self.all() if p.id == person_id), None)

    def next_id(self) -> int:
        return max((p.id for p in self.all()), default=0) + 1

    def add(self, name: str) -> Person:
        person = Person(self.next_id(), name.strip())
        self.path.parent.mkdir(parents=True, exist_ok=True)
        is_new = not self.path.exists() or self.path.stat().st_size == 0
        with self.path.open("a", encoding="utf-8-sig", newline="") as f:
            writer = csv.writer(f)
            if is_new:
                writer.writerow(self.FIELDS)
            writer.writerow((person.name, person.id))
        return person
