"""Pluggable interest profiles: one Markdown file per profile in `profiles/`.

    python -m rag_papers.profiles validate [profiles/*.md]

A profile is added by pull request and is only ever read by the code, never executed. Format:

    ---
    name: Advances in current frontier models
    categories: [cs.CL, cs.LG]
    maintainer: github-login
    ---
    ## Description
    One paragraph, the text the papers are matched against.

    ## Includes
    - topic that counts

    ## Excludes
    - topic that does not count

    ## Examples
    relevant: 2501.09223, 2610.09587
    not_relevant: 2610.10622

Example IDs are shown to the classifier, so they are never used to evaluate it.
"""

import argparse
import hashlib
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import yaml

PROFILE_DIR = Path("profiles")
SLUG = re.compile(r"^[a-z0-9][a-z0-9-]{1,48}$")
CATEGORY = re.compile(r"^[a-z\-]+(\.[A-Za-z\-]+)?$")
ARXIV_ID = re.compile(r"^\d{4}\.\d{4,5}$")
SECTIONS = ("description", "includes", "excludes", "examples")
MAX_CHARS = 6000


class ProfileError(ValueError):
    pass


@dataclass(frozen=True)
class Profile:
    slug: str
    name: str
    categories: tuple[str, ...]
    description: str
    includes: tuple[str, ...] = ()
    excludes: tuple[str, ...] = ()
    relevant_examples: tuple[str, ...] = ()
    not_relevant_examples: tuple[str, ...] = ()
    maintainer: str = ""
    extra: dict = field(default_factory=dict, compare=False)

    @property
    def example_ids(self) -> set[str]:
        return set(self.relevant_examples) | set(self.not_relevant_examples)

    @property
    def fingerprint(self) -> str:
        """Changes whenever anything the classifier sees changes, so results are tied to a version."""
        text = "\n".join(
            [
                self.description,
                *self.includes,
                "|",
                *self.excludes,
                "|",
                *self.relevant_examples,
                "|",
                *self.not_relevant_examples,
                "|",
                *self.categories,
            ]
        )
        return hashlib.sha256(text.encode()).hexdigest()[:12]

    def as_prompt(self) -> str:
        lines = [f"Profile: {self.name}", "", self.description]
        if self.includes:
            lines += ["", "Counts as relevant:", *[f"- {x}" for x in self.includes]]
        if self.excludes:
            lines += ["", "Does NOT count as relevant:", *[f"- {x}" for x in self.excludes]]
        return "\n".join(lines)


def _sections(body: str) -> dict[str, str]:
    found: dict[str, list[str]] = {}
    current = None
    for line in body.splitlines():
        heading = re.match(r"^##\s+(.+?)\s*$", line)
        if heading:
            current = heading.group(1).strip().lower()
            if current in found:
                raise ProfileError(f"section '{current}' appears twice")
            found[current] = []
        elif current is not None:
            found[current].append(line)
    unknown = set(found) - set(SECTIONS)
    if unknown:
        raise ProfileError(f"unknown sections: {', '.join(sorted(unknown))}")
    return {k: "\n".join(v).strip() for k, v in found.items()}


def _bullets(text: str) -> tuple[str, ...]:
    items = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if not line.startswith("- "):
            raise ProfileError(f"expected a '- ' bullet, got: {line[:60]}")
        items.append(line[2:].strip())
    return tuple(items)


def _examples(text: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    lists = {"relevant": (), "not_relevant": ()}
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        key, _, value = line.partition(":")
        key = key.strip().lower()
        if key not in lists:
            raise ProfileError(f"examples lines must start with 'relevant:' or 'not_relevant:', got: {line[:60]}")
        ids = tuple(x.strip() for x in value.split(",") if x.strip())
        bad = [x for x in ids if not ARXIV_ID.match(x)]
        if bad:
            raise ProfileError(f"not arXiv IDs without version: {', '.join(bad)}")
        lists[key] = ids
    overlap = set(lists["relevant"]) & set(lists["not_relevant"])
    if overlap:
        raise ProfileError(f"IDs listed as both relevant and not relevant: {', '.join(sorted(overlap))}")
    return lists["relevant"], lists["not_relevant"]


def parse(text: str, slug: str) -> Profile:
    if not SLUG.match(slug):
        raise ProfileError(f"file name '{slug}' must be lowercase letters, digits and dashes")
    if len(text) > MAX_CHARS:
        raise ProfileError(f"profile is longer than {MAX_CHARS} characters")
    match = re.match(r"^---\n(.*?)\n---\n(.*)$", text, re.S)
    if not match:
        raise ProfileError("profile must start with a '---' front matter block")
    try:
        meta = yaml.safe_load(match.group(1)) or {}
    except yaml.YAMLError as exc:
        raise ProfileError(f"front matter is not valid YAML: {exc}") from None
    if not isinstance(meta, dict):
        raise ProfileError("front matter must be key: value pairs")
    name = str(meta.get("name") or "").strip()
    if not name:
        raise ProfileError("front matter needs a 'name'")
    categories = meta.get("categories")
    if not isinstance(categories, list) or not categories:
        raise ProfileError("front matter needs 'categories', a non-empty list such as [cs.CL]")
    categories = tuple(str(c).strip() for c in categories)
    bad = [c for c in categories if not CATEGORY.match(c)]
    if bad:
        raise ProfileError(f"not arXiv categories: {', '.join(bad)}")
    sections = _sections(match.group(2))
    description = sections.get("description", "").strip()
    if len(description) < 40:
        raise ProfileError("the Description section needs at least a sentence (40+ characters)")
    relevant, not_relevant = _examples(sections.get("examples", ""))
    known = {"name", "categories", "maintainer"}
    return Profile(
        slug=slug,
        name=name,
        categories=categories,
        description=description,
        includes=_bullets(sections.get("includes", "")),
        excludes=_bullets(sections.get("excludes", "")),
        relevant_examples=relevant,
        not_relevant_examples=not_relevant,
        maintainer=str(meta.get("maintainer") or ""),
        extra={k: v for k, v in meta.items() if k not in known},
    )


def load(path: Path) -> Profile:
    return parse(path.read_text(encoding="utf-8"), path.stem)


def load_all(directory: Path = PROFILE_DIR) -> list[Profile]:
    return [load(p) for p in sorted(directory.glob("*.md"))]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=["validate"])
    parser.add_argument("paths", nargs="*", type=Path)
    args = parser.parse_args()
    paths = args.paths or sorted(PROFILE_DIR.glob("*.md"))
    failed = False
    for path in paths:
        try:
            profile = load(path)
            print(f"ok  {path}: {profile.name} ({', '.join(profile.categories)})")
        except (ProfileError, OSError, UnicodeDecodeError) as exc:
            failed = True
            print(f"::error file={path}::{exc}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
