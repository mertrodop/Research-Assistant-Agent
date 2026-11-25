"""Staging area and vault. Only `Vault.commit` writes to the vault."""

from __future__ import annotations

import difflib
import logging
import re
import shutil
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import yaml

from ..schema import OcrDocument
from .note import render_note

log = logging.getLogger(__name__)
MAX_SLUG = 100


def slugify(title: str, fallback: str) -> str:
    """Lowercase ASCII words joined by '-', at most 100 chars, cut at a word boundary."""
    for text in (title, fallback):
        ascii_text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
        slug = ""
        for word in re.findall(r"[a-z0-9]+", ascii_text.lower()):
            joined = f"{slug}-{word}" if slug else word
            if len(joined) > MAX_SLUG:
                slug = slug or word[:MAX_SLUG]
                break
            slug = joined
        if slug:
            return slug
    return "untitled"


def _check_slug(slug: str) -> None:
    if not re.fullmatch(r"[a-z0-9]+(-[a-z0-9]+)*", slug):
        raise ValueError(f"invalid slug: {slug!r}")


def _front(text: str) -> dict:
    return yaml.safe_load(text.split("---", 2)[1]) or {}


@dataclass
class StagedNote:
    slug: str
    note_path: Path
    figure_dir: Path
    status: str


class Vault:
    def __init__(self, root: Path, staging: Path) -> None:
        self.root = Path(root)
        self.staging = Path(staging)

    def _papers(self, base: Path) -> Path:
        return base / "Papers"

    def _sha_at(self, base: Path, slug: str) -> str | None:
        note = self._papers(base) / f"{slug}.md"
        return _front(note.read_text(encoding="utf-8")).get("sha256") if note.is_file() else None

    def _slug(self, document: OcrDocument) -> str:
        base = slugify(document.title, Path(document.source_pdf).stem)
        n = 1
        while True:
            slug = base if n == 1 else f"{base}-{n}"
            owners = {self._sha_at(self.root, slug), self._sha_at(self.staging, slug)} - {None}
            if not owners or document.sha256 in owners:
                return slug
            n += 1

    def stage(self, document: OcrDocument, figures_dir: Path | None = None) -> StagedNote:
        """Render and write the note and its figures under staging/Papers/."""
        slug = self._slug(document)
        text = render_note(document, slug)  # validates frontmatter before any write
        note_path = self._papers(self.staging) / f"{slug}.md"
        figure_dir = self._papers(self.staging) / slug
        note_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.rmtree(figure_dir, ignore_errors=True)
        figures = [f for page in document.pages for f in page.figures]
        if figures and figures_dir is not None:
            figure_dir.mkdir()
            for figure in figures:
                shutil.copy2(Path(figures_dir) / figure.filename, figure_dir / figure.filename)
        note_path.write_text(text, encoding="utf-8")
        return StagedNote(slug, note_path, figure_dir, document.status)

    def diff(self, slug: str) -> str:
        """Unified diff of the staged note against the vault note; full text if new."""
        _check_slug(slug)
        staged = (self._papers(self.staging) / f"{slug}.md").read_text(encoding="utf-8")
        current = self._papers(self.root) / f"{slug}.md"
        if not current.is_file():
            return staged
        return "".join(
            difflib.unified_diff(
                current.read_text(encoding="utf-8").splitlines(keepends=True),
                staged.splitlines(keepends=True),
                "vault",
                "staged",
            )
        )

    def commit(self, slug: str) -> Path:
        """Move the staged note and figures into the vault; rename the raw PDF to <slug>.pdf."""
        _check_slug(slug)
        staged = self._papers(self.staging) / f"{slug}.md"
        if not staged.is_file():
            raise FileNotFoundError(f"nothing staged for {slug}")
        text = staged.read_text(encoding="utf-8")
        front = _front(text)
        source_pdf = Path(front.get("source_pdf", ""))
        if source_pdf.is_file() and source_pdf.name != f"{slug}.pdf":
            renamed = source_pdf.with_name(f"{slug}.pdf")
            if renamed.exists():
                log.warning("%s already exists; left %s unrenamed", renamed, source_pdf)
            else:
                source_pdf.rename(renamed)
                line = yaml.safe_dump({"source_pdf": str(renamed)}, allow_unicode=True).strip()
                text = re.sub(r"(?m)^source_pdf: .*$", lambda _: line, text, count=1)
        target = self._papers(self.root) / f"{slug}.md"
        target.parent.mkdir(parents=True, exist_ok=True)
        figures = self._papers(self.staging) / slug
        if figures.is_dir():
            shutil.rmtree(self._papers(self.root) / slug, ignore_errors=True)
            shutil.move(figures, self._papers(self.root) / slug)
        target.write_text(text, encoding="utf-8")
        staged.unlink()
        # the pipeline's own figure output for this paper
        shutil.rmtree(self.staging / "figures" / str(front.get("sha256", "x"))[:12], ignore_errors=True)
        return target

    def search(self, query: str, limit: int = 5) -> list[dict]:
        words = query.lower().split()
        hits = []
        for note in self._papers(self.root).glob("*.md"):
            text = note.read_text(encoding="utf-8")
            lines = text.splitlines()
            count = sum(text.lower().count(word) for word in words)
            first = next((l for l in lines if any(w in l.lower() for w in words)), "")
            if words and count:
                title = _front(text).get("title", note.stem)
                hits.append((count, {"slug": note.stem, "title": title, "line": first.strip()[:200]}))
        hits.sort(key=lambda hit: -hit[0])
        return [hit for _, hit in hits[:limit]]

    def read(self, slug: str, max_chars: int = 4000) -> str:
        _check_slug(slug)
        note = self._papers(self.root) / f"{slug}.md"
        return note.read_text(encoding="utf-8")[:max_chars]
