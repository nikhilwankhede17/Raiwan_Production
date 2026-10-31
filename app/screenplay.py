"""Screenplay block parser.

Parses plain-text and Fountain-style screenplays into typed blocks (scene headings,
action, character cues, parentheticals, dialogue, transitions, ...), groups them into
scenes, infers character records and reports uncertain parses as warnings.

It is heuristic: anything it cannot classify confidently becomes `action` and, where
useful, a warning explains why so the user can correct it manually.
"""
from __future__ import annotations

import hashlib
import re
import uuid
from dataclasses import dataclass, field

from app.models import Character, ParseWarning, Scene, ScriptElement, Shot

# --------------------------------------------------------------------------- patterns

_IE = r"INT\.?\s*/\s*EXT|EXT\.?\s*/\s*INT|INTERIOR|EXTERIOR|INT|EXT|EST|I\s*/\s*E|E\s*/\s*I"
_HEADING_RE = re.compile(
    rf"^(?:(?P<num>\d+[A-Z]?)[.):]?\s+)?(?P<ie>{_IE})(?P<sep>\.|\s|$|[-–—])\s*[-–—.]?\s*(?P<rest>.*?)\s*$",
    re.IGNORECASE,
)
_SCENE_WORD_RE = re.compile(r"^(?:SCENE|SC\.)\s+(?P<num>\d+[A-Z]?)\b[\s.:\-–—]*(?P<rest>.*?)\s*$", re.IGNORECASE)
# Kept for backwards compatibility with code that imported it.
SCENE_HEADING = re.compile(r"^\s*(?:(?:INT\.?|EXT\.?|INT\.?/EXT\.?|EXT\.?/INT\.?|I/E)\s+.+|SCENE\s+\d+.*)$", re.IGNORECASE)

_TAG_RE = re.compile(r"\s*#(?P<tag>[\w.\-]+)#\s*$")
_TIME_SPLIT_RE = re.compile(r"\s+[-–—]+\s+|\s*--\s*|\s+[–—]\s*|\s*[–—]\s+")
_TIME_WORDS = {
    "DAY", "NIGHT", "MORNING", "EVENING", "AFTERNOON", "DUSK", "DAWN", "SUNSET", "SUNRISE",
    "CONTINUOUS", "LATER", "MOMENTS LATER", "SAME", "SAME TIME", "MAGIC HOUR", "NOON", "MIDNIGHT",
}
_TRANSITION_RE = re.compile(
    r"^(?:[A-Z][A-Z0-9 .'’\-]*\sTO:|FADE IN:?|FADE OUT\.?|FADE TO BLACK\.?|CUT TO BLACK\.?|SMASH CUT\.?"
    r"|MATCH CUT\.?|DISSOLVE\.?|INTERCUT(?:\s.*)?|BACK TO SCENE\.?|THE END\.?)$"
)
_PAGE_ARTIFACT_RE = re.compile(r"^(?:\(MORE\)|\(?CONTINUED\)?:?(?:\s*\(\d+\))?|\d{1,4}[A-Z]?\.?)$")
_BARE_EXT_RE = re.compile(r"\s+(V\.\s?O\.?|O\.\s?S\.?|O\.\s?C\.?|CONT['’]?D\.?|CONT\.)$")
_NON_CUE_RE = re.compile(
    r"^(?:THE END|END|CONTINUED|MORE|BLACK|BLACKNESS|LATER|CONTINUOUS|SILENCE|DARKNESS"
    r"|(?:INSERT|MONTAGE|END MONTAGE|FLASHBACK|END FLASHBACK|SERIES OF SHOTS|END SERIES OF SHOTS|SUPER"
    r"|TITLE CARD|TITLE|CHYRON|BACK TO|SPLIT SCREEN|ON SCREEN|LEGEND|SUBTITLE)(?:\s.*)?)$"
)
_TITLE_KEY_RE = re.compile(
    r"^(Title|Credit|Author|Authors|Source|Draft date|Date|Contact|Copyright|Notes|Revision|Format):",
    re.IGNORECASE,
)
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_EXT_ALIASES = {"VO": "V.O.", "OS": "O.S.", "OC": "O.C.", "CONTD": "CONT'D", "CONT": "CONT'D", "CONTINUING": "CONT'D"}
_CONTENT_TYPES = {"action", "character", "dialogue", "parenthetical", "centered"}

Line = tuple[int, str]  # (1-based line number, stripped text)


@dataclass
class ParsedScreenplay:
    scenes: list[Scene] = field(default_factory=list)
    characters: list[Character] = field(default_factory=list)
    warnings: list[ParseWarning] = field(default_factory=list)
    front_matter: list[ScriptElement] = field(default_factory=list)
    title_page: dict[str, str] = field(default_factory=dict)


# --------------------------------------------------------------------------- helpers

def _uid() -> str:
    return uuid.uuid4().hex[:12]


def _hash(*parts: object) -> str:
    return hashlib.sha1("|".join(str(p) for p in parts).encode("utf-8")).hexdigest()[:12]


def character_id_for(cue_key: str) -> str:
    """Stable ID so re-parsing the same script keeps the same character records."""
    return "c" + _hash("character", cue_key)[:11]


class _Warnings:
    def __init__(self) -> None:
        self.items: list[ParseWarning] = []

    def add(self, code: str, message: str, line: int | None = None, severity: str = "warning",
            scene_id: str | None = None, key: object = "") -> None:
        self.items.append(ParseWarning(
            id=_hash("parse", code, line, key), code=code, severity=severity, message=message,
            line=line, scene_id=scene_id, entity_type="scene" if scene_id else None, entity_id=scene_id,
        ))


def _norm_ext(raw: str) -> str:
    value = raw.strip().upper().replace("’", "'")
    return _EXT_ALIASES.get(re.sub(r"[^A-Z]", "", value), value)


def _cue_key(name: str) -> str:
    return re.sub(r"\s+", " ", name.upper().replace("’", "'")).strip()


def parse_heading(text: str) -> dict | None:
    """Return heading parts if `text` is a scene heading, else None."""
    t = text.strip()
    forced = t.startswith(".") and not t.startswith("..") and len(t) > 1
    if forced:
        t = t[1:].strip()
    tag = None
    m_tag = _TAG_RE.search(t)
    if m_tag:
        tag, t = m_tag.group("tag"), t[:m_tag.start()].rstrip()

    m = _HEADING_RE.match(t)
    ie = "Unspecified"
    rest = t
    if m:
        prefix = m.group("ie")
        # Lowercase prefixes are only trusted with a dot ("int. cafe"), to avoid catching prose.
        if prefix != prefix.upper() and m.group("sep") != "." and not forced:
            m = None
        else:
            letters = re.sub(r"[^A-Z]", "", prefix.upper())
            ie = {"INTERIOR": "INT", "EXTERIOR": "EXT", "IE": "INT/EXT", "EI": "INT/EXT",
                  "INTEXT": "INT/EXT", "EXTINT": "INT/EXT"}.get(letters, letters)
            rest = m.group("rest")
            if m.group("num"):
                tag = tag or m.group("num")
                rest = re.sub(rf"\s+{re.escape(m.group('num'))}$", "", rest)
    if not m:
        m2 = _SCENE_WORD_RE.match(t)
        if m2:
            tag = tag or m2.group("num")
            rest = m2.group("rest")
        elif not forced:
            return None

    # A heading line that is clearly a sentence of prose is not a heading.
    if not forced and m and len(rest) > 120:
        return None

    pieces = [p.strip() for p in _TIME_SPLIT_RE.split(rest) if p.strip()] if rest else []
    location, time_of_day = "Unspecified", "Unspecified"
    if len(pieces) > 1:
        location, time_of_day = " - ".join(pieces[:-1]), pieces[-1]
    elif pieces and pieces[0].upper() in _TIME_WORDS:
        time_of_day = pieces[0]
    elif pieces:
        words = pieces[0].split()
        for size in (2, 1):
            if len(words) > size and " ".join(words[-size:]).upper() in _TIME_WORDS:
                location, time_of_day = " ".join(words[:-size]), " ".join(words[-size:])
                break
        else:
            location = pieces[0]
    return {"forced": forced, "ie": ie, "location": location, "time_of_day": time_of_day,
            "tag": tag, "text": (text.strip()[1:].strip() if forced else text.strip())}


def _heading_metadata(heading: str) -> tuple[str, str]:
    """Backwards-compatible helper: (location, time_of_day)."""
    parts = parse_heading(heading)
    if not parts:
        return "Unspecified", "Unspecified"
    return parts["location"] or "Unspecified", parts["time_of_day"]


def parse_cue(text: str) -> tuple[str, str, list[str], bool] | None:
    """If `text` looks like a character cue return (key, display_name, extensions, dual)."""
    t = text.strip()
    forced = t.startswith("@")
    if forced:
        t = t[1:].strip()
    dual = t.endswith("^")
    if dual:
        t = t[:-1].rstrip()
    extensions: list[str] = []
    while (m := re.search(r"\(([^()]*)\)\s*$", t)) and t[:m.start()].strip():
        extensions.insert(0, _norm_ext(m.group(1)))
        t = t[:m.start()].rstrip()
    while m := _BARE_EXT_RE.search(t):
        extensions.insert(0, _norm_ext(m.group(1)))
        t = t[:m.start()].rstrip()
    name = re.sub(r"\s+", " ", t).strip()
    if not name or "(" in name or ")" in name or len(name) > 40 or len(name.split()) > 5:
        return None
    if not forced:
        if not any(c.isupper() for c in name):
            return None  # needs cased letters; uncased scripts can force cues with "@"
        if any(c.islower() for c in re.sub(r"\bMc(?=[A-Z])", "MC", name)):
            return None
        if name[-1] in ".!?,:;-–—" or " - " in name or ":" in name:
            return None
        if _NON_CUE_RE.match(name) or _TRANSITION_RE.match(name):
            return None
        # Long all-caps lines are usually action ("THE DOOR BURSTS OPEN"); allow them only with an extension.
        if len(name.split()) > 3 and not extensions:
            return None
    display = name if forced else _title_name(name)
    return _cue_key(name), display, extensions, dual


def _title_name(name: str) -> str:
    """'DR. MCCOY' / 'DR. McCOY' -> 'Dr. McCoy'; "MRS. O'NEIL" -> "Mrs. O'Neil"."""
    titled = name.upper().title()
    return re.sub(r"\bMc([a-z])", lambda m: "Mc" + m.group(1).upper(), titled)


def _special(line_no: int, t: str) -> ScriptElement | None:
    """Classify single-line block types that never belong to dialogue."""
    if re.fullmatch(r"={3,}", t):
        return ScriptElement(type="page_break", text=t, line=line_no)
    if t.startswith("#") and not re.match(r"^#\d", t):
        return ScriptElement(type="section", text=t.lstrip("#").strip(), line=line_no)
    if t.startswith("=") and not t.startswith("=="):
        return ScriptElement(type="synopsis", text=t[1:].strip(), line=line_no)
    if t.startswith("[[") and t.endswith("]]"):
        return ScriptElement(type="note", text=t[2:-2].strip(), line=line_no)
    if m := re.fullmatch(r">\s*(.*?)\s*<", t):
        return ScriptElement(type="centered", text=m.group(1), line=line_no)
    if t.startswith(">"):
        return ScriptElement(type="transition", text=t[1:].strip(), line=line_no)
    if t.startswith("!"):
        return ScriptElement(type="action", text=t[1:].strip(), line=line_no)
    if _TRANSITION_RE.match(t):
        return ScriptElement(type="transition", text=t, line=line_no)
    if parse_heading(t):
        return ScriptElement(type="scene_heading", text=t[1:].strip() if t.startswith(".") else t, line=line_no)
    return None


# --------------------------------------------------------------------------- pre-processing

def _prepare_lines(script: str, warnings: _Warnings) -> tuple[list[str], list[Line], dict[str, str], list[ScriptElement]]:
    text = script.replace("\r\n", "\n").replace("\r", "\n").lstrip("\ufeff")
    text = text.replace("\t", "    ").replace("\u00a0", " ")
    cleaned = _CONTROL_RE.sub("", text)
    if len(cleaned) != len(text):
        warnings.add("CONTROL_CHARS_REMOVED", f"Removed {len(text) - len(cleaned)} invisible control character(s).",
                     severity="info")
    raw_lines = cleaned.split("\n")

    # Fountain boneyard /* ... */ is ignored. If it is never closed, keep the text and warn.
    stripped, inside, start = [], False, None
    for idx, line in enumerate(raw_lines):
        out, pos = "", 0
        while pos <= len(line):
            if inside:
                end = line.find("*/", pos)
                if end == -1:
                    break
                inside, pos = False, end + 2
            else:
                s = line.find("/*", pos)
                if s == -1:
                    out += line[pos:]
                    break
                out, inside, start, pos = out + line[pos:s], True, idx + 1, s + 2
        stripped.append(out)
    if inside:
        warnings.add("UNCLOSED_BONEYARD", "A '/*' comment is never closed with '*/'; its text was kept as script.",
                     line=start)
        stripped = raw_lines

    lines: list[Line] = [(i + 1, s.strip()) for i, s in enumerate(stripped)]

    # Fountain title page: key: value block at the very top.
    title_page: dict[str, str] = {}
    title_elements: list[ScriptElement] = []
    first = next((i for i, (_, t) in enumerate(lines) if t), None)
    if first is not None and _TITLE_KEY_RE.match(lines[first][1]):
        key = None
        i = first
        while i < len(lines) and lines[i][1]:
            line_no, t = lines[i]
            m = _TITLE_KEY_RE.match(t)
            if m:
                key = m.group(1).lower()
                title_page[key] = t[m.end():].strip()
            elif key:
                title_page[key] = (title_page[key] + "\n" + t).strip()
            title_elements.append(ScriptElement(type="title_page", text=t, line=line_no))
            lines[i] = (line_no, "")
            i += 1

    # Drop PDF page artefacts such as "(MORE)", "CONTINUED:" and bare page numbers.
    artifacts = 0
    kept: list[Line] = []
    for line_no, t in lines:
        if t and _PAGE_ARTIFACT_RE.match(t):
            artifacts += 1
            continue
        kept.append((line_no, t))
    if artifacts:
        warnings.add("PAGE_ARTIFACTS_REMOVED",
                     f"Ignored {artifacts} page artefact line(s) such as (MORE), CONTINUED: or bare page numbers.",
                     severity="info")
    return raw_lines, kept, title_page, title_elements


def _paragraphs(lines: list[Line]) -> list[list[Line]]:
    paragraphs: list[list[Line]] = []
    current: list[Line] = []
    for line_no, t in lines:
        if not t:
            if current:
                paragraphs.append(current)
                current = []
            continue
        # A scene heading always starts a new block, even without a blank line before it.
        special = _special(line_no, t) if current else None
        if special and special.type == "scene_heading":
            paragraphs.append(current)
            current = []
        current.append((line_no, t))
    if current:
        paragraphs.append(current)
    return paragraphs


def _known_cues(paragraphs: list[list[Line]]) -> set[str]:
    """Pre-pass: cue names found in unambiguous positions (block start, followed by text)."""
    known: set[str] = set()
    for para in paragraphs:
        for i, (line_no, t) in enumerate(para[:-1]):
            if i > 0 and not _special(*para[i - 1]):
                break
            if _special(line_no, t):
                continue
            cue = parse_cue(t)
            if cue and not _special(*para[i + 1]):
                known.add(cue[0])
            break
    return known


# --------------------------------------------------------------------------- block parsing

def _parse_blocks(paragraphs: list[list[Line]], known: set[str], warnings: _Warnings,
                  names: dict[str, str]) -> list[ScriptElement]:
    elements: list[ScriptElement] = []

    def add_action(line_no: int, text: str, continue_block: bool) -> None:
        if continue_block and elements and elements[-1].type == "action":
            elements[-1].text += "\n" + text
        else:
            elements.append(ScriptElement(type="action", text=text, line=line_no))
        if text.startswith("(") and text.endswith(")"):
            warnings.add("ORPHAN_PARENTHETICAL",
                         f"Parenthetical '{text[:60]}' is not under a character cue; treated as action.", line=line_no)

    for para in paragraphs:
        i = 0
        action_open = False
        while i < len(para):
            line_no, t = para[i]
            special = _special(line_no, t)
            if special and special.type != "action":
                elements.append(special)
                action_open = False
                i += 1
                continue
            if special:  # forced action
                add_action(line_no, special.text, action_open)
                action_open = True
                i += 1
                continue

            cue = parse_cue(t)
            prev_is_header = bool(elements) and elements[-1].type in ("scene_heading", "transition") and not action_open
            at_block_start = i == 0 or prev_is_header
            has_next = i + 1 < len(para) and not _special(*para[i + 1])
            if cue and has_next and (at_block_start or cue[0] in known):
                key, display, extensions, dual = cue
                display = names.setdefault(key, display)
                elements.append(ScriptElement(type="character", text=t, line=line_no, character=display,
                                              extensions=extensions, dual=dual))
                spoken = 0
                buffer: list[Line] = []

                def flush() -> None:
                    nonlocal spoken
                    if buffer:
                        elements.append(ScriptElement(type="dialogue", text=" ".join(b[1] for b in buffer),
                                                      line=buffer[0][0], character=display, extensions=extensions))
                        spoken += 1
                        buffer.clear()

                j = i + 1
                while j < len(para):
                    ln2, t2 = para[j]
                    if _special(ln2, t2) and _special(ln2, t2).type in ("transition", "scene_heading"):
                        break
                    nxt = parse_cue(t2)
                    if nxt and nxt[0] in known and j + 1 < len(para) and (buffer or spoken):
                        break  # next speaker without a blank line in between
                    if t2.startswith("("):
                        flush()
                        if ")" not in t2:
                            close = next((k for k in range(j + 1, len(para)) if ")" in para[k][1]), None)
                            if close is None:
                                warnings.add("UNCLOSED_PARENTHETICAL",
                                             f"Parenthetical '{t2[:60]}' has no closing ')'.", line=ln2)
                                elements.append(ScriptElement(type="parenthetical", text=t2, line=ln2,
                                                              character=display))
                                j += 1
                                continue
                            t2 = " ".join(p[1] for p in para[j:close + 1])
                            j = close
                        end = t2.index(")") + 1
                        elements.append(ScriptElement(type="parenthetical", text=t2[:end], line=ln2,
                                                      character=display))
                        remainder = t2[end:].strip()
                        if remainder:
                            buffer.append((ln2, remainder))
                    else:
                        buffer.append((ln2, t2))
                    j += 1
                flush()
                if spoken == 0:
                    warnings.add("CUE_WITHOUT_DIALOGUE",
                                 f"Character cue '{t}' has a parenthetical but no dialogue.", line=line_no)
                action_open = False
                i = j
                continue

            if cue and i == 0 and len(para) == 1 and (cue[0] in known or cue[2]):
                warnings.add("CUE_WITHOUT_DIALOGUE",
                             f"'{t}' looks like a character cue but no dialogue follows directly "
                             "(blank line after the cue?). Treated as action.", line=line_no)
            add_action(line_no, t, action_open)
            action_open = True
            i += 1
    return elements


# --------------------------------------------------------------------------- scenes & characters

def parse_screenplay_document(script: str) -> ParsedScreenplay:
    """Parse a screenplay into scenes, characters and warnings. Never raises on bad input."""
    warnings = _Warnings()
    result = ParsedScreenplay()
    if not script or not script.lstrip("\ufeff").strip():
        warnings.add("EMPTY_SCRIPT", "The screenplay is empty.", severity="error")
        result.warnings = warnings.items
        return result

    raw_lines, lines, result.title_page, title_elements = _prepare_lines(script, warnings)
    paragraphs = _paragraphs(lines)
    known = _known_cues(paragraphs)
    names: dict[str, str] = {}
    elements = _parse_blocks(paragraphs, known, warnings, names)
    result.front_matter.extend(title_elements)

    if not any(e.type in _CONTENT_TYPES or e.type == "scene_heading" for e in elements):
        warnings.add("NO_SCRIPT_CONTENT", "No action, dialogue or scene headings were found.", severity="error")
        result.front_matter.extend(elements)
        result.warnings = warnings.items
        return result

    # Group into (heading_element | None, elements)
    groups: list[tuple[ScriptElement | None, list[ScriptElement]]] = []
    pre: list[ScriptElement] = []
    for el in elements:
        if el.type == "scene_heading":
            groups.append((el, []))
        elif groups:
            groups[-1][1].append(el)
        else:
            pre.append(el)

    last_line = len(raw_lines)
    first_content_line = next((e.line for e in pre if e.type in _CONTENT_TYPES), None)
    if not groups:
        warnings.add("NO_SCENE_HEADINGS",
                     "No INT./EXT. scene headings found; the whole script was placed in one draft scene.")
        groups = [(None, pre)]
        pre_title = "SCENE 1 — SCRIPT NOT FORMATTED AS SCREENPLAY"
    elif first_content_line is not None:
        warnings.add("PRE_HEADING_CONTENT",
                     "Action or dialogue appears before the first scene heading; it was kept as a prologue scene.",
                     line=first_content_line)
        groups.insert(0, (None, pre))
        pre_title = "PROLOGUE / PRE-SCENE MATERIAL"
    else:
        result.front_matter.extend(pre)
        pre_title = ""

    heading_lines = [h.line for h, _ in groups if h is not None]
    stats: dict[str, dict] = {}
    seen_tags: dict[str, int] = {}
    for number, (heading_el, scene_elements) in enumerate(groups, start=1):
        scene_id = _uid()
        if heading_el is not None:
            parts = parse_heading(heading_el.text) or parse_heading("." + heading_el.text) or {}
            heading = heading_el.text
            start = heading_el.line
            later = [ln for ln in heading_lines if ln > start]
            end = (later[0] - 1) if later else last_line
            body = "\n".join(raw_lines[start:end]).strip()
            if not parts.get("location") or parts.get("location") == "Unspecified":
                warnings.add("HEADING_NO_LOCATION", f"Scene heading '{heading}' has no location.",
                             line=start, scene_id=scene_id)
            if parts.get("time_of_day", "Unspecified") == "Unspecified":
                warnings.add("HEADING_NO_TIME", f"Scene heading '{heading}' has no time of day (e.g. '- DAY').",
                             line=start, severity="info", scene_id=scene_id)
            if parts.get("ie", "Unspecified") == "Unspecified":
                warnings.add("HEADING_NO_INT_EXT", f"Scene heading '{heading}' does not say INT. or EXT.",
                             line=start, severity="info", scene_id=scene_id)
            tag = parts.get("tag")
            if tag:
                if tag in seen_tags:
                    warnings.add("DUPLICATE_SCENE_NUMBER",
                                 f"Scene number '{tag}' is used by scene {seen_tags[tag]} and scene {number}.",
                                 line=start, scene_id=scene_id)
                seen_tags.setdefault(tag, number)
        else:
            parts = {}
            heading = pre_title
            start = first_content_line or 1
            end = (heading_lines[0] - 1) if heading_lines else last_line
            body = "\n".join(raw_lines[start - 1:end]).strip()

        if not any(e.type in _CONTENT_TYPES for e in scene_elements):
            warnings.add("EMPTY_SCENE", f"Scene {number} '{heading}' has no action or dialogue.",
                         line=start, severity="info", scene_id=scene_id)

        cast: list[str] = []
        for el in scene_elements:
            if el.type != "character" or not el.character:
                continue
            key = _cue_key(el.character)
            entry = stats.setdefault(key, {"name": el.character, "scenes": [], "count": 0, "exts": [],
                                           "order": len(stats)})
            entry["count"] += 1
            for ext in el.extensions:
                if ext not in entry["exts"]:
                    entry["exts"].append(ext)
            if number not in entry["scenes"]:
                entry["scenes"].append(number)
            if el.character not in cast:
                cast.append(el.character)

        result.scenes.append(Scene(
            id=scene_id, number=number, heading=heading, body=body, characters=cast,
            location=parts.get("location") or "Unspecified", time_of_day=parts.get("time_of_day", "Unspecified"),
            interior_exterior=parts.get("ie", "Unspecified"), scene_tag=parts.get("tag"),
            start_line=start, end_line=end, elements=scene_elements,
        ))

    # Flag cue names that differ only by punctuation/spacing, e.g. "DR. SMITH" vs "DR SMITH".
    squashed: dict[str, str] = {}
    for key in stats:
        flat = re.sub(r"[^A-Z0-9]", "", key)
        if flat in squashed and squashed[flat] != key:
            warnings.add("SIMILAR_CHARACTER_NAMES",
                         f"Character cues '{squashed[flat]}' and '{key}' may be the same person.", key=flat)
        squashed.setdefault(flat, key)

    if any(h is not None for h, _ in groups) and not stats:
        warnings.add("NO_CHARACTER_CUES",
                     "No character cues were detected. Cues must be on their own line in capitals, directly "
                     "above the dialogue (or prefixed with '@').", severity="info")

    ordered = sorted(stats.items(), key=lambda kv: (-len(kv[1]["scenes"]), -kv[1]["count"], kv[1]["order"]))
    result.characters = [
        Character(id=character_id_for(key), name=entry["name"], cue_name=key,
                  description="Define age, appearance, personality and backstory in the character bible.",
                  scene_numbers=entry["scenes"], dialogue_count=entry["count"], extensions_seen=entry["exts"])
        for key, entry in ordered
    ]
    result.warnings = warnings.items
    return result


def parse_screenplay(script: str) -> tuple[list[Scene], list[Character]]:
    """Backwards-compatible API: (scenes, characters)."""
    parsed = parse_screenplay_document(script)
    return parsed.scenes, parsed.characters


# --------------------------------------------------------------------------- shot plan

def _scene_action_text(scene: Scene) -> str:
    action = " ".join(e.text for e in scene.elements if e.type == "action")
    return action or scene.body


def _dialogue_for(scene: Scene, character: str, limit: int = 3) -> list[str]:
    return [e.text for e in scene.elements if e.type == "dialogue" and e.character == character][:limit]


def build_draft_shot_plan(scenes: list[Scene], visual_style: str, aspect_ratio: str) -> list[Shot]:
    """Create editable coverage suggestions. These are planning placeholders, not generated media."""
    shots: list[Shot] = []
    for scene in scenes:
        scene_characters = scene.characters
        cast_text = ", ".join(scene_characters) if scene_characters else "no clearly detected characters"
        location = scene.location if scene.location != "Unspecified" else scene.heading
        action_text = _scene_action_text(scene)
        shots.append(Shot(
            id=_uid(), scene_id=scene.id, scene_number=scene.number, number=1,
            shot_type="establishing_wide", duration_seconds=4.0, characters=scene_characters,
            visual_prompt=(f"{visual_style}; {aspect_ratio} cinematic establishing wide shot of {location}; "
                           f"scene tone and lighting should match the screenplay; cast present if appropriate: {cast_text}; "
                           "coherent production design, deliberate composition, no text or subtitles"),
            motion_prompt="Slow controlled push-in or gentle lateral camera move; keep geography and screen direction clear.",
        ))
        number = 2
        for character in scene_characters:
            lines = _dialogue_for(scene, character)
            shots.append(Shot(
                id=_uid(), scene_id=scene.id, scene_number=scene.number, number=number,
                shot_type="character_coverage", duration_seconds=4.0, characters=[character],
                visual_prompt=(f"{visual_style}; medium close-up of {character} at {location}; maintain the approved "
                               f"character reference design; motivated cinematic lighting; {aspect_ratio}; "
                               "natural expression, consistent costume and screen direction, no text"),
                motion_prompt="Subtle eye movement, natural breathing and restrained head motion; avoid exaggerated gestures.",
                audio_notes=("Source dialogue: " + " / ".join(lines)) if lines
                else "Dialogue and ambience to be added during sound pass.",
            ))
            number += 1
        action_lower = action_text.lower()
        if any(token in action_lower for token in ("runs", "running", "opens", "closes", "grabs", "drops", "falls", "hits", "looks at", "reveals")):
            shots.append(Shot(
                id=_uid(), scene_id=scene.id, scene_number=scene.number, number=number,
                shot_type="action_insert", duration_seconds=3.0, characters=scene_characters,
                visual_prompt=(f"{visual_style}; cinematic detail insert of the key physical action in this scene at {location}; "
                               f"action context: {action_text[:350]}; physically plausible hands and objects; no text"),
                motion_prompt="One readable action with a clear start and end pose; stable camera unless motivated by the action.",
            ))
            number += 1
        shots.append(Shot(
            id=_uid(), scene_id=scene.id, scene_number=scene.number, number=number,
            shot_type="reaction_or_transition", duration_seconds=3.0, characters=scene_characters,
            visual_prompt=(f"{visual_style}; cinematic reaction or transition shot at {location}; "
                           f"use the emotional beat implied by this screenplay excerpt: {scene.body[-350:]}; "
                           f"consistent characters and lighting; {aspect_ratio}; no text"),
            motion_prompt="Hold on a readable expression or compose a visual transition; leave handles for editing.",
        ))
    return shots
