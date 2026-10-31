"""Block-parser tests: dialogue, parentheticals, cue extensions, headings, empty and malformed input."""
import pytest

from app.screenplay import (
    _heading_metadata, build_draft_shot_plan, parse_cue, parse_heading, parse_screenplay,
    parse_screenplay_document,
)


def blocks(script, scene=0):
    doc = parse_screenplay_document(script)
    return [(e.type, e.text) for e in doc.scenes[scene].elements]


def codes(script):
    return [w.code for w in parse_screenplay_document(script).warnings]


# --------------------------------------------------------------------------- dialogue

def test_action_and_dialogue_blocks():
    doc = parse_screenplay_document("INT. CAFE - DAY\n\nMAYA sits alone.\n\nMAYA\nYou came.\n\nARJUN\nI said I would.\n")
    assert [(e.type, e.text, e.character) for e in doc.scenes[0].elements] == [
        ("action", "MAYA sits alone.", None),
        ("character", "MAYA", "Maya"),
        ("dialogue", "You came.", "Maya"),
        ("character", "ARJUN", "Arjun"),
        ("dialogue", "I said I would.", "Arjun"),
    ]
    assert [c.name for c in doc.characters] == ["Maya", "Arjun"]
    assert doc.characters[0].dialogue_count == 1


def test_multiline_dialogue_is_one_block():
    assert blocks("INT. ROOM - DAY\n\nMAYA\nI waited.\nAll night.\n\nShe leaves.") == [
        ("character", "MAYA"), ("dialogue", "I waited. All night."), ("action", "She leaves.")]


def test_indented_pdf_style_dialogue():
    script = "INT. ROOM - DAY\n\n          MAYA\n     I waited\n     all night.\n\nShe leaves."
    assert blocks(script)[:2] == [("character", "MAYA"), ("dialogue", "I waited all night.")]


def test_back_to_back_speakers_without_blank_line_use_known_cues():
    script = "INT. ROOM - DAY\n\nMAYA\nHi.\n\nARJUN\nHey.\nMAYA\nAgain?\n"
    assert blocks(script) == [("character", "MAYA"), ("dialogue", "Hi."), ("character", "ARJUN"),
                              ("dialogue", "Hey."), ("character", "MAYA"), ("dialogue", "Again?")]


def test_dialogue_with_mixed_case_name_tokens():
    doc = parse_screenplay_document("INT. ROOM - DAY\n\nDR. McCOY\nHe's dead, Jim.\n\nMRS. O'NEIL\nOh no.")
    assert [c.name for c in doc.characters] == ["Dr. McCoy", "Mrs. O'Neil"]
    assert [e.type for e in doc.scenes[0].elements] == ["character", "dialogue", "character", "dialogue"]


def test_uppercase_action_lines_are_not_cues():
    script = "INT. ROOM - DAY\n\nBANG!\nThe door flies open.\n\nTHE WHOLE ROOM GOES DARK\nNobody moves.\n\nSILENCE.\n"
    doc = parse_screenplay_document(script)
    assert doc.characters == []
    assert all(e.type == "action" for e in doc.scenes[0].elements)


def test_dual_dialogue_and_forced_cue():
    script = "INT. ROOM - DAY\n\nMAYA\nNow!\n\nARJUN ^\nNow!\n\n@McGregor\nWhat?\n"
    doc = parse_screenplay_document(script)
    cues = [e for e in doc.scenes[0].elements if e.type == "character"]
    assert [c.dual for c in cues] == [False, True, False]
    assert cues[2].character == "McGregor"


# --------------------------------------------------------------------------- parentheticals

def test_parenthetical_before_and_inside_dialogue():
    script = "INT. ROOM - DAY\n\nMAYA\n(quietly)\nI know.\n(beat)\nI always knew.\n"
    assert blocks(script) == [("character", "MAYA"), ("parenthetical", "(quietly)"), ("dialogue", "I know."),
                              ("parenthetical", "(beat)"), ("dialogue", "I always knew.")]


def test_parenthetical_inline_with_dialogue():
    assert blocks("INT. ROOM - DAY\n\nMAYA\n(smiling) Hello there.\n") == [
        ("character", "MAYA"), ("parenthetical", "(smiling)"), ("dialogue", "Hello there.")]


def test_multiline_parenthetical():
    assert blocks("INT. ROOM - DAY\n\nMAYA\n(looking out at\nthe rain)\nIt's late.\n") == [
        ("character", "MAYA"), ("parenthetical", "(looking out at the rain)"), ("dialogue", "It's late.")]


def test_unclosed_parenthetical_warns_and_keeps_dialogue():
    script = "INT. ROOM - DAY\n\nMAYA\n(whispering\nCome here.\n"
    doc = parse_screenplay_document(script)
    assert "UNCLOSED_PARENTHETICAL" in [w.code for w in doc.warnings]
    assert ("dialogue", "Come here.") in [(e.type, e.text) for e in doc.scenes[0].elements]


def test_orphan_parenthetical_is_action_with_warning():
    script = "INT. ROOM - DAY\n\n(beat)\n\nMaya waits."
    assert blocks(script)[0] == ("action", "(beat)")
    assert "ORPHAN_PARENTHETICAL" in codes(script)


def test_parenthetical_only_cue_warns():
    script = "INT. ROOM - DAY\n\nMAYA\n(nods)\n\nShe leaves."
    assert "CUE_WITHOUT_DIALOGUE" in codes(script)


# --------------------------------------------------------------------------- cue suffixes

@pytest.mark.parametrize("cue,name,exts", [
    ("MAYA (V.O.)", "Maya", ["V.O."]),
    ("MAYA (O.S.)", "Maya", ["O.S."]),
    ("MAYA (O.C.)", "Maya", ["O.C."]),
    ("MAYA (CONT'D)", "Maya", ["CONT'D"]),
    ("MAYA (CONT’D)", "Maya", ["CONT'D"]),        # curly apostrophe
    ("MAYA (CONTD)", "Maya", ["CONT'D"]),
    ("MAYA (cont'd)", "Maya", ["CONT'D"]),
    ("MAYA (VO)", "Maya", ["V.O."]),
    ("MAYA (V.O.) (CONT'D)", "Maya", ["V.O.", "CONT'D"]),
    ("MAYA V.O.", "Maya", ["V.O."]),               # bare suffix without parentheses
    ("MAYA CONT'D", "Maya", ["CONT'D"]),
    ("MAYA (ON PHONE)", "Maya", ["ON PHONE"]),
    ("YOUNG MAYA (V.O.)", "Young Maya", ["V.O."]),
])
def test_cue_extensions(cue, name, exts):
    key, display, extensions, _dual = parse_cue(cue)
    assert display == name and extensions == exts and key == name.upper()


def test_cue_suffixes_merge_into_one_character():
    script = ("INT. ROOM - DAY\n\nMAYA (V.O.)\nIt began here.\n\nMAYA (O.S.)\nHello?\n\n"
              "She enters.\n\nMAYA (CONT'D)\nAnyone?\n\nMAYA (CONT’D)\nNo.\n")
    doc = parse_screenplay_document(script)
    assert len(doc.characters) == 1
    maya = doc.characters[0]
    assert maya.name == "Maya" and maya.dialogue_count == 4
    assert maya.extensions_seen == ["V.O.", "O.S.", "CONT'D"]
    dialogue = [e for e in doc.scenes[0].elements if e.type == "dialogue"]
    assert [d.extensions for d in dialogue] == [["V.O."], ["O.S."], ["CONT'D"], ["CONT'D"]]


def test_more_and_continued_page_artifacts_are_ignored():
    script = "INT. ROOM - DAY\n\nMAYA\nI was going to say\n(MORE)\nMAYA (CONT'D)\nsomething important.\n\n12.\nCONTINUED:\n\nShe stops."
    doc = parse_screenplay_document(script)
    types = [(e.type, e.text) for e in doc.scenes[0].elements]
    assert ("dialogue", "I was going to say") in types and ("dialogue", "something important.") in types
    assert not any("MORE" in t or "CONTINUED" in t or t == "12." for _, t in types)
    assert "PAGE_ARTIFACTS_REMOVED" in codes(script)


# --------------------------------------------------------------------------- scene headings

@pytest.mark.parametrize("heading,ie,location,time", [
    ("INT. CAFE - DAY", "INT", "CAFE", "DAY"),
    ("EXT. STREET -- NIGHT", "EXT", "STREET", "NIGHT"),
    ("EXT. STREET — NIGHT", "EXT", "STREET", "NIGHT"),
    ("INT./EXT. CAR - MOVING", "INT/EXT", "CAR", "MOVING"),
    ("EXT./INT. CAR - DAY", "INT/EXT", "CAR", "DAY"),
    ("INT/EXT CAR - DAY", "INT/EXT", "CAR", "DAY"),
    ("I/E. CAR - DAY", "INT/EXT", "CAR", "DAY"),
    ("EST. CITY SKYLINE - DAWN", "EST", "CITY SKYLINE", "DAWN"),
    ("INTERIOR. LAB - NIGHT", "INT", "LAB", "NIGHT"),
    ("EXT STREET - DAY", "EXT", "STREET", "DAY"),
    ("int. kitchen - night", "INT", "kitchen", "night"),
    ("INT. HOUSE - KITCHEN - NIGHT", "INT", "HOUSE - KITCHEN", "NIGHT"),
    ("INT. CAFE DAY", "INT", "CAFE", "DAY"),
    ("INT. CAFE - MOMENTS LATER", "INT", "CAFE", "MOMENTS LATER"),
    ("INT. WAREHOUSE", "INT", "WAREHOUSE", "Unspecified"),
])
def test_heading_variants(heading, ie, location, time):
    parts = parse_heading(heading)
    assert parts is not None, heading
    assert (parts["ie"], parts["location"], parts["time_of_day"]) == (ie, location, time)


@pytest.mark.parametrize("heading,tag", [
    ("12 INT. CAFE - DAY 12", "12"),
    ("12A. EXT. ROOF - NIGHT", "12A"),
    ("INT. CAFE - DAY #7B#", "7B"),
    ("SCENE 4: THE ROOFTOP", "4"),
])
def test_numbered_headings(heading, tag):
    parts = parse_heading(heading)
    assert parts["tag"] == tag
    assert not parts["location"].endswith(tag) or heading.startswith("SCENE")


def test_forced_fountain_heading():
    doc = parse_screenplay_document("INT. CAFE - DAY\n\nShe waits.\n\n.FLASHBACK - THE WEDDING\n\nMusic plays.")
    assert [s.heading for s in doc.scenes] == ["INT. CAFE - DAY", "FLASHBACK - THE WEDDING"]
    assert "HEADING_NO_INT_EXT" in [w.code for w in doc.warnings]


@pytest.mark.parametrize("prose", [
    "Internet cafes line the street.",
    "Interior designers argue in the hallway.",
    "External hard drives sit on the desk.",
    "...and then she leaves.",
    "integer overflow, the screen says.",
])
def test_prose_is_not_a_heading(prose):
    assert parse_heading(prose) is None


def test_heading_without_blank_line_before_it_still_splits_scenes():
    doc = parse_screenplay_document("INT. CAFE - DAY\nMaya sips coffee.\nEXT. STREET - NIGHT\nRain.")
    assert [s.heading for s in doc.scenes] == ["INT. CAFE - DAY", "EXT. STREET - NIGHT"]
    assert doc.scenes[0].body == "Maya sips coffee."


def test_heading_missing_location_and_duplicate_scene_numbers_warn():
    script = "INT.\n\nDark.\n\n5 INT. CAFE - DAY 5\n\nA.\n\n5 EXT. CAFE - DAY 5\n\nB."
    found = codes(script)
    assert "HEADING_NO_LOCATION" in found and "DUPLICATE_SCENE_NUMBER" in found


def test_scene_line_ranges_and_metadata():
    doc = parse_screenplay_document("INT. CAFE - DAY\n\nA.\n\nEXT. STREET - NIGHT\n\nB.\n")
    first, second = doc.scenes
    assert (first.start_line, first.end_line, second.start_line) == (1, 4, 5)
    assert (second.interior_exterior, second.location, second.time_of_day) == ("EXT", "STREET", "NIGHT")


def test_backwards_compatible_heading_metadata():
    assert _heading_metadata("INT. CAFE - DAY") == ("CAFE", "DAY")
    assert _heading_metadata("Not a heading") == ("Unspecified", "Unspecified")


# --------------------------------------------------------------------------- transitions & fountain extras

def test_transitions():
    script = "FADE IN:\n\nINT. CAFE - DAY\n\nShe waits.\n\nCUT TO:\n\nEXT. STREET - NIGHT\n\n> BURN TO WHITE.\n\nFADE OUT."
    doc = parse_screenplay_document(script)
    assert doc.front_matter[0].type == "transition"  # FADE IN: before the first scene is not a prologue scene
    assert len(doc.scenes) == 2
    assert [e.type for e in doc.scenes[0].elements] == ["action", "transition"]
    assert [(e.type, e.text) for e in doc.scenes[1].elements] == [("transition", "BURN TO WHITE."), ("transition", "FADE OUT.")]


def test_fountain_title_page_notes_sections_and_boneyard():
    script = ("Title: Rain\nAuthor: Someone\n\nINT. CAFE - DAY\n\n# Act One\n\n= Maya waits.\n\n[[check props]]\n\n"
              "/* cut this\nMAYA\nOld line.\n*/\n\nMAYA\nNew line.\n")
    doc = parse_screenplay_document(script)
    assert doc.title_page == {"title": "Rain", "author": "Someone"}
    types = [(e.type, e.text) for e in doc.scenes[0].elements]
    assert types == [("section", "Act One"), ("synopsis", "Maya waits."), ("note", "check props"),
                     ("character", "MAYA"), ("dialogue", "New line.")]


# --------------------------------------------------------------------------- empty and malformed input

@pytest.mark.parametrize("script", ["", "   ", "\n\n\t\n", "\ufeff"])
def test_empty_input(script):
    doc = parse_screenplay_document(script)
    assert doc.scenes == [] and doc.characters == []
    assert [w.code for w in doc.warnings][0] in ("EMPTY_SCRIPT",)
    assert parse_screenplay(script) == ([], [])


def test_only_notes_has_no_content():
    doc = parse_screenplay_document("[[a note]]\n\n# Section\n")
    assert doc.scenes == []
    assert "NO_SCRIPT_CONTENT" in [w.code for w in doc.warnings]


def test_no_headings_becomes_one_scene_with_warning():
    doc = parse_screenplay_document("Maya walks in.\n\nMAYA\nHello?\n")
    assert len(doc.scenes) == 1 and doc.scenes[0].characters == ["Maya"]
    assert "NO_SCENE_HEADINGS" in [w.code for w in doc.warnings]


def test_content_before_first_heading_becomes_prologue():
    doc = parse_screenplay_document("A voice in the dark.\n\nINT. CAFE - DAY\n\nLight.")
    assert [s.heading for s in doc.scenes] == ["PROLOGUE / PRE-SCENE MATERIAL", "INT. CAFE - DAY"]
    assert "PRE_HEADING_CONTENT" in [w.code for w in doc.warnings]


def test_cue_followed_by_blank_line_warns():
    script = "INT. ROOM - DAY\n\nMAYA\nHi.\n\nMAYA\n\nHello?\n"
    assert "CUE_WITHOUT_DIALOGUE" in codes(script)


def test_cue_at_end_of_file_is_action():
    doc = parse_screenplay_document("INT. ROOM - DAY\n\nShe turns.\n\nMAYA (V.O.)")
    assert doc.scenes[0].elements[-1].type == "action"
    assert "CUE_WITHOUT_DIALOGUE" in [w.code for w in doc.warnings]


def test_empty_scene_warns():
    assert "EMPTY_SCENE" in codes("INT. CAFE - DAY\n\nEXT. STREET - NIGHT\n\nRain.")


def test_unclosed_boneyard_keeps_text():
    doc = parse_screenplay_document("INT. CAFE - DAY\n\n/* draft\nShe waits.\n")
    assert "UNCLOSED_BONEYARD" in [w.code for w in doc.warnings]
    assert "She waits." in doc.scenes[0].elements[-1].text


def test_crlf_tabs_and_control_characters():
    script = "INT. CAFE - DAY\r\n\r\n\tMAYA\r\n\tHi\x07 there.\r\n"
    doc = parse_screenplay_document(script)
    assert [(e.type, e.text) for e in doc.scenes[0].elements] == [("character", "MAYA"), ("dialogue", "Hi there.")]
    assert "CONTROL_CHARS_REMOVED" in [w.code for w in doc.warnings]


def test_similar_character_names_warn():
    assert "SIMILAR_CHARACTER_NAMES" in codes("INT. ROOM - DAY\n\nDR. SMITH\nHi.\n\nDR SMITH\nHi again.")


@pytest.mark.parametrize("garbage", [
    "))))(((( \n\n((\n)\n", "INT.\nINT.\nINT.", "^^^\n@\n.\n>\n<\n=\n#\n[[\n]]", "A" * 50_000,
    "\x00\x01\x02 binary \xff\xfe", "MAYA\n" * 200, "(" * 500 + "\nMAYA\n(" + "\n" * 10,
], ids=["parens", "bare-int", "fountain-symbols", "huge-line", "binary", "repeated-cue", "open-parens"])
def test_malformed_input_never_crashes(garbage):
    doc = parse_screenplay_document(garbage)
    assert isinstance(doc.warnings, list)
    build_draft_shot_plan(doc.scenes, "style", "16:9")


def test_character_ids_are_stable_across_parses():
    script = "INT. ROOM - DAY\n\nMAYA\nHi.\n"
    assert parse_screenplay_document(script).characters[0].id == parse_screenplay_document(script).characters[0].id


def test_shot_plan_uses_parsed_dialogue():
    scenes, _ = parse_screenplay("INT. ROOM - NIGHT\n\nMAYA\n(softly)\nHello.\n")
    coverage = [s for s in build_draft_shot_plan(scenes, "noir", "16:9") if s.shot_type == "character_coverage"]
    assert coverage[0].audio_notes == "Source dialogue: Hello."
