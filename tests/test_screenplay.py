from app.screenplay import parse_screenplay, build_draft_shot_plan


def test_parses_multiple_scenes_and_character_cues():
    script = '''INT. CAFE - DAY
MAYA sits at a table.

MAYA
You came.

ARJUN
I said I would.

EXT. CAFE - NIGHT
Rain falls on the empty street.
'''
    scenes, characters = parse_screenplay(script)
    assert len(scenes) == 2
    assert scenes[0].heading == "INT. CAFE - DAY"
    assert {c.name for c in characters} == {"Maya", "Arjun"}
    assert scenes[0].characters == ["Maya", "Arjun"]


def test_unformatted_script_becomes_one_scene():
    scenes, characters = parse_screenplay("A person walks into a room. Nobody speaks.")
    assert len(scenes) == 1
    assert scenes[0].number == 1
    assert characters == []


def test_shot_plan_is_editable_and_linked_to_scene():
    scenes, _ = parse_screenplay("INT. ROOM - NIGHT\nA door opens.\n\nMAYA\nHello.")
    shots = build_draft_shot_plan(scenes, "film noir", "16:9")
    assert len(shots) >= 3
    assert all(shot.scene_id == scenes[0].id for shot in shots)
    assert all(shot.status == "planned" for shot in shots)
