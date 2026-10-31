"""API, versioning, autosave and snapshot tests. Each test uses an isolated temporary workspace (conftest.py)."""
import hashlib
import json

import pytest
from fastapi.testclient import TestClient

from app import storage
from app.main import app

SCRIPT = """INT. CAFE - DAY

MAYA sits at a table. She opens a letter.

MAYA
(quietly)
You came.

ARJUN (O.S.)
I said I would.

EXT. STREET - NIGHT

Rain falls on the empty street.
"""


@pytest.fixture
def client():
    return TestClient(app)


def new_project(client, script=SCRIPT, breakdown=True):
    pid = client.post("/api/projects", json={"title": "Test Film"}).json()["id"]
    assert client.put(f"/api/projects/{pid}/script", json={"script": script}).status_code == 200
    if breakdown:
        assert client.post(f"/api/projects/{pid}/breakdown").status_code == 200
    return pid, client.get(f"/api/projects/{pid}").json()


# --------------------------------------------------------------------------- immutable original & versions

def test_first_import_is_read_only_original(client, isolated_workspace):
    pid, project = new_project(client, breakdown=False)
    original = isolated_workspace / pid / "screenplay" / "original.txt"
    assert original.read_text(encoding="utf-8") == SCRIPT.strip()
    assert project["script_versions"][0]["sha256"] == hashlib.sha256(SCRIPT.strip().encode()).hexdigest()
    assert project["current_script_version"] == 1
    with pytest.raises(FileExistsError):
        storage._write_once(original, "overwrite attempt")
    assert client.get(f"/api/projects/{pid}/script/original").text == SCRIPT.strip()


def test_script_edits_create_new_versions_and_never_touch_original(client, isolated_workspace):
    pid, project = new_project(client)
    edited = SCRIPT.replace("You came.", "You actually came.")
    r = client.put(f"/api/projects/{pid}/script", json={"script": edited, "note": "Line tweak",
                                                        "base_revision": project["revision"]})
    assert r.status_code == 200
    body = r.json()
    assert body["current_script_version"] == 2 and body["script"] == edited.strip()
    assert body["script_versions"][1]["note"] == "Line tweak"
    assert (isolated_workspace / pid / "screenplay" / "original.txt").read_text(encoding="utf-8") == SCRIPT.strip()
    assert client.get(f"/api/projects/{pid}/script/versions/1").text == SCRIPT.strip()
    assert client.get(f"/api/projects/{pid}/script/versions/2").text == edited.strip()
    # Editing the script does not wipe the edited breakdown; it is flagged as out of date instead.
    assert len(body["scenes"]) == 2
    codes = [w["code"] for w in client.get(f"/api/projects/{pid}/warnings").json()["warnings"]]
    assert "BREAKDOWN_OUT_OF_DATE" in codes
    reasons = [s["reason"] for s in client.get(f"/api/projects/{pid}/snapshots").json()]
    assert "pre-script-edit" in reasons


def test_saving_identical_script_does_not_add_version(client):
    pid, project = new_project(client, breakdown=False)
    r = client.put(f"/api/projects/{pid}/script", json={"script": SCRIPT, "visual_style": "noir"})
    assert r.json()["current_script_version"] == 1 and len(r.json()["script_versions"]) == 1
    assert r.json()["visual_style"] == "noir"


def test_tampered_version_fails_integrity_check(client, isolated_workspace):
    pid, _ = new_project(client, breakdown=False)
    path = isolated_workspace / pid / "screenplay" / "original.txt"
    path.chmod(0o666)
    path.write_text("tampered", encoding="utf-8")
    r = client.get(f"/api/projects/{pid}/script/versions/1")
    assert r.status_code == 500 and r.json()["code"] == "STORAGE_ERROR"


def test_empty_script_rejected(client):
    pid = client.post("/api/projects", json={"title": "X"}).json()["id"]
    assert client.put(f"/api/projects/{pid}/script", json={"script": ""}).status_code == 422
    assert client.put(f"/api/projects/{pid}/script", json={"script": "   \n  "}).status_code == 422


def test_breakdown_of_unparseable_script_returns_warnings(client):
    pid, _ = new_project(client, script="[[only a note]]", breakdown=False)
    r = client.post(f"/api/projects/{pid}/breakdown")
    assert r.status_code == 400
    assert r.json()["detail"]["code"] == "NOTHING_PARSED"
    assert "NO_SCRIPT_CONTENT" in [w["code"] for w in r.json()["detail"]["warnings"]]


def test_legacy_phase0_project_gets_original_preserved(client, isolated_workspace):
    pid = "a" * 32
    legacy = {"id": pid, "title": "Old", "status": "script_ready", "script": SCRIPT,
              "characters": [], "scenes": [], "shots": []}
    (isolated_workspace / pid).mkdir()
    (isolated_workspace / pid / "project.json").write_text(json.dumps(legacy), encoding="utf-8")
    body = client.get(f"/api/projects/{pid}").json()
    assert body["script_versions"][0]["source"] == "legacy_import"
    assert (isolated_workspace / pid / "screenplay" / "original.txt").read_text(encoding="utf-8") == SCRIPT


# --------------------------------------------------------------------------- breakdown

def test_breakdown_stores_blocks_warnings_and_version(client):
    _, project = new_project(client)
    assert project["breakdown_script_version"] == 1
    scene = project["scenes"][0]
    assert [e["type"] for e in scene["elements"]][:4] == ["action", "character", "parenthetical", "dialogue"]
    assert {c["name"] for c in project["characters"]} == {"Maya", "Arjun"}
    arjun = next(c for c in project["characters"] if c["name"] == "Arjun")
    assert arjun["extensions_seen"] == ["O.S."]
    assert project["shots"] and all(s["status"] == "planned" for s in project["shots"])


def test_rebuild_requires_confirmation_and_keeps_character_bible(client):
    pid, project = new_project(client)
    shot = project["shots"][0]
    rev = client.patch(f"/api/projects/{pid}/shots/{shot['id']}",
                       json={"base_revision": project["revision"], "visual_prompt": "my prompt"}).json()["revision"]
    maya = next(c for c in project["characters"] if c["name"] == "Maya")
    rev = client.patch(f"/api/projects/{pid}/characters/{maya['id']}",
                       json={"base_revision": rev, "description": "Late 20s, architect."}).json()["revision"]
    r = client.post(f"/api/projects/{pid}/breakdown", json={"base_revision": rev})
    assert r.status_code == 409 and r.json()["detail"]["code"] == "BREAKDOWN_WOULD_REPLACE_EDITS"
    r = client.post(f"/api/projects/{pid}/breakdown", json={"base_revision": rev, "force": True})
    assert r.status_code == 200
    rebuilt = r.json()
    kept = next(c for c in rebuilt["characters"] if c["name"] == "Maya")
    assert kept["id"] == maya["id"] and kept["description"] == "Late 20s, architect."
    assert all(s["visual_prompt"] != "my prompt" for s in rebuilt["shots"])
    reasons = [s["reason"] for s in client.get(f"/api/projects/{pid}/snapshots").json()]
    assert "pre-breakdown" in reasons


# --------------------------------------------------------------------------- record edits (autosave)

def test_edit_shot_prompts_duration_and_status_persist(client):
    pid, project = new_project(client)
    shot = project["shots"][1]
    r = client.patch(f"/api/projects/{pid}/shots/{shot['id']}", json={
        "base_revision": project["revision"], "visual_prompt": "Close on Maya's hands", "motion_prompt": "Static",
        "duration_seconds": 5.55, "status": "approved", "characters": ["Maya", " Maya ", ""], "notes": "Hero shot"})
    assert r.status_code == 200
    saved = next(s for s in client.get(f"/api/projects/{pid}").json()["shots"] if s["id"] == shot["id"])
    assert saved["visual_prompt"] == "Close on Maya's hands" and saved["motion_prompt"] == "Static"
    assert saved["duration_seconds"] == 5.55 and saved["status"] == "approved"
    assert saved["characters"] == ["Maya"] and saved["manually_edited"] is True
    assert r.json()["revision"] == project["revision"] + 1


@pytest.mark.parametrize("payload", [
    {"duration_seconds": 0}, {"duration_seconds": -2}, {"duration_seconds": 500}, {"status": "done"},
    {"shot_type": ""}, {"duration_seconds": "abc"},
])
def test_invalid_shot_edits_rejected(client, payload):
    pid, project = new_project(client)
    r = client.patch(f"/api/projects/{pid}/shots/{project['shots'][0]['id']}", json=payload)
    assert r.status_code == 422
    assert client.get(f"/api/projects/{pid}").json()["revision"] == project["revision"]


def test_stale_revision_is_rejected_with_409(client):
    pid, project = new_project(client)
    shot_id = project["shots"][0]["id"]
    assert client.patch(f"/api/projects/{pid}/shots/{shot_id}",
                        json={"base_revision": project["revision"], "notes": "first"}).status_code == 200
    r = client.patch(f"/api/projects/{pid}/shots/{shot_id}",
                     json={"base_revision": project["revision"], "notes": "second (stale)"})
    assert r.status_code == 409 and r.json()["code"] == "REVISION_CONFLICT"
    saved = next(s for s in client.get(f"/api/projects/{pid}").json()["shots"] if s["id"] == shot_id)
    assert saved["notes"] == "first"


def test_edit_scene_fields(client):
    pid, project = new_project(client)
    scene = project["scenes"][0]
    r = client.patch(f"/api/projects/{pid}/scenes/{scene['id']}", json={
        "location": "ROOFTOP CAFE", "time_of_day": "DUSK", "interior_exterior": "EXT",
        "characters": ["Maya"], "notes": "Keep the letter visible."})
    saved = r.json()["scenes"][0]
    assert (saved["location"], saved["time_of_day"], saved["interior_exterior"]) == ("ROOFTOP CAFE", "DUSK", "EXT")
    assert saved["characters"] == ["Maya"] and saved["notes"] == "Keep the letter visible."
    assert saved["elements"] == scene["elements"]  # parsed source blocks are not altered by edits


def test_character_rename_propagates_and_rejects_duplicates(client):
    pid, project = new_project(client)
    maya = next(c for c in project["characters"] if c["name"] == "Maya")
    body = client.patch(f"/api/projects/{pid}/characters/{maya['id']}", json={"name": "Maya Rao"}).json()
    assert "Maya Rao" in body["scenes"][0]["characters"]
    assert all("Maya" not in s["characters"] for s in body["shots"])
    assert any(e.get("character") == "Maya Rao" for e in body["scenes"][0]["elements"])
    r = client.patch(f"/api/projects/{pid}/characters/{maya['id']}", json={"name": "arjun"})
    assert r.status_code == 400


def test_unknown_records_and_bad_ids_404(client):
    pid, _ = new_project(client)
    assert client.patch(f"/api/projects/{pid}/shots/nope", json={"notes": "x"}).status_code == 404
    assert client.get("/api/projects/../../etc").status_code == 404
    assert client.get("/api/projects/" + "z" * 32).status_code == 404


def test_safe_path_blocks_traversal():
    with pytest.raises(ValueError):
        storage.safe_path("a" * 32, "../outside.txt")
    with pytest.raises(ValueError):
        storage.project_dir("../evil")


# --------------------------------------------------------------------------- warnings

def test_validation_warnings_and_acknowledgement(client):
    pid, project = new_project(client)
    shot = project["shots"][0]
    client.patch(f"/api/projects/{pid}/shots/{shot['id']}",
                 json={"visual_prompt": "", "duration_seconds": 30, "characters": ["Ghost"]})
    data = client.get(f"/api/projects/{pid}/warnings").json()
    codes = {w["code"] for w in data["warnings"]}
    assert {"SHOT_EMPTY_PROMPT", "SHOT_LONG_DURATION", "SHOT_UNKNOWN_CHARACTER"} <= codes
    long_warning = next(w for w in data["warnings"] if w["code"] == "SHOT_LONG_DURATION")
    acked = client.post(f"/api/projects/{pid}/warnings/{long_warning['id']}/ack", json={"acknowledged": True}).json()
    assert next(w for w in acked["warnings"] if w["id"] == long_warning["id"])["acknowledged"] is True
    assert acked["summary"]["acknowledged"] == 1
    # Fixing the record makes the warning disappear.
    client.patch(f"/api/projects/{pid}/shots/{shot['id']}", json={"visual_prompt": "Wide of the cafe"})
    codes = {w["code"] for w in client.get(f"/api/projects/{pid}/warnings").json()["warnings"]}
    assert "SHOT_EMPTY_PROMPT" not in codes
    assert client.post(f"/api/projects/{pid}/warnings/unknown/ack", json={}).status_code == 404


def test_parser_warnings_reach_the_api(client):
    script = "INT. CAFE\n\n(beat)\n\nMAYA\n(nods)\n\nShe leaves."
    pid, _ = new_project(client, script=script)
    codes = {w["code"] for w in client.get(f"/api/projects/{pid}/warnings").json()["warnings"]}
    assert {"HEADING_NO_TIME", "ORPHAN_PARENTHETICAL", "CUE_WITHOUT_DIALOGUE"} <= codes


# --------------------------------------------------------------------------- snapshots & manifest

def test_autosnapshot_is_throttled(client, monkeypatch):
    pid, project = new_project(client)
    shot_id = project["shots"][0]["id"]
    for i in range(3):
        client.patch(f"/api/projects/{pid}/shots/{shot_id}", json={"notes": f"edit {i}"})
    autos = [s for s in client.get(f"/api/projects/{pid}/snapshots").json() if s["reason"] == "autosave"]
    assert len(autos) == 1
    monkeypatch.setenv("FILM_HOUSE_AUTOSNAPSHOT_SECONDS", "0")
    client.patch(f"/api/projects/{pid}/shots/{shot_id}", json={"notes": "edit 4"})
    autos = [s for s in client.get(f"/api/projects/{pid}/snapshots").json() if s["reason"] == "autosave"]
    assert len(autos) == 2


def test_autosave_snapshots_are_pruned_but_manual_kept(client, monkeypatch):
    monkeypatch.setattr(storage, "AUTOSAVE_KEEP", 3)
    monkeypatch.setenv("FILM_HOUSE_AUTOSNAPSHOT_SECONDS", "0")
    pid, project = new_project(client)
    client.post(f"/api/projects/{pid}/snapshots", json={"label": "keep me"})
    for i in range(6):
        client.patch(f"/api/projects/{pid}/shots/{project['shots'][0]['id']}", json={"notes": str(i)})
    snaps = client.get(f"/api/projects/{pid}/snapshots").json()
    assert sum(s["reason"] == "autosave" for s in snaps) == 3
    assert any(s["label"] == "keep me" for s in snaps)


def test_snapshot_restore_roundtrip(client):
    pid, project = new_project(client)
    shot_id = project["shots"][0]["id"]
    snap = client.post(f"/api/projects/{pid}/snapshots", json={"label": "before"}).json()
    current = client.patch(f"/api/projects/{pid}/shots/{shot_id}", json={"visual_prompt": "changed"}).json()
    client.put(f"/api/projects/{pid}/script", json={"script": SCRIPT + "\nEXT. PARK - DAY\n\nBirds."})
    current = client.get(f"/api/projects/{pid}").json()
    r = client.post(f"/api/projects/{pid}/snapshots/{snap['id']}/restore", json={"base_revision": current["revision"]})
    assert r.status_code == 200
    restored = r.json()
    assert next(s for s in restored["shots"] if s["id"] == shot_id)["visual_prompt"] == project["shots"][0]["visual_prompt"]
    assert len(restored["script_versions"]) == 2          # script history is never forgotten
    assert restored["current_script_version"] == 1 and restored["script"] == SCRIPT.strip()
    assert restored["revision"] > current["revision"]
    reasons = [s["reason"] for s in client.get(f"/api/projects/{pid}/snapshots").json()]
    assert "pre-restore" in reasons
    assert client.post(f"/api/projects/{pid}/snapshots/bogus/restore", json={}).status_code == 404


def test_manifest_lists_files_with_hashes(client, isolated_workspace):
    pid, _ = new_project(client)
    manifest = client.get(f"/api/projects/{pid}/manifest").json()
    files = {f["path"]: f for f in manifest["files"]}
    original = files["screenplay/original.txt"]
    assert original["kind"] == "script_original" and original["read_only"] is True
    assert original["sha256"] == hashlib.sha256(SCRIPT.strip().encode()).hexdigest()
    assert "project.json" in files and any(f["kind"] == "snapshot" for f in manifest["files"])
    assert (isolated_workspace / pid / "manifests" / "assets.json").exists()


def test_csv_export_still_works(client):
    pid, _ = new_project(client)
    r = client.get(f"/api/projects/{pid}/shot-list.csv")
    assert r.status_code == 200 and r.text.startswith("scene_number,scene_heading,shot_number")


def test_ui_is_served_with_editing_tabs(client):
    html = client.get("/").text
    for marker in ('data-tab="scenes"', 'data-tab="characters"', 'data-tab="shots"',
                   'data-tab="warnings"', 'data-tab="history"', "base_revision"):
        assert marker in html
