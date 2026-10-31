"""Phase 3 API and integration tests: subjects, reference library, approval, scenes, ComfyUI bridge."""
import base64
import hashlib
import json

import pytest
from fastapi.testclient import TestClient

from app import jobs, storage
from app.main import app
from tests.test_assets import jpeg_bytes, png_bytes
from tests.test_generation import MAPPING, FakeComfy, WORKFLOW

SCRIPT = "INT. CAFE - DAY\n\nMAYA sits by the window.\n\nMAYA\nYou came.\n\nEXT. STREET - NIGHT\n\nRain.\n"


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture(autouse=True)
def quiet_jobs(monkeypatch):
    monkeypatch.setenv("FILM_HOUSE_JOB_POLL_SECONDS", "0")
    monkeypatch.setattr(jobs.MANAGER, "inline", True)
    yield
    jobs.MANAGER._active.clear()


def make_project(client) -> str:
    pid = client.post("/api/projects", json={"title": "Studio"}).json()["id"]
    client.put(f"/api/projects/{pid}/script", json={"script": SCRIPT})
    client.post(f"/api/projects/{pid}/breakdown")
    return pid


def b64(data: bytes) -> str:
    return base64.b64encode(data).decode()


def import_asset(client, pid, data=None, filename="face.png", association=None, expect=201, **extra):
    body = {"filename": filename, "data_base64": b64(data if data is not None else png_bytes(20, 20)),
            "association": association or {}, **extra}
    response = client.post(f"/api/projects/{pid}/assets", json=body)
    assert response.status_code == expect, response.text
    return response.json()


def approve(client, pid, asset_id, expect=200):
    response = client.post(f"/api/projects/{pid}/assets/{asset_id}/state", json={"state": "approved"})
    assert response.status_code == expect, response.text
    return response.json()


def new_character(client, pid, name="Maya Rao", **extra):
    response = client.post(f"/api/projects/{pid}/characters", json={"name": name, **extra})
    assert response.status_code == 201, response.text
    return response.json()


def new_location(client, pid, name="Rooftop Cafe", **extra):
    response = client.post(f"/api/projects/{pid}/locations", json={"name": name, **extra})
    assert response.status_code == 201, response.text
    return response.json()


def new_prop(client, pid, name="Brass Key", **extra):
    response = client.post(f"/api/projects/{pid}/props", json={"name": name, **extra})
    assert response.status_code == 201, response.text
    return response.json()


# --------------------------------------------------------------------------- characters

def test_character_crud_with_identity_fields(client, isolated_workspace):
    pid = make_project(client)
    created = new_character(client, pid, name="Maya Rao", age_range="late 20s",
                            physical_description="Tall, left-handed", facial_traits="Scar above right brow",
                            hairstyle="Shoulder-length black", body_type="Lean", personality="Guarded")
    assert created["id"] and created["origin"] == "manual" and created["cue_name"] == "MAYA RAO"
    assert created["age_range"] == "late 20s" and created["facial_traits"] == "Scar above right brow"

    # Stable id across reload, and the parser's own characters are untouched.
    project = client.get(f"/api/projects/{pid}").json()
    stored = next(c for c in project["characters"] if c["id"] == created["id"])
    assert stored["physical_description"] == "Tall, left-handed"
    assert "Maya" in [c["name"] for c in project["characters"]]       # parsed cue still there

    edited = client.patch(f"/api/projects/{pid}/characters/{created['id']}",
                          json={"hairstyle": "Cropped", "personality": "Warmer after scene 4"})
    assert edited.status_code == 200
    again = client.get(f"/api/projects/{pid}").json()
    stored = next(c for c in again["characters"] if c["id"] == created["id"])
    assert stored["hairstyle"] == "Cropped" and stored["facial_traits"] == "Scar above right brow"

    assert client.post(f"/api/projects/{pid}/characters", json={"name": "maya rao"}).status_code == 400
    assert client.post(f"/api/projects/{pid}/characters", json={"name": ""}).status_code == 422

    deleted = client.delete(f"/api/projects/{pid}/characters/{created['id']}")
    assert deleted.status_code == 200 and deleted.json()["deleted"] == created["id"]
    assert created["id"] not in [c["id"] for c in client.get(f"/api/projects/{pid}").json()["characters"]]
    assert client.delete(f"/api/projects/{pid}/characters/{created['id']}").status_code == 404


def test_wardrobe_variants_do_not_touch_identity(client):
    pid = make_project(client)
    character = new_character(client, pid, physical_description="Tall, left-handed")
    variant = client.post(f"/api/projects/{pid}/characters/{character['id']}/wardrobe",
                          json={"name": "Rain coat", "description": "Olive parka, soaked",
                                "scene_numbers": [2, 1, 2]})
    assert variant.status_code == 201
    wardrobe_id = variant.json()["id"]
    assert variant.json()["scene_numbers"] == [1, 2]

    project = client.get(f"/api/projects/{pid}").json()
    stored = next(c for c in project["characters"] if c["id"] == character["id"])
    assert stored["physical_description"] == "Tall, left-handed"        # identity unchanged
    assert [w["name"] for w in stored["wardrobe"]] == ["Rain coat"]

    edited = client.patch(f"/api/projects/{pid}/characters/{character['id']}/wardrobe/{wardrobe_id}",
                          json={"description": "Olive parka, dry"})
    assert edited.status_code == 200 and edited.json()["description"] == "Olive parka, dry"
    assert client.post(f"/api/projects/{pid}/characters/{character['id']}/wardrobe",
                       json={"name": "rain coat"}).status_code == 400
    assert client.delete(f"/api/projects/{pid}/characters/{character['id']}/wardrobe/{wardrobe_id}").status_code == 200
    assert client.delete(f"/api/projects/{pid}/characters/{character['id']}/wardrobe/nope").status_code == 404


# --------------------------------------------------------------------------- locations and props

def test_location_and_prop_crud_with_scene_links(client):
    pid = make_project(client)
    scenes = client.get(f"/api/projects/{pid}").json()["scenes"]
    location = new_location(client, pid, description="Glass walls, neon spill",
                            continuity_requirements="Rain always visible on glass",
                            interior_exterior="INT", time_of_day_variants=["DAY", "NIGHT", "DAY"],
                            scene_ids=[scenes[0]["id"]])
    assert location["time_of_day_variants"] == ["DAY", "NIGHT"]
    assert location["scene_ids"] == [scenes[0]["id"]]
    assert location["continuity_requirements"] == "Rain always visible on glass"

    character = new_character(client, pid)
    prop = new_prop(client, pid, prop_type="hand prop", owner_character_id=character["id"],
                    continuity_requirements="Bent tooth on the left")
    assert prop["owner_character_id"] == character["id"]

    assert client.patch(f"/api/projects/{pid}/locations/{location['id']}",
                        json={"scene_ids": ["nope"]}).status_code == 400
    assert client.patch(f"/api/projects/{pid}/props/{prop['id']}",
                        json={"owner_character_id": "missing"}).status_code == 404
    assert client.post(f"/api/projects/{pid}/locations", json={"name": "ROOFTOP CAFE"}).status_code == 400

    edited = client.patch(f"/api/projects/{pid}/locations/{location['id']}",
                          json={"geography_notes": "Entrance is camera-left"})
    assert edited.status_code == 200 and edited.json()["geography_notes"] == "Entrance is camera-left"

    # Deleting the owning character clears the prop's owner but keeps the prop.
    client.delete(f"/api/projects/{pid}/characters/{character['id']}")
    project = client.get(f"/api/projects/{pid}").json()
    assert next(p for p in project["props"] if p["id"] == prop["id"])["owner_character_id"] is None
    assert client.delete(f"/api/projects/{pid}/props/{prop['id']}").status_code == 200
    assert client.delete(f"/api/projects/{pid}/locations/{location['id']}").status_code == 200
    # one parsed character (MAYA) remains; the one created above was deleted
    assert client.get(f"/api/projects/{pid}/studio").json()["counts"] == {
        "characters": 1, "locations": 0, "props": 0, "assets": 0, "approved": 0, "draft": 0, "rejected": 0}


# --------------------------------------------------------------------------- reference library

def test_import_copies_into_project_storage_with_hash(client, isolated_workspace):
    pid = make_project(client)
    character = new_character(client, pid)
    data = png_bytes(32, 24)
    asset = import_asset(client, pid, data, "maya_face.png",
                         {"subject_type": "character", "subject_id": character["id"], "role": "face"},
                         license="CC0 photo study", notes="front lighting")
    assert asset["id"] and asset["path"] == f"assets/{asset['id']}.png"
    assert asset["sha256"] == hashlib.sha256(data).hexdigest()
    assert asset["bytes"] == len(data) and (asset["width"], asset["height"]) == (32, 24)
    assert asset["state"] == "draft" and asset["version"] == 1 and asset["role"] == "face"
    assert asset["source"] == "import" and asset["license"] == "CC0 photo study"
    assert asset["file_status"] == "ok" and asset["subject_label"] == character["name"]

    stored = isolated_workspace / pid / asset["path"]
    assert stored.read_bytes() == data
    assert not (stored.stat().st_mode & 0o200), "stored asset must be read-only"

    served = client.get(f"/api/projects/{pid}/assets/{asset['id']}/file")
    assert served.status_code == 200 and served.content == data
    assert served.headers["content-type"] == "image/png"

    manifest = client.get(f"/api/projects/{pid}/manifest").json()
    entry = next(f for f in manifest["files"] if f["path"] == asset["path"])
    assert entry["kind"] == "reference_asset" and entry["sha256"] == asset["sha256"]


def test_import_from_an_absolute_source_path(client, tmp_path, isolated_workspace):
    pid = make_project(client)
    source = tmp_path / "location_wide.png"
    source.write_bytes(png_bytes(40, 30))
    response = client.post(f"/api/projects/{pid}/assets", json={"source_path": str(source)})
    assert response.status_code == 201, response.text
    asset = response.json()
    assert asset["original_filename"] == "location_wide.png"
    assert asset["source_detail"] == str(source)
    assert source.exists(), "the user's original file must not be moved or deleted"
    assert (isolated_workspace / pid / asset["path"]).read_bytes() == png_bytes(40, 30)


@pytest.mark.parametrize("payload,code", [
    ({"data_base64": None, "source_path": None}, "ONE_SOURCE_REQUIRED"),
    ({"data_base64": "x", "source_path": "C:/a.png"}, "ONE_SOURCE_REQUIRED"),
])
def test_exactly_one_import_source_is_required(client, payload, code):
    pid = make_project(client)
    response = client.post(f"/api/projects/{pid}/assets", json={"filename": "a.png", **payload})
    assert response.status_code == 422 and response.json()["detail"]["code"] == code


@pytest.mark.parametrize("path", ["../../../../etc/passwd", "assets/x.png", "C:/Windows/System32/notepad.exe"])
def test_unsafe_or_non_image_source_paths_are_refused(client, path):
    pid = make_project(client)
    response = client.post(f"/api/projects/{pid}/assets", json={"source_path": path})
    assert response.status_code == 422
    assert response.json()["detail"]["code"] in ("BAD_SOURCE_PATH", "SOURCE_NOT_FOUND", "UNSUPPORTED_TYPE",
                                                  "FILE_TOO_LARGE", "CORRUPT_IMAGE")
    assert client.get(f"/api/projects/{pid}/assets").json()["assets"] == []


def test_corrupt_and_unsupported_uploads_are_refused_and_nothing_is_stored(client, isolated_workspace):
    pid = make_project(client)
    for name, data in [("truncated.png", png_bytes(64, 64, noisy=True)[:-8]),
                       ("document.pdf", b"%PDF-1.7" + b"x" * 300),
                       ("empty.png", b"")]:
        response = client.post(f"/api/projects/{pid}/assets",
                               json={"filename": name, "data_base64": b64(data)})
        assert response.status_code == 422, name
        assert response.json()["detail"]["code"] in ("CORRUPT_IMAGE", "UNSUPPORTED_TYPE", "EMPTY_FILE")
    bad_base64 = client.post(f"/api/projects/{pid}/assets",
                             json={"filename": "a.png", "data_base64": "@@@not base64@@@"})
    assert bad_base64.status_code == 422 and bad_base64.json()["detail"]["code"] == "BAD_ENCODING"
    assert client.get(f"/api/projects/{pid}/assets").json()["assets"] == []
    asset_dir = isolated_workspace / pid / "assets"
    assert not asset_dir.exists() or list(asset_dir.glob("*")) == []


def test_duplicate_import_is_refused_unless_allowed(client):
    pid = make_project(client)
    data = png_bytes(16, 16)
    first = import_asset(client, pid, data, "a.png")
    again = client.post(f"/api/projects/{pid}/assets", json={"filename": "copy.png", "data_base64": b64(data)})
    assert again.status_code == 409
    assert again.json()["detail"]["code"] == "DUPLICATE_ASSET"
    assert again.json()["detail"]["asset_id"] == first["id"]
    allowed = import_asset(client, pid, data, "copy.png", allow_duplicate=True)
    assert allowed["id"] != first["id"] and allowed["duplicate_of"] == first["id"]
    assert allowed["sha256"] == first["sha256"]
    assert len(client.get(f"/api/projects/{pid}/assets").json()["assets"]) == 2


def test_extension_mismatch_is_reported_and_stored_by_real_format(client):
    pid = make_project(client)
    asset = import_asset(client, pid, jpeg_bytes(20, 10), "mislabelled.png")
    assert asset["extension_mismatch"] is True
    assert asset["image_format"] == "JPEG" and asset["path"].endswith(".jpg")


def test_association_validation(client):
    pid = make_project(client)
    character = new_character(client, pid)
    for association, status in [
        ({"subject_type": "character", "subject_id": "missing"}, 404),
        ({"subject_type": "character"}, 400),
        ({"subject_id": character["id"]}, 400),
        ({"subject_type": "character", "subject_id": character["id"], "wardrobe_id": "nope"}, 404),
        ({"wardrobe_id": "nope"}, 400),
    ]:
        response = client.post(f"/api/projects/{pid}/assets",
                               json={"filename": "a.png", "data_base64": b64(png_bytes(8 + status % 7, 8)),
                                     "association": association})
        assert response.status_code == status, (association, response.text)


def test_versions_count_up_per_subject_and_role(client):
    pid = make_project(client)
    character = new_character(client, pid)
    face = {"subject_type": "character", "subject_id": character["id"], "role": "face"}
    body = {"subject_type": "character", "subject_id": character["id"], "role": "full_body"}
    first = import_asset(client, pid, png_bytes(10, 10), "f1.png", face)
    second = import_asset(client, pid, png_bytes(11, 10), "f2.png", face)
    third = import_asset(client, pid, png_bytes(12, 10), "b1.png", body)
    assert (first["version"], second["version"], third["version"]) == (1, 2, 1)
    # Deleting the newest does not let the next import reuse its number.
    client.post(f"/api/projects/{pid}/assets/{second['id']}/state", json={"state": "rejected"})
    client.request("DELETE", f"/api/projects/{pid}/assets/{second['id']}", json={"confirm": True})
    fourth = import_asset(client, pid, png_bytes(13, 10), "f3.png", face)
    assert fourth["version"] == 2


def test_asset_edit_moves_association_and_licence(client):
    pid = make_project(client)
    character = new_character(client, pid)
    location = new_location(client, pid)
    asset = import_asset(client, pid, png_bytes(10, 12), "x.png")
    assert asset["subject_id"] is None
    moved = client.patch(f"/api/projects/{pid}/assets/{asset['id']}",
                         json={"subject_type": "location", "subject_id": location["id"],
                               "role": "location_wide", "license": "my own photo"})
    assert moved.status_code == 200
    assert moved.json()["subject_id"] == location["id"] and moved.json()["role"] == "location_wide"
    assert moved.json()["license"] == "my own photo"
    assert moved.json()["subject_label"] == location["name"]
    cleared = client.patch(f"/api/projects/{pid}/assets/{asset['id']}", json={"clear_association": True})
    assert cleared.status_code == 200 and cleared.json()["subject_id"] is None


# --------------------------------------------------------------------------- approval lifecycle

def test_approval_transitions_are_recorded(client):
    pid = make_project(client)
    character = new_character(client, pid)
    asset = import_asset(client, pid, png_bytes(20, 20), "a.png",
                         {"subject_type": "character", "subject_id": character["id"], "role": "face"})
    assert asset["state"] == "draft" and asset["approved_at"] is None

    approved = approve(client, pid, asset["id"])
    assert approved["state"] == "approved" and approved["approved_at"]
    assert approved["state_history"][-1]["from_state"] == "draft"

    rejected = client.post(f"/api/projects/{pid}/assets/{asset['id']}/state",
                           json={"state": "rejected", "note": "wrong hair"}).json()
    assert rejected["state"] == "rejected" and rejected["rejected_at"]
    assert rejected["state_history"][-1]["note"] == "wrong hair"

    back = approve(client, pid, asset["id"])
    assert back["state"] == "approved" and len(back["state_history"]) == 3
    unchanged = approve(client, pid, asset["id"])
    assert unchanged["changed"] is False and len(unchanged["state_history"]) == 3
    assert client.post(f"/api/projects/{pid}/assets/{asset['id']}/state",
                       json={"state": "nonsense"}).status_code == 422


def test_canonical_requires_approval_and_explicit_replacement(client):
    pid = make_project(client)
    character = new_character(client, pid)
    association = {"subject_type": "character", "subject_id": character["id"], "role": "face"}
    first = import_asset(client, pid, png_bytes(20, 20), "v1.png", association)
    second = import_asset(client, pid, png_bytes(21, 20), "v2.png", association)

    draft_attempt = client.post(f"/api/projects/{pid}/characters/{character['id']}/canonical",
                                json={"asset_id": first["id"]})
    assert draft_attempt.status_code == 409
    assert draft_attempt.json()["detail"]["code"] == "ASSET_NOT_APPROVED"

    approve(client, pid, first["id"])
    approve(client, pid, second["id"])
    chosen = client.post(f"/api/projects/{pid}/characters/{character['id']}/canonical",
                         json={"asset_id": first["id"]})
    assert chosen.status_code == 200 and chosen.json()["canonical_reference_id"] == first["id"]

    blocked = client.post(f"/api/projects/{pid}/characters/{character['id']}/canonical",
                          json={"asset_id": second["id"]})
    assert blocked.status_code == 409
    assert blocked.json()["detail"]["code"] == "CANONICAL_EXISTS"
    assert blocked.json()["detail"]["detail"]["current_asset_id"] == first["id"]

    replaced = client.post(f"/api/projects/{pid}/characters/{character['id']}/canonical",
                           json={"asset_id": second["id"], "confirm_replace": True, "note": "better light"})
    assert replaced.status_code == 200
    assert replaced.json()["canonical_reference_id"] == second["id"]
    assert replaced.json()["replaced_asset_id"] == first["id"]
    assert replaced.json()["canonical_history"][-1]["note"] == "better light"

    # the replaced version is preserved, still approved
    kept = client.get(f"/api/projects/{pid}/assets/{first['id']}").json()
    assert kept["state"] == "approved" and kept["file_status"] == "ok"

    cleared = client.request("DELETE", f"/api/projects/{pid}/characters/{character['id']}/canonical", json={})
    assert cleared.status_code == 200 and cleared.json()["cleared_asset_id"] == second["id"]


def test_canonical_rejects_assets_from_another_subject_or_wardrobe(client):
    pid = make_project(client)
    first_character = new_character(client, pid, name="Maya Rao")
    other = new_character(client, pid, name="Arjun Patel")
    wardrobe = client.post(f"/api/projects/{pid}/characters/{first_character['id']}/wardrobe",
                           json={"name": "Parka"}).json()
    asset = import_asset(client, pid, png_bytes(20, 20), "a.png",
                         {"subject_type": "character", "subject_id": first_character["id"], "role": "face"})
    approve(client, pid, asset["id"])
    wrong_subject = client.post(f"/api/projects/{pid}/characters/{other['id']}/canonical",
                                json={"asset_id": asset["id"]})
    assert wrong_subject.status_code == 409 and wrong_subject.json()["detail"]["code"] == "ASSET_WRONG_SUBJECT"
    wrong_wardrobe = client.post(
        f"/api/projects/{pid}/characters/{first_character['id']}/wardrobe/{wardrobe['id']}/canonical",
        json={"asset_id": asset["id"]})
    assert wrong_wardrobe.status_code == 409 and wrong_wardrobe.json()["detail"]["code"] == "ASSET_WRONG_WARDROBE"

    costume = import_asset(client, pid, png_bytes(22, 20), "parka.png",
                           {"subject_type": "character", "subject_id": first_character["id"],
                            "wardrobe_id": wardrobe["id"], "role": "wardrobe"})
    approve(client, pid, costume["id"])
    base = client.post(f"/api/projects/{pid}/characters/{first_character['id']}/canonical",
                       json={"asset_id": asset["id"]})
    assert base.status_code == 200
    ok = client.post(f"/api/projects/{pid}/characters/{first_character['id']}/wardrobe/{wardrobe['id']}/canonical",
                     json={"asset_id": costume["id"]})
    assert ok.status_code == 200 and ok.json()["canonical_reference_id"] == costume["id"]
    # identity canonical and wardrobe canonical are tracked separately
    studio = client.get(f"/api/projects/{pid}/studio").json()
    character_record = next(c for c in studio["characters"] if c["id"] == first_character["id"])
    assert character_record["canonical_reference_id"] == asset["id"]
    assert character_record["wardrobe"][0]["canonical_reference_id"] == costume["id"]


def test_canonical_asset_cannot_be_rejected_and_demotion_is_explicit(client):
    pid = make_project(client)
    location = new_location(client, pid)
    asset = import_asset(client, pid, png_bytes(30, 20), "wide.png",
                         {"subject_type": "location", "subject_id": location["id"], "role": "location_wide"})
    approve(client, pid, asset["id"])
    client.post(f"/api/projects/{pid}/locations/{location['id']}/canonical", json={"asset_id": asset["id"]})

    refused = client.post(f"/api/projects/{pid}/assets/{asset['id']}/state", json={"state": "rejected"})
    assert refused.status_code == 409
    assert refused.json()["detail"]["code"] == "REFERENCED_AS_CANONICAL"
    assert "canonical reference" in refused.json()["detail"]["message"]

    needs_confirm = client.post(f"/api/projects/{pid}/assets/{asset['id']}/state", json={"state": "draft"})
    assert needs_confirm.status_code == 409
    assert needs_confirm.json()["detail"]["code"] == "REFERENCED_AS_CANONICAL"

    demoted = client.post(f"/api/projects/{pid}/assets/{asset['id']}/state",
                          json={"state": "draft", "confirm_demote_canonical": True})
    assert demoted.status_code == 200 and demoted.json()["state"] == "draft"
    assert demoted.json()["unlinked"][0]["kind"] == "canonical"
    studio = client.get(f"/api/projects/{pid}/studio").json()
    assert next(l for l in studio["locations"] if l["id"] == location["id"])["canonical_reference_id"] is None


# --------------------------------------------------------------------------- safe deletion

def test_deletion_requires_confirmation_and_checks_references(client, isolated_workspace):
    pid = make_project(client)
    character = new_character(client, pid)
    asset = import_asset(client, pid, png_bytes(20, 20), "a.png",
                         {"subject_type": "character", "subject_id": character["id"], "role": "face"})
    approve(client, pid, asset["id"])
    client.post(f"/api/projects/{pid}/characters/{character['id']}/canonical", json={"asset_id": asset["id"]})

    in_use = client.request("DELETE", f"/api/projects/{pid}/assets/{asset['id']}", json={"confirm": True})
    assert in_use.status_code == 409 and in_use.json()["detail"]["code"] == "ASSET_IN_USE"
    assert (isolated_workspace / pid / asset["path"]).exists()

    references = client.get(f"/api/projects/{pid}/assets/{asset['id']}/references").json()
    assert references["deletable"] is False
    assert any(r["kind"] == "canonical" and r["blocking"] for r in references["references"])

    client.request("DELETE", f"/api/projects/{pid}/characters/{character['id']}/canonical", json={})
    unconfirmed = client.request("DELETE", f"/api/projects/{pid}/assets/{asset['id']}", json={})
    assert unconfirmed.status_code == 400
    assert unconfirmed.json()["detail"]["code"] == "CONFIRMATION_REQUIRED"
    assert (isolated_workspace / pid / asset["path"]).exists()

    deleted = client.request("DELETE", f"/api/projects/{pid}/assets/{asset['id']}", json={"confirm": True})
    assert deleted.status_code == 200 and deleted.json()["file_removed"] is True
    assert not (isolated_workspace / pid / asset["path"]).exists()
    assert client.get(f"/api/projects/{pid}/assets/{asset['id']}").status_code == 404
    reasons = [s["reason"] for s in client.get(f"/api/projects/{pid}/snapshots").json()]
    assert reasons.count("pre-asset-delete") == 1, "a refused delete must not snapshot"


def test_deleting_a_subject_keeps_its_images(client, isolated_workspace):
    pid = make_project(client)
    character = new_character(client, pid)
    asset = import_asset(client, pid, png_bytes(20, 20), "a.png",
                         {"subject_type": "character", "subject_id": character["id"], "role": "face"})
    approve(client, pid, asset["id"])
    client.post(f"/api/projects/{pid}/characters/{character['id']}/canonical", json={"asset_id": asset["id"]})
    result = client.delete(f"/api/projects/{pid}/characters/{character['id']}").json()
    assert result["assets_unassigned"] == [asset["id"]]
    kept = client.get(f"/api/projects/{pid}/assets/{asset['id']}").json()
    assert kept["subject_id"] is None and kept["state"] == "approved"
    assert (isolated_workspace / pid / asset["path"]).exists()


def test_missing_file_is_reported_not_crashed(client, isolated_workspace):
    pid = make_project(client)
    asset = import_asset(client, pid, png_bytes(20, 20), "a.png")
    path = isolated_workspace / pid / asset["path"]
    path.chmod(0o666)
    path.unlink()
    assert client.get(f"/api/projects/{pid}/assets/{asset['id']}").json()["file_status"] == "missing"
    assert client.get(f"/api/projects/{pid}/assets/{asset['id']}/file").status_code == 404
    warnings = client.get(f"/api/projects/{pid}/warnings").json()["warnings"]
    assert any(w["code"] == "ASSET_FILE_MISSING" and w["severity"] == "error" for w in warnings)
    # the record can still be deleted to clean up
    gone = client.request("DELETE", f"/api/projects/{pid}/assets/{asset['id']}", json={"confirm": True})
    assert gone.status_code == 200 and gone.json()["file_removed"] is False


def test_modified_file_is_detected(client, isolated_workspace):
    pid = make_project(client)
    asset = import_asset(client, pid, png_bytes(20, 20), "a.png")
    path = isolated_workspace / pid / asset["path"]
    path.chmod(0o666)
    path.write_bytes(png_bytes(21, 21))
    assert client.get(f"/api/projects/{pid}/assets/{asset['id']}").json()["file_status"] == "modified"
    warnings = client.get(f"/api/projects/{pid}/warnings").json()["warnings"]
    assert any(w["code"] == "ASSET_FILE_MODIFIED" for w in warnings)


def test_failed_file_write_leaves_no_record(client, monkeypatch, isolated_workspace):
    pid = make_project(client)
    monkeypatch.setattr("app.main.write_asset_file",
                        lambda *a, **k: (_ for _ in ()).throw(storage.StorageError("disk full")))
    response = client.post(f"/api/projects/{pid}/assets",
                           json={"filename": "a.png", "data_base64": b64(png_bytes(20, 20))})
    assert response.status_code == 500 and response.json()["code"] == "STORAGE_ERROR"
    assert client.get(f"/api/projects/{pid}/assets").json()["assets"] == []


def test_failed_record_save_removes_the_orphan_file(client, monkeypatch, isolated_workspace):
    pid = make_project(client)
    real_save = storage.save_project

    def fail_once(project):
        raise OSError("project.json is locked")

    monkeypatch.setattr("app.main.save_project", fail_once)
    with pytest.raises(OSError):
        client.post(f"/api/projects/{pid}/assets",
                    json={"filename": "a.png", "data_base64": b64(png_bytes(20, 20))})
    monkeypatch.setattr("app.main.save_project", real_save)
    assert list((isolated_workspace / pid / "assets").glob("*")) == []
    assert client.get(f"/api/projects/{pid}/assets").json()["assets"] == []


# --------------------------------------------------------------------------- scene integration

def test_scene_associations_and_approved_selections(client):
    pid = make_project(client)
    project = client.get(f"/api/projects/{pid}").json()
    scene = project["scenes"][0]
    original_body = scene["body"]
    character = new_character(client, pid)
    location = new_location(client, pid)
    prop = new_prop(client, pid)
    wardrobe = client.post(f"/api/projects/{pid}/characters/{character['id']}/wardrobe",
                           json={"name": "Parka"}).json()

    look = import_asset(client, pid, png_bytes(25, 25), "look.png",
                        {"subject_type": "character", "subject_id": character["id"],
                         "wardrobe_id": wardrobe["id"], "role": "wardrobe"})
    wide = import_asset(client, pid, png_bytes(26, 25), "wide.png",
                        {"subject_type": "location", "subject_id": location["id"], "role": "location_wide"})

    draft_selection = client.put(f"/api/projects/{pid}/scenes/{scene['id']}/associations", json={
        "character_ids": [character["id"]], "location_id": location["id"],
        "location_reference_id": wide["id"]})
    assert draft_selection.status_code == 409
    assert draft_selection.json()["detail"]["code"] == "ASSET_NOT_APPROVED"

    approve(client, pid, look["id"])
    approve(client, pid, wide["id"])
    response = client.put(f"/api/projects/{pid}/scenes/{scene['id']}/associations", json={
        "character_ids": [character["id"]], "location_id": location["id"], "prop_ids": [prop["id"]],
        "location_reference_id": wide["id"],
        "appearance_selections": [{"character_id": character["id"], "wardrobe_id": wardrobe["id"],
                                   "asset_id": look["id"]}]})
    assert response.status_code == 200, response.text
    saved = response.json()["scene"]
    assert saved["character_ids"] == [character["id"]] and saved["location_id"] == location["id"]
    assert saved["prop_ids"] == [prop["id"]] and saved["location_reference_id"] == wide["id"]
    assert saved["appearance_selections"][0]["asset_id"] == look["id"]
    assert saved["body"] == original_body, "the screenplay text must not change"
    assert saved["heading"] == scene["heading"] and saved["elements"] == scene["elements"]
    assert client.get(f"/api/projects/{pid}/script/original").text == SCRIPT.strip()

    wrong = client.put(f"/api/projects/{pid}/scenes/{scene['id']}/associations", json={
        "appearance_selections": [{"character_id": character["id"], "asset_id": look["id"]}]})
    assert wrong.status_code == 409 and wrong.json()["detail"]["code"] == "ASSET_WRONG_WARDROBE"
    assert client.put(f"/api/projects/{pid}/scenes/{scene['id']}/associations",
                      json={"character_ids": ["nope"]}).status_code == 404
    assert client.put(f"/api/projects/{pid}/scenes/nope/associations", json={}).status_code == 404


def test_scene_selection_is_cleared_when_an_asset_stops_being_approved(client):
    pid = make_project(client)
    scene = client.get(f"/api/projects/{pid}").json()["scenes"][0]
    location = new_location(client, pid)
    wide = import_asset(client, pid, png_bytes(30, 30), "wide.png",
                        {"subject_type": "location", "subject_id": location["id"], "role": "location_wide"})
    approve(client, pid, wide["id"])
    client.put(f"/api/projects/{pid}/scenes/{scene['id']}/associations",
               json={"location_id": location["id"], "location_reference_id": wide["id"]})
    result = client.post(f"/api/projects/{pid}/assets/{wide['id']}/state",
                         json={"state": "rejected", "note": "wrong season"}).json()
    assert any(u["kind"] == "scene_location" for u in result["unlinked"])
    refreshed = client.get(f"/api/projects/{pid}").json()["scenes"][0]
    assert refreshed["location_reference_id"] is None


def test_scene_reference_needs_a_location_first(client):
    pid = make_project(client)
    scene = client.get(f"/api/projects/{pid}").json()["scenes"][0]
    location = new_location(client, pid)
    wide = import_asset(client, pid, png_bytes(30, 30), "w.png",
                        {"subject_type": "location", "subject_id": location["id"], "role": "location_wide"})
    approve(client, pid, wide["id"])
    response = client.put(f"/api/projects/{pid}/scenes/{scene['id']}/associations",
                          json={"location_reference_id": wide["id"]})
    assert response.status_code == 400 and response.json()["detail"]["code"] == "SCENE_HAS_NO_LOCATION"


def test_rebuilding_the_breakdown_keeps_studio_records(client):
    pid = make_project(client)
    character = new_character(client, pid)
    location = new_location(client, pid)
    asset = import_asset(client, pid, png_bytes(20, 20), "a.png",
                         {"subject_type": "location", "subject_id": location["id"], "role": "location_wide"})
    approve(client, pid, asset["id"])
    client.post(f"/api/projects/{pid}/locations/{location['id']}/canonical", json={"asset_id": asset["id"]})
    rebuilt = client.post(f"/api/projects/{pid}/breakdown", json={"force": True})
    assert rebuilt.status_code == 200
    studio = client.get(f"/api/projects/{pid}/studio").json()
    assert location["id"] in [l["id"] for l in studio["locations"]]
    assert character["id"] in [c["id"] for c in studio["characters"]]
    assert next(l for l in studio["locations"] if l["id"] == location["id"])["canonical_reference_id"] == asset["id"]
    assert len(studio["assets"]) == 1


# --------------------------------------------------------------------------- revision safety

def test_studio_edits_respect_the_revision_counter(client):
    pid = make_project(client)
    project = client.get(f"/api/projects/{pid}").json()
    stale = project["revision"]
    new_character(client, pid, name="A")
    conflict = client.post(f"/api/projects/{pid}/characters",
                           json={"name": "B", "base_revision": stale})
    assert conflict.status_code == 409 and conflict.json()["code"] == "REVISION_CONFLICT"
    assert [c["name"] for c in client.get(f"/api/projects/{pid}").json()["characters"]].count("B") == 0


def test_studio_data_survives_snapshot_restore(client):
    pid = make_project(client)
    character = new_character(client, pid)
    asset = import_asset(client, pid, png_bytes(20, 20), "a.png",
                         {"subject_type": "character", "subject_id": character["id"], "role": "face"})
    approve(client, pid, asset["id"])
    client.post(f"/api/projects/{pid}/characters/{character['id']}/canonical", json={"asset_id": asset["id"]})
    snapshot = client.post(f"/api/projects/{pid}/snapshots", json={"label": "with studio"}).json()
    client.delete(f"/api/projects/{pid}/characters/{character['id']}")
    project = client.get(f"/api/projects/{pid}").json()
    restored = client.post(f"/api/projects/{pid}/snapshots/{snapshot['id']}/restore",
                           json={"base_revision": project["revision"]}).json()
    assert character["id"] in [c["id"] for c in restored["characters"]]
    assert next(c for c in restored["characters"]
                if c["id"] == character["id"])["canonical_reference_id"] == asset["id"]
    assert [a["id"] for a in restored["assets"]] == [asset["id"]]


# --------------------------------------------------------------------------- Phase 4 integration

def ready_workflow(client, pid, with_reference=True):
    workflow = dict(WORKFLOW)
    mapping = dict(MAPPING)
    if with_reference:
        workflow["12"] = {"class_type": "LoadImage", "inputs": {"image": "example.png"}}
        mapping["reference_image"] = {"node_id": "12", "input": "image"}
    imported = client.post(f"/api/projects/{pid}/workflows", json={"name": "wf", "workflow": workflow})
    assert imported.status_code == 201, imported.text
    wid = imported.json()["record"]["id"]
    assert client.put(f"/api/projects/{pid}/workflows/{wid}/mapping",
                      json={"mapping": mapping}).status_code == 200
    return wid


def approved_asset(client, pid):
    character = new_character(client, pid)
    asset = import_asset(client, pid, png_bytes(40, 40), "ref.png",
                         {"subject_type": "character", "subject_id": character["id"], "role": "face"})
    approve(client, pid, asset["id"])
    client.post(f"/api/projects/{pid}/characters/{character['id']}/canonical", json={"asset_id": asset["id"]})
    return character, asset


def test_approved_asset_is_used_as_the_workflow_reference(client, monkeypatch):
    pid = make_project(client)
    wid = ready_workflow(client, pid)
    character, asset = approved_asset(client, pid)
    fake = FakeComfy()
    monkeypatch.setattr(jobs.MANAGER, "client_factory", lambda: fake)
    response = client.post(f"/api/projects/{pid}/jobs", json={
        "workflow_id": wid, "prompt": "portrait", "reference_asset_id": asset["id"],
        "target_type": "character", "target_id": character["id"]})
    assert response.status_code == 201, response.text
    job = response.json()
    assert job["status"] == "succeeded", job["error"]
    assert job["request"]["reference_asset_id"] == asset["id"]
    assert fake.uploads and fake.uploads[0].endswith(f"{asset['id']}.png")
    assert fake.submitted[0]["prompt"]["12"]["inputs"]["image"] == f"ai-film-house/{asset['id']}.png"
    assert any("library asset" in line for line in job["log"])
    listing = client.get(f"/api/projects/{pid}/reference-images").json()
    assert asset["id"] in [a["id"] for a in listing["assets"]]


def test_draft_asset_needs_an_explicit_opt_in(client, monkeypatch):
    pid = make_project(client)
    wid = ready_workflow(client, pid)
    character = new_character(client, pid)
    draft = import_asset(client, pid, png_bytes(40, 40), "d.png",
                         {"subject_type": "character", "subject_id": character["id"], "role": "face"})
    monkeypatch.setattr(jobs.MANAGER, "client_factory", lambda: FakeComfy())
    refused = client.post(f"/api/projects/{pid}/jobs", json={
        "workflow_id": wid, "prompt": "p", "reference_asset_id": draft["id"]})
    assert refused.status_code == 409
    assert refused.json()["detail"]["code"] == "ASSET_NOT_APPROVED"
    assert "allow_draft_reference" in refused.json()["detail"]["message"]
    allowed = client.post(f"/api/projects/{pid}/jobs", json={
        "workflow_id": wid, "prompt": "p", "reference_asset_id": draft["id"], "allow_draft_reference": True})
    assert allowed.status_code == 201 and allowed.json()["status"] == "succeeded"


def test_rejected_asset_is_never_used(client, monkeypatch):
    pid = make_project(client)
    wid = ready_workflow(client, pid)
    character = new_character(client, pid)
    asset = import_asset(client, pid, png_bytes(40, 40), "r.png",
                         {"subject_type": "character", "subject_id": character["id"], "role": "face"})
    client.post(f"/api/projects/{pid}/assets/{asset['id']}/state", json={"state": "rejected"})
    monkeypatch.setattr(jobs.MANAGER, "client_factory", lambda: FakeComfy())
    for body in ({"reference_asset_id": asset["id"]},
                 {"reference_asset_id": asset["id"], "allow_draft_reference": True}):
        response = client.post(f"/api/projects/{pid}/jobs", json={"workflow_id": wid, "prompt": "p", **body})
        assert response.status_code == 409
        assert response.json()["detail"]["code"] == "ASSET_REJECTED"


def test_workflow_without_a_reference_node_explains_itself(client, monkeypatch):
    pid = make_project(client)
    wid = ready_workflow(client, pid, with_reference=False)
    _character, asset = approved_asset(client, pid)
    monkeypatch.setattr(jobs.MANAGER, "client_factory", lambda: FakeComfy())
    response = client.post(f"/api/projects/{pid}/jobs", json={
        "workflow_id": wid, "prompt": "p", "reference_asset_id": asset["id"]})
    assert response.status_code == 409
    detail = response.json()["detail"]
    assert detail["code"] == "ROLE_UNMAPPED"
    assert "no reference image node mapped" in detail["message"]
    assert "LoadImage" in detail["message"] and "text-to-image" in detail["message"]
    assert client.get(f"/api/projects/{pid}/jobs").json()["jobs"] == []


def test_two_reference_sources_are_refused(client):
    pid = make_project(client)
    wid = ready_workflow(client, pid)
    _character, asset = approved_asset(client, pid)
    response = client.post(f"/api/projects/{pid}/jobs", json={
        "workflow_id": wid, "prompt": "p", "reference_asset_id": asset["id"],
        "reference_path": "references/input/other.png"})
    assert response.status_code == 422
    assert "not both" in json.dumps(response.json())


def test_missing_asset_file_blocks_generation(client, monkeypatch, isolated_workspace):
    pid = make_project(client)
    wid = ready_workflow(client, pid)
    _character, asset = approved_asset(client, pid)
    path = isolated_workspace / pid / asset["path"]
    path.chmod(0o666)
    path.write_bytes(png_bytes(9, 9))                 # content no longer matches the recorded hash
    monkeypatch.setattr(jobs.MANAGER, "client_factory", lambda: FakeComfy())
    response = client.post(f"/api/projects/{pid}/jobs", json={
        "workflow_id": wid, "prompt": "p", "reference_asset_id": asset["id"]})
    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "ASSET_FILE_MODIFIED"
    assert client.get(f"/api/projects/{pid}/jobs").json()["jobs"] == []


def test_generated_output_can_be_imported_as_a_draft_reference(client, monkeypatch, isolated_workspace):
    pid = make_project(client)
    wid = ready_workflow(client, pid, with_reference=False)
    location = new_location(client, pid)
    monkeypatch.setattr(jobs.MANAGER, "client_factory", lambda: FakeComfy())
    job = client.post(f"/api/projects/{pid}/jobs", json={
        "workflow_id": wid, "prompt": "wide shot of the cafe", "target_type": "location",
        "target_id": location["id"]}).json()
    assert job["status"] == "succeeded", job["error"]

    response = client.post(f"/api/projects/{pid}/assets/from-job", json={
        "job_id": job["id"], "output_index": 0, "notes": "first pass",
        "association": {"subject_type": "location", "subject_id": location["id"], "role": "location_wide"}})
    assert response.status_code == 201, response.text
    asset = response.json()
    assert asset["source"] == "generated" and asset["job_id"] == job["id"]
    assert asset["state"] == "draft" and asset["subject_id"] == location["id"]
    assert asset["source_detail"] == job["outputs"][0]["path"]
    assert asset["sha256"] == job["outputs"][0]["sha256"]
    assert (isolated_workspace / pid / job["outputs"][0]["path"]).exists(), "job output must stay in place"
    assert (isolated_workspace / pid / asset["path"]).exists()

    again = client.post(f"/api/projects/{pid}/assets/from-job",
                        json={"job_id": job["id"], "output_index": 0})
    assert again.status_code == 409 and again.json()["detail"]["code"] == "DUPLICATE_ASSET"
    assert client.post(f"/api/projects/{pid}/assets/from-job",
                       json={"job_id": job["id"], "output_index": 7}).status_code == 404
    assert client.post(f"/api/projects/{pid}/assets/from-job",
                       json={"job_id": "a" * 12, "output_index": 0}).status_code == 404

    approve(client, pid, asset["id"])
    canonical = client.post(f"/api/projects/{pid}/locations/{location['id']}/canonical",
                            json={"asset_id": asset["id"]})
    assert canonical.status_code == 200


def test_job_reference_shows_up_in_asset_references(client, monkeypatch):
    pid = make_project(client)
    wid = ready_workflow(client, pid)
    _character, asset = approved_asset(client, pid)
    monkeypatch.setattr(jobs.MANAGER, "client_factory", lambda: FakeComfy())
    client.post(f"/api/projects/{pid}/jobs", json={
        "workflow_id": wid, "prompt": "p", "reference_asset_id": asset["id"]})
    references = client.get(f"/api/projects/{pid}/assets/{asset['id']}/references").json()
    job_refs = [r for r in references["references"] if r["kind"] == "generation_job"]
    assert job_refs and job_refs[0]["blocking"] is False       # a finished job does not block deletion


def test_editing_the_project_while_a_generation_job_runs(client, monkeypatch):
    """Studio edits and a live job must not block or corrupt each other."""
    pid = make_project(client)
    wid = ready_workflow(client, pid, with_reference=False)
    fake = FakeComfy(states=["running"])
    monkeypatch.setattr(jobs.MANAGER, "client_factory", lambda: fake)
    monkeypatch.setattr(jobs.MANAGER, "inline", False)
    started = client.post(f"/api/projects/{pid}/jobs", json={"workflow_id": wid, "prompt": "slow"}).json()
    for _ in range(300):
        current = client.get(f"/api/projects/{pid}/jobs/{started['id']}").json()
        if current["status"] == "running":
            break
    assert current["status"] == "running"

    character = new_character(client, pid, name="While Generating")
    asset = import_asset(client, pid, png_bytes(33, 33), "live.png",
                         {"subject_type": "character", "subject_id": character["id"], "role": "face"})
    approve(client, pid, asset["id"])
    canonical = client.post(f"/api/projects/{pid}/characters/{character['id']}/canonical",
                            json={"asset_id": asset["id"]})
    assert canonical.status_code == 200
    scene = client.get(f"/api/projects/{pid}").json()["scenes"][0]
    assert client.put(f"/api/projects/{pid}/scenes/{scene['id']}/associations",
                      json={"character_ids": [character["id"]]}).status_code == 200

    client.post(f"/api/projects/{pid}/jobs/{started['id']}/cancel")
    finished = jobs.MANAGER.wait(pid, started["id"], timeout=10.0)
    assert finished.status == "cancelled"
    studio = client.get(f"/api/projects/{pid}/studio").json()
    assert character["id"] in [c["id"] for c in studio["characters"]]
    assert len(studio["assets"]) == 1 and studio["assets"][0]["state"] == "approved"


# --------------------------------------------------------------------------- overview & UI

def test_studio_overview_shape(client):
    pid = make_project(client)
    character = new_character(client, pid)
    location = new_location(client, pid)
    new_prop(client, pid)
    asset = import_asset(client, pid, png_bytes(20, 20), "a.png",
                         {"subject_type": "character", "subject_id": character["id"], "role": "face"})
    approve(client, pid, asset["id"])
    client.post(f"/api/projects/{pid}/characters/{character['id']}/canonical", json={"asset_id": asset["id"]})
    studio = client.get(f"/api/projects/{pid}/studio").json()
    assert studio["counts"]["characters"] == 2 and studio["counts"]["approved"] == 1
    record = next(c for c in studio["characters"] if c["id"] == character["id"])
    assert record["canonical"]["id"] == asset["id"]
    assert record["assets"][0]["is_canonical"] is True
    assert studio["scenes"][0]["number"] == 1 and "cue_names" in studio["scenes"][0]
    assert next(l for l in studio["locations"] if l["id"] == location["id"])["canonical"] is None
    assert client.get("/api/projects/" + "0" * 32 + "/studio").status_code == 404


def test_asset_filters(client):
    pid = make_project(client)
    character = new_character(client, pid)
    first = import_asset(client, pid, png_bytes(10, 10), "a.png",
                         {"subject_type": "character", "subject_id": character["id"], "role": "face"})
    second = import_asset(client, pid, png_bytes(11, 10), "b.png")
    approve(client, pid, first["id"])
    assert [a["id"] for a in client.get(f"/api/projects/{pid}/assets?state=approved").json()["assets"]] == [first["id"]]
    assert [a["id"] for a in client.get(f"/api/projects/{pid}/assets?unassigned=true").json()["assets"]] == [second["id"]]
    assert len(client.get(f"/api/projects/{pid}/assets?subject_type=character").json()["assets"]) == 1
    limits = client.get(f"/api/projects/{pid}/assets").json()["limits"]
    assert limits["max_bytes"] > 0 and "PNG" in limits["formats"]


def test_studio_page_is_served_with_every_section(client):
    page = client.get("/studio")
    assert page.status_code == 200
    for marker in ("Characters", "Locations", "Props", "References", "/api/projects", "canonical", "Approve"):
        assert marker.lower() in page.text.lower(), marker
    assert 'href="/studio"' in client.get("/").text
