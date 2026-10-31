"""Phase 4: ComfyUI client/adapter and API-format workflow handling.

Every HTTP call is mocked at urllib level or with a fake client; no ComfyUI and no network needed.
"""
import io
import json
import urllib.error
import urllib.request

import pytest

from app import comfy
from app.comfy import ComfyAdapter, ComfyClient, ComfyError
from app.workflows import (
    MAX_SEED, ROLE_SPECS, WorkflowError, apply_mapping, inspect_workflow, load_api_workflow, output_node_ids,
    suggest_mapping, validate_against_server, validate_mapping, workflow_models,
)

LOCAL = "http://127.0.0.1:8188"

# A small but realistic txt2img API workflow.
WORKFLOW = {
    "4": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "sd_xl_base_1.0.safetensors"},
          "_meta": {"title": "Load Checkpoint"}},
    "5": {"class_type": "EmptyLatentImage", "inputs": {"width": 1024, "height": 1024, "batch_size": 1}},
    "6": {"class_type": "CLIPTextEncode", "inputs": {"text": "a cafe", "clip": ["4", 1]},
          "_meta": {"title": "Positive"}},
    "7": {"class_type": "CLIPTextEncode", "inputs": {"text": "blurry", "clip": ["4", 1]},
          "_meta": {"title": "Negative"}},
    "3": {"class_type": "KSampler", "inputs": {"seed": 42, "steps": 20, "cfg": 7.0, "sampler_name": "euler",
                                               "scheduler": "normal", "denoise": 1.0, "model": ["4", 0],
                                               "positive": ["6", 0], "negative": ["7", 0], "latent_image": ["5", 0]}},
    "8": {"class_type": "VAEDecode", "inputs": {"samples": ["3", 0], "vae": ["4", 2]}},
    "9": {"class_type": "SaveImage", "inputs": {"filename_prefix": "FilmHouse", "images": ["8", 0]}},
}
MAPPING = {
    "prompt": {"node_id": "6", "input": "text"},
    "negative_prompt": {"node_id": "7", "input": "text"},
    "seed": {"node_id": "3", "input": "seed"},
    "width": {"node_id": "5", "input": "width"},
    "height": {"node_id": "5", "input": "height"},
    "reference_image": None,
    "output_nodes": ["9"],
}


def _response(payload, status=200, content_type="application/json"):
    body = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
    stream = io.BytesIO(body)
    stream.status = status
    stream.headers = {"Content-Type": content_type}
    stream.__enter__ = lambda: stream
    stream.__exit__ = lambda *a: None
    return stream


@pytest.fixture
def captured(monkeypatch):
    """Record requests and return queued replies instead of opening sockets."""
    calls = []
    replies = {}

    def fake_urlopen(request, timeout=None):
        calls.append({"url": request.full_url, "method": request.get_method(), "timeout": timeout,
                      "body": json.loads(request.data) if request.data and
                      request.headers.get("Content-type", "").startswith("application/json") else request.data,
                      "headers": dict(request.headers)})
        for key, value in replies.items():
            if key in request.full_url:
                if isinstance(value, Exception):
                    raise value
                return _response(value)
        return _response({})

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    return {"calls": calls, "replies": replies}


# --------------------------------------------------------------------------- localhost only

@pytest.mark.parametrize("url", ["https://comfy.example.com", "http://192.168.0.5:8188", "", "ftp://127.0.0.1"])
def test_client_refuses_non_local_urls(url, monkeypatch):
    def boom(*args, **kwargs):
        raise AssertionError("a non-local request was attempted")
    monkeypatch.setattr(urllib.request, "urlopen", boom)
    with pytest.raises(ComfyError) as info:
        ComfyClient(url)
    assert info.value.code == "NOT_LOCAL" and "this computer" in info.value.message


def test_adapter_validate_config_reports_unavailable_without_raising(captured):
    captured["replies"]["/system_stats"] = urllib.error.URLError("connection refused")
    result = ComfyAdapter(ComfyClient(LOCAL)).validate_config()
    assert result["ok"] is False and result["code"] == "UNREACHABLE"
    assert "Start ComfyUI" in result["message"]


def test_adapter_validate_config_success(captured):
    captured["replies"]["/system_stats"] = {
        "system": {"comfyui_version": "0.3.60", "pytorch_version": "2.7.0+cu128"},
        "devices": [{"name": "cuda:0 NVIDIA GeForce RTX 4090", "type": "cuda", "vram_total": 25_757_220_864}]}
    result = ComfyAdapter(ComfyClient(LOCAL)).validate_config()
    assert result["ok"] and result["comfyui_version"] == "0.3.60" and result["gpu_available"] is True
    assert result["devices"][0]["vram_total_gb"] == 24.0


def test_capabilities_are_honest_about_progress():
    caps = ComfyAdapter().capabilities()
    assert caps["local_only"] is True and caps["features"]["progress_percent"] is False
    assert caps["features"]["cancel_running"] and caps["features"]["cancel_queued"]


# --------------------------------------------------------------------------- client behaviour

def test_timeouts_are_always_set(captured):
    captured["replies"]["/system_stats"] = {"system": {}, "devices": []}
    ComfyClient(LOCAL).system_stats()
    assert captured["calls"][0]["timeout"] == 5.0


def test_timeout_error_is_readable(captured):
    captured["replies"]["/system_stats"] = TimeoutError()
    with pytest.raises(ComfyError) as info:
        ComfyClient(LOCAL).system_stats()
    assert info.value.code == "TIMEOUT" and "did not answer" in info.value.message


def test_submit_sends_prompt_and_client_id(captured):
    captured["replies"]["/prompt"] = {"prompt_id": "abc-123", "number": 1, "node_errors": {}}
    client = ComfyClient(LOCAL, client_id="cid")
    assert client.submit(WORKFLOW) == "abc-123"
    call = captured["calls"][0]
    assert call["method"] == "POST" and call["url"] == f"{LOCAL}/prompt"
    assert call["body"]["client_id"] == "cid" and call["body"]["prompt"]["9"]["class_type"] == "SaveImage"


def test_submit_translates_comfy_validation_errors(captured):
    detail = {"error": {"type": "prompt_outputs_failed_validation", "message": "Prompt outputs failed validation"},
              "node_errors": {"4": {"errors": [{"message": "Value not in list: ckpt_name",
                                                "extra_info": {"input_name": "ckpt_name"}}]}}}
    captured["replies"]["/prompt"] = urllib.error.HTTPError(f"{LOCAL}/prompt", 400, "Bad Request", {},
                                                            io.BytesIO(json.dumps(detail).encode()))
    with pytest.raises(ComfyError) as info:
        ComfyClient(LOCAL).submit(WORKFLOW)
    assert info.value.code == "REJECTED"
    assert "Prompt outputs failed validation" in info.value.message
    assert "node 4 input 'ckpt_name'" in info.value.message


def test_submit_without_prompt_id_is_an_error(captured):
    captured["replies"]["/prompt"] = {"error": "no"}
    with pytest.raises(ComfyError) as info:
        ComfyClient(LOCAL).submit(WORKFLOW)
    assert info.value.code == "REJECTED" and "nothing was queued" in info.value.message


def test_job_state_pending_running_done_missing(captured):
    client = ComfyClient(LOCAL)
    captured["replies"]["/queue"] = {"queue_running": [[1, "running-id", {}, {}, []]],
                                     "queue_pending": [[2, "a", {}, {}, []], [3, "mine", {}, {}, []]]}
    captured["replies"]["/history/"] = {}
    assert client.job_state("mine") == {"state": "pending", "queue_position": 2, "queue_pending": 2}
    assert client.job_state("running-id")["state"] == "running"
    assert client.job_state("gone")["state"] == "missing"
    captured["replies"]["/history/"] = {"done-id": {"status": {"status_str": "success", "completed": True},
                                                    "outputs": {}}}
    state = client.job_state("done-id")
    assert state["state"] == "done" and state["status_str"] == "success"


def test_cancel_uses_delete_for_pending_and_interrupt_for_running(captured):
    adapter = ComfyAdapter(ComfyClient(LOCAL))
    captured["replies"]["/queue"] = {"queue_running": [[1, "run", {}, {}, []]],
                                     "queue_pending": [[2, "pend", {}, {}, []]]}
    captured["replies"]["/history/"] = {}
    result = adapter.cancel("pend")
    assert result["cancelled"] and "queue" in result["how"]
    assert captured["calls"][-1]["body"] == {"delete": ["pend"]}
    result = adapter.cancel("run")
    assert result["cancelled"] and "interrupted" in result["how"]
    assert captured["calls"][-1]["url"].endswith("/interrupt")
    result = adapter.cancel("unknown")
    assert result["cancelled"] is False


def test_collect_outputs_reads_history_images(captured):
    captured["replies"]["/history/"] = {"p1": {"outputs": {
        "9": {"images": [{"filename": "FilmHouse_00001_.png", "subfolder": "", "type": "output"}]},
        "10": {"images": [{"filename": "x.png", "subfolder": "sub", "type": "temp"}]},
        "11": {"text": ["not an image"]}}}}
    outputs = ComfyAdapter(ComfyClient(LOCAL)).collect_outputs("p1")
    assert [o["filename"] for o in outputs] == ["FilmHouse_00001_.png", "x.png"]
    assert outputs[1]["subfolder"] == "sub" and outputs[1]["type"] == "temp"
    captured["replies"]["/history/"] = {}
    assert ComfyAdapter(ComfyClient(LOCAL)).collect_outputs("p1") == []


def test_download_output_uses_view_endpoint(captured):
    captured["replies"]["/view"] = b"\x89PNG\r\n\x1a\nfake"
    data = ComfyClient(LOCAL).download_output("a.png", "sub", "output")
    assert data.startswith(b"\x89PNG")
    assert "filename=a.png" in captured["calls"][0]["url"] and "subfolder=sub" in captured["calls"][0]["url"]


def test_upload_image_posts_multipart(captured, tmp_path):
    captured["replies"]["/upload/image"] = {"name": "ref.png", "subfolder": "ai-film-house", "type": "input"}
    image = tmp_path / "ref.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\ndata")
    result = ComfyClient(LOCAL).upload_image(image)
    assert result["name"] == "ai-film-house/ref.png"
    call = captured["calls"][0]
    assert call["method"] == "POST" and "multipart/form-data; boundary=" in call["headers"]["Content-type"]
    assert b'name="image"; filename="ref.png"' in call["body"] and b"\x89PNG" in call["body"]


def test_models_listing_tolerates_missing_endpoint(captured):
    captured["replies"]["/models/checkpoints"] = urllib.error.HTTPError(
        f"{LOCAL}/models/checkpoints", 404, "Not Found", {}, io.BytesIO(b"{}"))
    assert ComfyClient(LOCAL).models("checkpoints") == []


# --------------------------------------------------------------------------- workflow parsing

def test_load_and_inspect_api_workflow():
    workflow = load_api_workflow(WORKFLOW)
    inspection = inspect_workflow(workflow)
    assert inspection["node_count"] == 7
    assert inspection["output_nodes"] == ["9"] and inspection["saving_output_nodes"] == ["9"]
    assert inspection["class_types"]["CLIPTextEncode"] == 2
    latent = next(n for n in inspection["nodes"] if n["id"] == "5")
    assert {f["input"] for f in latent["fields"]} == {"width", "height", "batch_size"}
    sampler = next(n for n in inspection["nodes"] if n["id"] == "3")
    assert {w["input"] for w in sampler["wired_inputs"]} == {"model", "positive", "negative", "latent_image"}
    assert next(f for f in sampler["fields"] if f["input"] == "seed")["roles"] == ["seed", "width", "height"]
    assert workflow_models(workflow) == [{"node_id": "4", "class_type": "CheckpointLoaderSimple",
                                          "input": "ckpt_name", "name": "sd_xl_base_1.0.safetensors",
                                          "folder": "checkpoints"}]
    assert next(n for n in inspection["nodes"] if n["id"] == "6")["title"] == "Positive"


def test_ui_format_is_rejected_with_advice():
    ui = {"last_node_id": 9, "nodes": [{"id": 1, "type": "KSampler"}], "links": [], "groups": []}
    with pytest.raises(WorkflowError) as info:
        load_api_workflow(ui)
    assert info.value.code == "UI_FORMAT" and "Export (API)" in info.value.message


@pytest.mark.parametrize("bad,code", [
    ("{not json", "INVALID_JSON"),
    ([], "NOT_API_FORMAT"),
    ({}, "EMPTY_WORKFLOW"),
    ({"1": {"inputs": {}}}, "NOT_API_FORMAT"),
    ({"1": "nope"}, "NOT_API_FORMAT"),
    ({"1": {"class_type": "X", "inputs": []}}, "NOT_API_FORMAT"),
    ({"1": {"class_type": "X", "inputs": {"a": ["99", 0]}}}, "BROKEN_LINK"),
])
def test_malformed_workflows_are_rejected(bad, code):
    with pytest.raises(WorkflowError) as info:
        load_api_workflow(bad)
    assert info.value.code == code


def test_whole_prompt_request_body_is_unwrapped():
    workflow = load_api_workflow({"prompt": WORKFLOW, "client_id": "x"})
    assert "9" in workflow and workflow["9"]["class_type"] == "SaveImage"


def test_suggestions_trace_sampler_links_but_are_not_applied():
    suggestions = suggest_mapping(load_api_workflow(WORKFLOW))
    assert suggestions["prompt"]["node_id"] == "6" and suggestions["negative_prompt"]["node_id"] == "7"
    assert suggestions["seed"] == {"node_id": "3", "input": "seed",
                                   "why": "KSampler has a 'seed' input"}
    assert suggestions["width"]["node_id"] == "5" and suggestions["height"]["node_id"] == "5"
    assert "reference_image" not in suggestions
    # Suggestions are not a mapping: an imported workflow starts unmapped.
    assert validate_mapping(load_api_workflow(WORKFLOW), {"output_nodes": ["9"]})["valid"] is False


# --------------------------------------------------------------------------- mapping validation

def test_valid_mapping_passes():
    result = validate_mapping(load_api_workflow(WORKFLOW), MAPPING)
    assert result["valid"] and result["errors"] == []
    assert result["mapped_roles"] == ["height", "negative_prompt", "prompt", "seed", "width"]
    assert result["warnings"] == []


def test_prompt_is_required():
    result = validate_mapping(load_api_workflow(WORKFLOW), {**MAPPING, "prompt": None})
    assert not result["valid"]
    assert result["errors"][0]["code"] == "ROLE_UNMAPPED" and result["errors"][0]["role"] == "prompt"


def test_unmapped_optional_roles_warn_about_consequences():
    result = validate_mapping(load_api_workflow(WORKFLOW),
                              {"prompt": {"node_id": "6", "input": "text"}, "output_nodes": ["9"]})
    assert result["valid"] is True
    codes = {(w["role"], w["code"]) for w in result["warnings"]}
    assert ("seed", "ROLE_UNMAPPED") in codes and ("width", "ROLE_UNMAPPED") in codes
    assert any("reproduce runs" in w["message"] for w in result["warnings"])


@pytest.mark.parametrize("mapping,code,fragment", [
    ({"prompt": {"node_id": "99", "input": "text"}}, "NODE_MISSING", "not in this workflow"),
    ({"prompt": {"node_id": "6", "input": "nope"}}, "INPUT_MISSING", "has no input 'nope'"),
    ({"prompt": {"node_id": "3", "input": "positive"}}, "INPUT_WIRED", "is wired from node"),
    ({"prompt": {"node_id": "3", "input": "seed"}}, "TYPE_MISMATCH", "needs a str input"),
    ({"prompt": {"node_id": "6", "input": "text"}, "seed": {"node_id": "6", "input": "text"}},
     "TYPE_MISMATCH", "needs a int input"),
])
def test_mapping_errors_are_specific(mapping, code, fragment):
    result = validate_mapping(load_api_workflow(WORKFLOW), {**mapping, "output_nodes": ["9"]})
    assert not result["valid"]
    assert any(e["code"] == code and fragment in e["message"] for e in result["errors"]), result["errors"]


def test_duplicate_targets_are_rejected():
    mapping = {"prompt": {"node_id": "6", "input": "text"}, "negative_prompt": {"node_id": "6", "input": "text"},
               "output_nodes": ["9"]}
    result = validate_mapping(load_api_workflow(WORKFLOW), mapping)
    assert any(e["code"] == "DUPLICATE_TARGET" for e in result["errors"])


def test_workflow_without_save_node_is_rejected():
    workflow = {k: v for k, v in WORKFLOW.items() if k != "9"}
    result = validate_mapping(load_api_workflow(workflow), {"prompt": {"node_id": "6", "input": "text"}})
    assert not result["valid"]
    assert any(e["code"] == "NO_OUTPUT_NODE" for e in result["errors"])


def test_preview_only_workflow_warns():
    workflow = dict(WORKFLOW)
    workflow["9"] = {"class_type": "PreviewImage", "inputs": {"images": ["8", 0]}}
    result = validate_mapping(load_api_workflow(workflow), MAPPING)
    assert result["valid"] and any(w["code"] == "PREVIEW_ONLY" for w in result["warnings"])


def test_reference_image_mapping_checks_filename_shape():
    workflow = dict(WORKFLOW)
    workflow["12"] = {"class_type": "LoadImage", "inputs": {"image": "example.png", "upload": "image"}}
    mapping = {**MAPPING, "reference_image": {"node_id": "12", "input": "image"}}
    assert validate_mapping(load_api_workflow(workflow), mapping)["valid"]
    assert suggest_mapping(load_api_workflow(workflow))["reference_image"]["node_id"] == "12"
    workflow["12"]["inputs"]["image"] = "not-a-file"
    result = validate_mapping(load_api_workflow(workflow), mapping)
    assert result["valid"] and any(w["code"] == "REFERENCE_INPUT_ODD" for w in result["warnings"])


# --------------------------------------------------------------------------- live validation

OBJECT_INFO = {
    "CheckpointLoaderSimple": {"input": {"required": {"ckpt_name": [["sd_xl_base_1.0.safetensors"]]}}},
    "EmptyLatentImage": {"input": {"required": {"width": ["INT"], "height": ["INT"], "batch_size": ["INT"]}}},
    "CLIPTextEncode": {"input": {"required": {"text": ["STRING"], "clip": ["CLIP"]}}},
    "KSampler": {"input": {"required": {"seed": ["INT"], "steps": ["INT"], "cfg": ["FLOAT"],
                                        "sampler_name": [[]], "scheduler": [[]], "denoise": ["FLOAT"],
                                        "model": ["MODEL"], "positive": ["CONDITIONING"],
                                        "negative": ["CONDITIONING"], "latent_image": ["LATENT"]}}},
    "VAEDecode": {"input": {"required": {"samples": ["LATENT"], "vae": ["VAE"]}}},
    "SaveImage": {"input": {"required": {"images": ["IMAGE"], "filename_prefix": ["STRING"]}}},
}


def test_live_validation_passes_with_matching_server():
    result = validate_against_server(load_api_workflow(WORKFLOW), MAPPING, OBJECT_INFO,
                                     {"checkpoints": ["sd_xl_base_1.0.safetensors"]})
    assert result["valid"] and result["errors"] == [] and result["warnings"] == []


def test_live_validation_reports_missing_custom_nodes():
    workflow = dict(WORKFLOW)
    workflow["20"] = {"class_type": "SuperCustomSampler", "inputs": {"x": 1}}
    result = validate_against_server(load_api_workflow(workflow), MAPPING, OBJECT_INFO, {})
    assert not result["valid"]
    assert result["errors"][0]["code"] == "UNKNOWN_NODE_CLASS"
    assert "SuperCustomSampler" in result["errors"][0]["message"]


def test_live_validation_reports_input_not_in_this_build():
    trimmed = {**OBJECT_INFO, "CLIPTextEncode": {"input": {"required": {"prompt": ["STRING"], "clip": ["CLIP"]}}}}
    result = validate_against_server(load_api_workflow(WORKFLOW), MAPPING, trimmed, {})
    assert not result["valid"]
    assert any(e["code"] == "INPUT_NOT_IN_NODE" and "has no input 'text'" in e["message"] for e in result["errors"])


def test_live_validation_warns_about_missing_model_file():
    result = validate_against_server(load_api_workflow(WORKFLOW), MAPPING, OBJECT_INFO,
                                     {"checkpoints": ["something_else.safetensors"]})
    assert result["valid"]
    assert result["warnings"][0]["code"] == "MODEL_NOT_FOUND"
    assert "sd_xl_base_1.0.safetensors" in result["warnings"][0]["message"]


# --------------------------------------------------------------------------- patching

def test_apply_mapping_patches_only_mapped_inputs():
    workflow = load_api_workflow(WORKFLOW)
    patched = apply_mapping(workflow, MAPPING, {"prompt": "a rooftop at dusk", "negative_prompt": "text, watermark",
                                                "seed": 777, "width": 832, "height": 1216})
    assert patched["6"]["inputs"]["text"] == "a rooftop at dusk"
    assert patched["7"]["inputs"]["text"] == "text, watermark"
    assert patched["3"]["inputs"]["seed"] == 777 and patched["3"]["inputs"]["steps"] == 20
    assert patched["5"]["inputs"]["width"] == 832 and patched["5"]["inputs"]["height"] == 1216
    assert workflow["6"]["inputs"]["text"] == "a cafe"          # original untouched
    assert patched["3"]["inputs"]["positive"] == ["6", 0]       # links preserved


def test_apply_mapping_refuses_unmapped_role():
    with pytest.raises(WorkflowError) as info:
        apply_mapping(load_api_workflow(WORKFLOW), {**MAPPING, "seed": None}, {"prompt": "x", "seed": 5})
    assert info.value.code == "ROLE_UNMAPPED"


@pytest.mark.parametrize("values,fragment", [
    ({"seed": -1}, "between 0"), ({"seed": MAX_SEED + 1}, "between 0"), ({"seed": "abc"}, "whole number"),
    ({"width": 10}, "between 64"), ({"height": 99999}, "between 64"), ({"prompt": "x" * 20_001}, "too long"),
])
def test_apply_mapping_validates_values(values, fragment):
    with pytest.raises(WorkflowError) as info:
        apply_mapping(load_api_workflow(WORKFLOW), MAPPING, {"prompt": "ok", **values})
    assert info.value.code == "BAD_VALUE" and fragment in info.value.message


def test_apply_mapping_detects_stale_mapping():
    workflow = load_api_workflow(WORKFLOW)
    del workflow["6"]["inputs"]["text"]
    with pytest.raises(WorkflowError) as info:
        apply_mapping(workflow, MAPPING, {"prompt": "x"})
    assert info.value.code == "MAPPING_STALE"


def test_roles_cover_the_phase4_requirements():
    assert set(ROLE_SPECS) == {"prompt", "negative_prompt", "seed", "width", "height", "reference_image"}
    assert ROLE_SPECS["prompt"]["required"] is True
    assert all(spec["required"] is False for role, spec in ROLE_SPECS.items() if role != "prompt")


def test_output_node_detection_covers_common_classes():
    workflow = {"1": {"class_type": "SaveImage", "inputs": {}},
                "2": {"class_type": "PreviewImage", "inputs": {}},
                "3": {"class_type": "SaveAnimatedWEBP", "inputs": {}},
                "4": {"class_type": "KSampler", "inputs": {}}}
    assert output_node_ids(load_api_workflow(workflow)) == ["1", "2", "3"]
