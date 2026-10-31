"""ComfyUI API-format workflow handling: parse, inspect, map, validate, patch.

The app never guesses how a workflow works. You import a workflow saved with ComfyUI's
"Save (API Format)", then map each role (prompt, negative prompt, seed, width, height, reference
image) to an explicit node id and input name. Mapping and validation are per workflow: two
workflows are never assumed to be interchangeable.
"""
from __future__ import annotations

import copy
import json
import re
from typing import Any, Iterable

MAX_NODES = 2000
MAX_SEED = 2 ** 63 - 1
MIN_DIMENSION, MAX_DIMENSION = 64, 8192

# Roles the app can drive. Each needs a literal (not wired) input of the given type.
ROLE_SPECS: dict[str, dict[str, Any]] = {
    "prompt": {"label": "Positive prompt", "types": ("str",), "required": True,
               "help": "Text input of the node that receives your shot description."},
    "negative_prompt": {"label": "Negative prompt", "types": ("str",), "required": False,
                        "help": "Text input of the negative conditioning node."},
    "seed": {"label": "Seed", "types": ("int",), "required": False,
             "help": "Sampler seed. Without it, runs are not reproducible from this app."},
    "width": {"label": "Width", "types": ("int",), "required": False,
              "help": "Latent/image width in pixels."},
    "height": {"label": "Height", "types": ("int",), "required": False,
               "help": "Latent/image height in pixels."},
    "reference_image": {"label": "Reference image", "types": ("str",), "required": False,
                        "help": "Filename input of a LoadImage node, for image-guided workflows."},
}
ROLES = tuple(ROLE_SPECS)

SAVE_CLASS_HINTS = ("SaveImage", "PreviewImage", "SaveAnimatedWEBP", "SaveAnimatedPNG", "SaveImageWebsocket")
MODEL_INPUT_NAMES = ("ckpt_name", "unet_name", "vae_name", "lora_name", "clip_name", "clip_name1", "clip_name2",
                     "control_net_name", "model_name", "style_model_name", "upscale_model_name")
MODEL_FOLDER_BY_INPUT = {
    "ckpt_name": "checkpoints", "unet_name": "diffusion_models", "vae_name": "vae", "lora_name": "loras",
    "clip_name": "text_encoders", "clip_name1": "text_encoders", "clip_name2": "text_encoders",
    "control_net_name": "controlnet", "style_model_name": "style_models", "upscale_model_name": "upscale_models",
}
_SEED_INPUTS = ("seed", "noise_seed")


class WorkflowError(Exception):
    def __init__(self, code: str, message: str, detail: Any = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.detail = detail

    def to_dict(self) -> dict[str, Any]:
        return {"code": self.code, "message": self.message, "detail": self.detail}


# --------------------------------------------------------------------------- parsing

def _value_type(value: Any) -> str | None:
    """Type name of a literal input value, or None if it is a link/unsupported."""
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "str"
    return None


def is_link(value: Any) -> bool:
    """ComfyUI wires inputs as [source_node_id, output_slot]."""
    return isinstance(value, (list, tuple)) and len(value) == 2 and isinstance(value[1], (int, float))


def load_api_workflow(data: Any) -> dict[str, dict[str, Any]]:
    """Validate and normalise a ComfyUI API-format workflow. Raises WorkflowError with advice."""
    if isinstance(data, (str, bytes)):
        try:
            data = json.loads(data)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise WorkflowError("INVALID_JSON", f"That file is not valid JSON ({exc}).") from exc
    if not isinstance(data, dict):
        raise WorkflowError("NOT_API_FORMAT", "A ComfyUI API workflow must be a JSON object of nodes "
                                              "keyed by node id.")
    if isinstance(data.get("prompt"), dict) and "class_type" in str(data.get("prompt")):
        data = data["prompt"]  # someone pasted a whole /prompt request body
    if isinstance(data.get("nodes"), list):
        raise WorkflowError(
            "UI_FORMAT",
            "This looks like a ComfyUI editor workflow, not an API workflow. In ComfyUI use "
            "Workflow → Export (API) (older builds: Save (API Format)) and import that file instead.",
            {"found_keys": sorted(k for k in data if k in ("nodes", "links", "last_node_id", "groups"))},
        )
    if not data:
        raise WorkflowError("EMPTY_WORKFLOW", "The workflow has no nodes.")
    if len(data) > MAX_NODES:
        raise WorkflowError("TOO_MANY_NODES", f"The workflow has {len(data)} nodes; the limit is {MAX_NODES}.")

    workflow: dict[str, dict[str, Any]] = {}
    for raw_id, node in data.items():
        node_id = str(raw_id)
        if not isinstance(node, dict):
            raise WorkflowError("NOT_API_FORMAT", f"Node '{node_id}' is not an object. "
                                                  "Make sure you exported the API format.")
        class_type = node.get("class_type")
        if not isinstance(class_type, str) or not class_type.strip():
            raise WorkflowError("NOT_API_FORMAT", f"Node '{node_id}' has no 'class_type'. "
                                                  "Make sure you exported the API format.")
        inputs = node.get("inputs", {})
        if not isinstance(inputs, dict):
            raise WorkflowError("NOT_API_FORMAT", f"Node '{node_id}' ({class_type}) has invalid 'inputs'.")
        meta = node.get("_meta") if isinstance(node.get("_meta"), dict) else {}
        workflow[node_id] = {"class_type": class_type.strip(), "inputs": copy.deepcopy(inputs), "_meta": meta}

    for node_id, node in workflow.items():
        for name, value in node["inputs"].items():
            if is_link(value) and str(value[0]) not in workflow:
                raise WorkflowError("BROKEN_LINK",
                                    f"Node '{node_id}' ({node['class_type']}) input '{name}' is wired to node "
                                    f"'{value[0]}', which is not in the file.")
    return workflow


def node_title(node: dict[str, Any]) -> str:
    meta = node.get("_meta") or {}
    title = meta.get("title")
    return str(title) if isinstance(title, str) and title.strip() else str(node["class_type"])


def is_output_node(node: dict[str, Any]) -> bool:
    class_type = node["class_type"]
    return class_type in SAVE_CLASS_HINTS or class_type.startswith(("Save", "Preview"))


def output_node_ids(workflow: dict[str, dict[str, Any]]) -> list[str]:
    return sorted((nid for nid, node in workflow.items() if is_output_node(node)), key=_as_sort_key)


def saving_output_node_ids(workflow: dict[str, dict[str, Any]]) -> list[str]:
    """Output nodes that write a file (Preview nodes keep images in the temp folder)."""
    return [nid for nid in output_node_ids(workflow)
            if not workflow[nid]["class_type"].startswith("Preview")]


def workflow_models(workflow: dict[str, dict[str, Any]]) -> list[dict[str, str]]:
    """Model files the workflow names, for the job's metadata record."""
    found: list[dict[str, str]] = []
    for node_id in sorted(workflow, key=_as_sort_key):
        node = workflow[node_id]
        for name, value in node["inputs"].items():
            if name in MODEL_INPUT_NAMES and isinstance(value, str) and value.strip():
                found.append({"node_id": node_id, "class_type": node["class_type"], "input": name,
                              "name": value, "folder": MODEL_FOLDER_BY_INPUT.get(name, "")})
    return found


def _as_sort_key(node_id: str) -> tuple[int, str]:
    return (0, f"{int(node_id):09d}") if node_id.isdigit() else (1, node_id)


# --------------------------------------------------------------------------- inspection

def inspect_workflow(workflow: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Everything the UI needs to offer an explicit mapping."""
    nodes = []
    for node_id in sorted(workflow, key=_as_sort_key):
        node = workflow[node_id]
        fields, wired = [], []
        for name, value in node["inputs"].items():
            if is_link(value):
                wired.append({"input": name, "from_node": str(value[0])})
                continue
            kind = _value_type(value)
            if kind is None:
                continue
            fields.append({"input": name, "type": kind,
                           "value": value if kind != "str" else str(value)[:400],
                           "roles": [r for r, spec in ROLE_SPECS.items() if kind in spec["types"]]})
        nodes.append({"id": node_id, "class_type": node["class_type"], "title": node_title(node),
                      "is_output": is_output_node(node), "fields": fields, "wired_inputs": wired})
    class_counts: dict[str, int] = {}
    for node in workflow.values():
        class_counts[node["class_type"]] = class_counts.get(node["class_type"], 0) + 1
    return {
        "node_count": len(workflow),
        "nodes": nodes,
        "class_types": dict(sorted(class_counts.items())),
        "output_nodes": output_node_ids(workflow),
        "saving_output_nodes": saving_output_node_ids(workflow),
        "models": workflow_models(workflow),
    }


def _text_nodes_for_sampler(workflow: dict[str, dict[str, Any]]) -> dict[str, str]:
    """Trace sampler positive/negative links back to the node that holds the text."""
    found: dict[str, str] = {}
    for node in workflow.values():
        for role, key in (("prompt", "positive"), ("negative_prompt", "negative")):
            target = node["inputs"].get(key)
            if not is_link(target):
                continue
            seen: set[str] = set()
            current = str(target[0])
            while current in workflow and current not in seen:
                seen.add(current)
                candidate = workflow[current]
                text = candidate["inputs"].get("text")
                if isinstance(text, str):
                    found.setdefault(role, current)
                    break
                next_link = next((v for v in candidate["inputs"].values() if is_link(v)), None)
                if next_link is None:
                    break
                current = str(next_link[0])
    return found


def suggest_mapping(workflow: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Best-effort suggestions. They are shown to you, never applied automatically."""
    suggestions: dict[str, dict[str, Any]] = {}
    traced = _text_nodes_for_sampler(workflow)
    for role, node_id in traced.items():
        suggestions[role] = {"node_id": node_id, "input": "text",
                             "why": f"wired to a sampler's {'positive' if role == 'prompt' else 'negative'} input"}
    if "prompt" not in suggestions:
        text_nodes = [nid for nid in sorted(workflow, key=_as_sort_key)
                      if isinstance(workflow[nid]["inputs"].get("text"), str)]
        if text_nodes:
            suggestions["prompt"] = {"node_id": text_nodes[0], "input": "text",
                                     "why": "first node with a literal 'text' input"}
        if len(text_nodes) > 1 and "negative_prompt" not in suggestions:
            suggestions["negative_prompt"] = {"node_id": text_nodes[1], "input": "text",
                                              "why": "second node with a literal 'text' input"}
    for node_id in sorted(workflow, key=_as_sort_key):
        node = workflow[node_id]
        for name in _SEED_INPUTS:
            value = node["inputs"].get(name)
            if "seed" not in suggestions and isinstance(value, int) and not isinstance(value, bool):
                suggestions["seed"] = {"node_id": node_id, "input": name,
                                       "why": f"{node['class_type']} has a '{name}' input"}
        for dimension in ("width", "height"):
            value = node["inputs"].get(dimension)
            if dimension not in suggestions and isinstance(value, int) and not isinstance(value, bool):
                suggestions[dimension] = {"node_id": node_id, "input": dimension,
                                          "why": f"{node['class_type']} has a '{dimension}' input"}
        if "reference_image" not in suggestions and isinstance(node["inputs"].get("image"), str):
            suggestions["reference_image"] = {"node_id": node_id, "input": "image",
                                              "why": f"{node['class_type']} loads an image file"}
    return suggestions


# --------------------------------------------------------------------------- mapping validation

def _problem(code: str, message: str, role: str | None = None, node_id: str | None = None) -> dict[str, Any]:
    return {"code": code, "message": message, "role": role, "node_id": node_id}


def _mapping_items(mapping: Any) -> list[tuple[str, dict[str, Any]]]:
    """Accept a dict or a Pydantic model; yield (role, {node_id, input}) for mapped roles."""
    if hasattr(mapping, "model_dump"):
        mapping = mapping.model_dump()
    if not isinstance(mapping, dict):
        return []
    items = []
    for role in ROLES:
        ref = mapping.get(role)
        if hasattr(ref, "model_dump"):
            ref = ref.model_dump()
        if isinstance(ref, dict) and ref.get("node_id") and ref.get("input"):
            items.append((role, {"node_id": str(ref["node_id"]), "input": str(ref["input"])}))
    return items


def _mapping_outputs(mapping: Any) -> list[str]:
    if hasattr(mapping, "model_dump"):
        mapping = mapping.model_dump()
    values = (mapping or {}).get("output_nodes") if isinstance(mapping, dict) else None
    return [str(v) for v in values] if isinstance(values, list) else []


def validate_mapping(workflow: dict[str, dict[str, Any]], mapping: Any) -> dict[str, Any]:
    """Offline checks: every mapped role must point at a literal input of the right type."""
    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    mapped = dict(_mapping_items(mapping))

    for role, spec in ROLE_SPECS.items():
        ref = mapped.get(role)
        if ref is None:
            if spec["required"]:
                errors.append(_problem("ROLE_UNMAPPED", f"{spec['label']} is not mapped. {spec['help']}", role))
            elif role in ("seed", "width", "height"):
                warnings.append(_problem(
                    "ROLE_UNMAPPED",
                    f"{spec['label']} is not mapped, so the workflow's own value is used and this app cannot "
                    f"{'reproduce runs' if role == 'seed' else 'change the output size'}.", role))
            continue
        node = workflow.get(ref["node_id"])
        if node is None:
            errors.append(_problem("NODE_MISSING", f"{spec['label']}: node '{ref['node_id']}' is not in this "
                                                   "workflow.", role, ref["node_id"]))
            continue
        if ref["input"] not in node["inputs"]:
            available = ", ".join(sorted(k for k, v in node["inputs"].items() if not is_link(v))) or "none"
            errors.append(_problem("INPUT_MISSING",
                                   f"{spec['label']}: node {ref['node_id']} ({node['class_type']}) has no input "
                                   f"'{ref['input']}'. Literal inputs: {available}.", role, ref["node_id"]))
            continue
        value = node["inputs"][ref["input"]]
        if is_link(value):
            errors.append(_problem("INPUT_WIRED",
                                   f"{spec['label']}: node {ref['node_id']} input '{ref['input']}' is wired from "
                                   f"node {value[0]}, so this app cannot set it. Pick a node whose value is typed "
                                   "in, or disconnect that link in ComfyUI.", role, ref["node_id"]))
            continue
        kind = _value_type(value)
        if kind not in spec["types"]:
            errors.append(_problem("TYPE_MISMATCH",
                                   f"{spec['label']} needs a {'/'.join(spec['types'])} input but node "
                                   f"{ref['node_id']} input '{ref['input']}' currently holds "
                                   f"{kind or 'an unsupported value'}.", role, ref["node_id"]))
            continue
        if role == "reference_image" and not re.search(r"\.(png|jpe?g|webp|bmp|tiff?)$", str(value), re.IGNORECASE):
            warnings.append(_problem("REFERENCE_INPUT_ODD",
                                     f"Reference image: node {ref['node_id']} input '{ref['input']}' holds "
                                     f"'{str(value)[:60]}', which does not look like an image filename.",
                                     role, ref["node_id"]))

    collisions: dict[tuple[str, str], list[str]] = {}
    for role, ref in mapped.items():
        collisions.setdefault((ref["node_id"], ref["input"]), []).append(role)
    for (node_id, input_name), roles in collisions.items():
        if len(roles) > 1:
            errors.append(_problem("DUPLICATE_TARGET", f"Roles {', '.join(roles)} all point at node {node_id} "
                                                       f"input '{input_name}'.", roles[0], node_id))

    chosen = _mapping_outputs(mapping)
    available_outputs = output_node_ids(workflow)
    saving = saving_output_node_ids(workflow)
    for node_id in chosen:
        if node_id not in workflow:
            errors.append(_problem("NODE_MISSING", f"Output node '{node_id}' is not in this workflow.",
                                   None, node_id))
        elif not is_output_node(workflow[node_id]):
            warnings.append(_problem("OUTPUT_NODE_ODD",
                                     f"Node {node_id} ({workflow[node_id]['class_type']}) is not a Save/Preview "
                                     "node; it may produce no files.", None, node_id))
    if not available_outputs:
        errors.append(_problem("NO_OUTPUT_NODE", "This workflow has no SaveImage/PreviewImage node, so nothing "
                                                 "could be collected. Add one in ComfyUI and export again."))
    elif not saving:
        warnings.append(_problem("PREVIEW_ONLY", "This workflow only previews images (no SaveImage node). "
                                                 "ComfyUI writes previews to its temp folder; outputs are still "
                                                 "copied here, but add a SaveImage node for reliable results."))
    return {"errors": errors, "warnings": warnings, "valid": not errors,
            "mapped_roles": sorted(mapped), "output_nodes": chosen or available_outputs}


def validate_against_server(workflow: dict[str, dict[str, Any]], mapping: Any, object_info: dict[str, Any],
                            installed_models: dict[str, Iterable[str]] | None = None) -> dict[str, Any]:
    """Live checks against this ComfyUI: node classes exist, mapped inputs exist, models are installed."""
    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    missing_classes = sorted({node["class_type"] for node in workflow.values() if node["class_type"] not in object_info})
    if missing_classes:
        errors.append(_problem("UNKNOWN_NODE_CLASS",
                               "This ComfyUI does not have these node types (install the custom nodes they come "
                               f"from, then validate again): {', '.join(missing_classes[:10])}"
                               + (" …" if len(missing_classes) > 10 else "")))

    for role, ref in _mapping_items(mapping):
        node = workflow.get(ref["node_id"])
        if node is None or node["class_type"] not in object_info:
            continue
        definition = object_info.get(node["class_type"]) or {}
        declared = definition.get("input") if isinstance(definition.get("input"), dict) else {}
        known: set[str] = set()
        for group in ("required", "optional", "hidden"):
            group_inputs = declared.get(group)
            if isinstance(group_inputs, dict):
                known.update(group_inputs)
        if known and ref["input"] not in known:
            errors.append(_problem("INPUT_NOT_IN_NODE",
                                   f"{ROLE_SPECS[role]['label']}: this ComfyUI's {node['class_type']} has no input "
                                   f"'{ref['input']}' (it accepts: {', '.join(sorted(known)[:8])}).",
                                   role, ref["node_id"]))

    for entry in workflow_models(workflow):
        folder, name = entry["folder"], entry["name"]
        available = list((installed_models or {}).get(folder) or [])
        if folder and available and not any(name == item or item.endswith(name) or name.endswith(item)
                                            for item in available):
            warnings.append(_problem("MODEL_NOT_FOUND",
                                     f"Node {entry['node_id']} ({entry['class_type']}) wants '{name}' from "
                                     f"{folder}, which this ComfyUI does not list. Install that file or change the "
                                     "workflow; otherwise queueing will fail.", None, entry["node_id"]))
    return {"errors": errors, "warnings": warnings, "valid": not errors}


# --------------------------------------------------------------------------- patching

def apply_mapping(workflow: dict[str, dict[str, Any]], mapping: Any, values: dict[str, Any]) -> dict[str, Any]:
    """Return a patched copy of the workflow with the supplied role values written in."""
    patched = copy.deepcopy(workflow)
    mapped = dict(_mapping_items(mapping))
    for role, raw in values.items():
        if raw is None or role not in ROLE_SPECS:
            continue
        ref = mapped.get(role)
        if ref is None:
            raise WorkflowError("ROLE_UNMAPPED", f"Cannot set {ROLE_SPECS[role]['label'].lower()}: that role is "
                                                 "not mapped for this workflow.")
        node = patched.get(ref["node_id"])
        if node is None or ref["input"] not in node["inputs"]:
            raise WorkflowError("MAPPING_STALE", f"Mapping for {role} points at node {ref['node_id']} input "
                                                 f"'{ref['input']}', which no longer exists. Re-map the workflow.")
        node["inputs"][ref["input"]] = _coerce(role, raw)
    return patched


def _coerce(role: str, value: Any) -> Any:
    types = ROLE_SPECS[role]["types"]
    if "int" in types:
        try:
            number = int(value)
        except (TypeError, ValueError) as exc:
            raise WorkflowError("BAD_VALUE", f"{ROLE_SPECS[role]['label']} must be a whole number.") from exc
        if role == "seed" and not 0 <= number <= MAX_SEED:
            raise WorkflowError("BAD_VALUE", f"Seed must be between 0 and {MAX_SEED}.")
        if role in ("width", "height") and not MIN_DIMENSION <= number <= MAX_DIMENSION:
            raise WorkflowError("BAD_VALUE", f"{ROLE_SPECS[role]['label']} must be between {MIN_DIMENSION} and "
                                             f"{MAX_DIMENSION} pixels.")
        return number
    text = str(value)
    if len(text) > 20_000:
        raise WorkflowError("BAD_VALUE", f"{ROLE_SPECS[role]['label']} is too long (limit 20000 characters).")
    return text


def role_specs() -> list[dict[str, Any]]:
    return [{"role": role, **{k: v for k, v in spec.items()}} for role, spec in ROLE_SPECS.items()]
