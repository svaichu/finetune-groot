# Local client architecture (minimal env)

Runs on the machine wired to ElRobot (where Station already runs). No torch, no
transformers, no CUDA — those stay on the cloud server.

## Why the real `PolicyClient` can't just be `pip install`ed here

`gr00t.policy.server_client.PolicyClient` is pure ZMQ + msgpack — cheap on its
own. But importing it means importing `gr00t.policy`, and `gr00t/policy/__init__.py`
eagerly does `from .gr00t_policy import Gr00tPolicy`, which imports `torch` and
`transformers` at module scope. There's no way to import the client without
paying for the whole model stack.

**Fix: lift, don't install.** Copy the self-contained pieces out of
`vendor/Isaac-GR00T/gr00t/policy/server_client.py` into a standalone file here —
`MsgSerializer` + `PolicyClient`, and the one dataclass they touch
(`ModalityConfig`, trivial — `dataclasses` + `numpy`, no `gr00t` package import).
Same wire protocol as the real server, zero heavy deps.

## Dependencies (everything here is small)

| package | why |
|---|---|
| `pyzmq` | ZMQ REQ socket to the cloud server |
| `msgpack` | wire format |
| `msgpack-numpy` | numpy array (de)serialization in the wire format |
| `numpy` | build obs arrays, decode action arrays |
| `Pillow` | decode `Frame`'s JPEG bytes → `uint8` array (GR00T wants decoded images, not JPEG) |

Station's own client stack (`station_py`, generated protobuf code under
`target/gen_python`) adds **nothing** — it's pure stdlib (`asyncio`, `struct`,
`socket`, `dataclasses`). Gremlin is a custom wire format, not `protobuf`'s
Python runtime, so there's no extra pip package for it either.

Nothing from `vendor/Isaac-GR00T/pyproject.toml` (torch 2.7.1, transformers,
flash-attn, torchvision, torchcodec, wandb, ...) is needed here.

## Files on the local client

```
remote-inference/
  policy_client.py   # lifted MsgSerializer + PolicyClient + ModalityConfig (no gr00t import)
  bridge.py           # the control loop (replaces run_policy.py's in-process SmolVLA call)
```

## `bridge.py` loop

```python
from policy_client import PolicyClient
from station_py import new_station_client, send_commands
from target.gen_python.protobuf.drivers.inferences import normvla
from target.gen_python.protobuf.station import commands

client = new_station_client(STATION_HOST)
policy = PolicyClient(host=TUNNEL_HOST, port=5555)
assert policy.ping()

while True:
    frame = fetch_frame(client)                      # Frame: joints + 2 JPEGs

    obs = {
        "video": {cam_key: decode_jpeg_to_uint8_array(img) for ...},   # (1,1,H,W,3) uint8
        "state": {state_key: np.array([[...position_norm...]], dtype=np.float32)},
        "language": {"task": [[task_str]]},
    }

    action, _ = policy.call_endpoint("get_action", obs)   # the one network hop
    goal_norm = action[action_key][0, 0]                  # first step of the chunk

    raw_goals = [unnormalize(g, lo, hi) for g, (lo, hi) in zip(goal_norm, ranges)]
    cmd = build_sync_write_command(bus_serial, motor_ids, raw_goals)
    send_commands(client, cmd)
```

Same shape as `smolvla_py/scripts/run_policy.py` — only the "predict" step
changed, from an in-process `policy.predict_action_chunk()` to one ZMQ round
trip (`policy.call_endpoint("get_action", obs)`). Same clip/safety checks
(`max-delta-ticks`) apply before sending.

## Observation key names

`video`/`state` keys must match whatever the checkpoint's modality config
calls them (`gr00t.get_modality_config()` on the server tells you — call it
once at startup via `policy.call_endpoint("get_modality_config", requires_input=False)`
rather than hardcoding names).
