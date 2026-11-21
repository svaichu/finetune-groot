# Remote GR00T inference for ElRobot

Two machines, one job each. Nothing crosses the network except the `get_action` call.

## Cloud server (this box)

Runs the model. Nothing robot-specific.

1. Get a checkpoint: base `nvidia/GR00T-N1.7-3B` from HF, or a fine-tuned checkpoint for ElRobot's embodiment (copy/download it here).
2. Start the policy server:
   ```bash
   cd vendor/Isaac-GR00T
   python gr00t/eval/run_gr00t_server.py \
     --model-path <checkpoint-or-hf-repo> \
     --embodiment-tag <tag> \
     --device cuda \
     --host 0.0.0.0 --port 5555
   ```
3. Expose port 5555 to the local client (tunnel — Tailscale / SSH reverse tunnel / ngrok; pick one).

That's it. This box never touches cameras, motors, or Station.

## Local client (machine wired to ElRobot, running Station)

Runs Station as today. Adds one bridge script in place of `run_policy.py`'s in-process SmolVLA call.

Per tick:
1. `read_from_tail("inference/normvla")` → `Frame` (joints + 2 JPEGs) — unchanged.
2. Build obs dict: joints' `position_norm` → state vector, JPEGs → image keys, task string → instruction.
3. `PolicyClient(host=<tunnel-addr>, port=5555).call_endpoint("get_action", obs)` — the one network call, replacing local `policy.predict_action_chunk()`.
4. Unnormalize chunk[0], clip to each joint's `(range_min, range_max)`.
5. Build sync-write `DriverCommand`, `send_commands(...)` — unchanged.

Dependencies: `station_py`, generated protobufs, `zmq`, `msgpack_numpy`. No torch/transformers needed here.

## Open item

Need a GR00T checkpoint trained for ElRobot's joint layout + 2×224×224 cameras (`normvla.Frame`'s shape). If none exists yet: collect episodes via Station's dataset tooling, convert for `robotdataset`, fine-tune with a new `embodiment_tag` first.

## gr3-humanoid sim env

Same server, different client: `gr3-humanoid/run_vla.py` (closed loop in the robotenv GR-3 pick-up scene, using
`vla_gr00t.py`'s `PolicyClient`). No GR-3-specific server flags are needed: `Gr00tPolicy` takes the modality config from the
checkpoint's processor, so a checkpoint fine-tuned with `finetune_groot/gr3_embodiment.py` (right arm 7 + gripper 2, absolute
joint targets, cameras `front`/`agent`) already carries it. Keys must match `gr3-humanoid/gr3_embodiment.py`; the client
checks this at start-up.

```bash
bash remote-inference/serve.sh --model-path <fine-tuned ckpt> --embodiment-tag new_embodiment
python run_vla.py --host localhost --port 5555      # in gr3-humanoid; --dry-run tests the env side without a server
```

The base `nvidia/GR00T-N1.7-3B` has no NEW_EMBODIMENT entry, so the server rejects that tag with it; train on GR-3 demos first
(`configs/finetune_gr3_pickup.yaml`, `configs/modalities/gr3_pickup.modality.yaml`).
