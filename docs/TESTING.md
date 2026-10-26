# Testing guide

```bash
conda activate smart-glasses
pytest                         # all
pytest -m ui                   # visual HUD + voice dialogue
pytest -m perf                 # performance budgets
pytest -m "not integration"    # no sample video / model files needed
pytest --cov=smart_glasses --cov-report=term-missing
python scripts/benchmark.py    # print real latency numbers
```

## Markers

| Marker | Meaning |
|---|---|
| `ui` | Rendered-pixel tests of the HUD and scripted spoken-dialogue tests |
| `perf` | Latency budgets (≈10× headroom to stay stable on slow machines) |
| `integration` | Uses `assets/demo/sample.mov` and/or `model_data/`; the real-YOLO test skips itself when weights are absent |

## How the UI is tested without a screen, camera or microphone

* **Visual UI** — `hud.render` returns a numpy image. Tests assert on pixels: output shape, amber outline on spoken objects, white radar dots on the correct half, divider line, changed status bar, no mutation of the input frame.
* **Voice UI** — `FakeListener` plays a script of utterances, `FakeCamera` supplies frames, `ScriptedDisplay` injects key presses (Esc/Space), and `FakeSpeaker` records which recorded prompts would have been played. A test then reads like a conversation:

```python
a, _ = build(cfg, prompts, utterances=[CMD_PEOPLE, CMD_NEW, "ليلى", "0"], faces=FakeFaces())
a.run()
assert speaker.names == ["commands/intro", "commands/f_r", "commands/f_r1", "commands/end"]
```

A scripted listener falls back to `"0"` (exit) when its script ends, so a wrong test can fail but never hang.

## Regression tests for former bugs

Each bug listed in [CHANGELOG.md](CHANGELOG.md) has a test (`test_largest_uses_real_area…`, `test_side_scales_with_frame_width`, `test_failure_sound_not_played_after_a_success…`, `test_announcer_rate_limits`, `test_new_person_not_stored_when_no_face_was_captured`, …).

## Not covered (needs hardware)

`GoogleListener` (microphone + network), `PlaysoundSpeaker`, `CvDisplay` windows, real YOLO accuracy and Arabic OCR accuracy. Verify those by hand with `smart-glasses --camera assets/demo/sample.mov`.
