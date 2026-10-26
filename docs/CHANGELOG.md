# Changelog

## 1.0.0 — restructure, hardening, tests, docs

### Repo clean-up
* Removed committed `__pycache__/` (py3.5/3.7/3.9 bytecode), `.pytest_cache/`, `tempCodeRunnerFile.py`, a duplicate 8 MB video, stray `out.jpg`/`ss.mp3`, the unused Keras `yolo3/` package, an unused font, and two scripts hard-coded to `D:/WORK/...` (`test.py`, `yolo_test.py`).
* Replaced the ad-hoc `Blocks/S_G.py` + `interface1.py` + `save_audio.py` with the `smart_glasses/` package and `scripts/`.
* Moved per-user runtime state (`Data.csv`, `names/`, trained `recognizer/trainingData.yml`) out of version control into a git-ignored `data/` folder (files kept on disk; still in git history).
* Audio assets reorganised under `assets/audio/{objects,directions,ocr,commands}`; added `assets/coco_classes.txt`.

### Bugs fixed
| Before | After |
|---|---|
| Announcements gated by `elapsed > 1000.0`, i.e. practically never spoke | Configurable `announce_interval_s` (3 s) via `Announcer` |
| Area computed as `(w-x)*(h-y)` → wrong "largest objects" | `w*h` |
| Left/right split hard-coded at x=320 | Half of the real frame width |
| OCR played the "nothing found" sound after a hit on the last line; loop variable shadowed | `find_keyword` returns the first match; failure only if none |
| `recognize_google` exceptions (unintelligible speech / offline) crashed the app | Treated as "not understood" → "please repeat" |
| New person ID = `len(rows)` (undefined if no CSV, reused after deletions) | `max(ID)+1` |
| Person stored even if no face was captured; registration could loop forever | Stored only after samples exist; time-outs on every camera loop |
| Recognition returned the first matching vote, never checked LBPH distance | Majority of frames **and** distance threshold → strangers rejected |
| First detected face used (`faces[0]`) | Largest face |
| `cv2.CAP_DSHOW` (Windows only) | Used only on Windows |
| Every window (`positions`, `Image`, `full`) opened separately | One combined HUD |
| `sys()` shadowed the `sys` module; stray `cProfile`/`tracemalloc` imports; dead commented code | Gone |
| Intro prompt was commented out | Played on start |
| Push-to-talk required the `keyboard` package (root on Linux) | Listens directly; no root needed |

### Added
* Dependency-injected `Assistant`, `Config` dataclass, lazy imports of heavy/optional dependencies.
* 76 tests (unit, UI, integration, performance), 90 % coverage, ruff config, GitHub Actions CI.
* `environment.yml`, `pyproject.toml`, `smart-glasses` command, `scripts/benchmark.py`, `scripts/download_models.py`.
* README, architecture and testing docs.
