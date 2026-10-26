# Usage

## Setup

```bash
conda env create -f environment.yml
conda activate smart-glasses
python scripts/download_models.py          # add --tiny for a smaller, faster, less accurate model
pip install ArabicOcr                      # only for text mode; it pulls in PyTorch
```

If the download script fails (some hosts block scripted downloads), save `yolov3.weights` and `yolov3.cfg` into `model_data/` by hand.

**Linux:** `playsound` needs GStreamer: `sudo apt install gir1.2-gstreamer-1.0 python3-gst-1.0`. PortAudio for the microphone comes with the conda env.

## Running

```bash
smart-glasses                          # webcam 0
smart-glasses --camera 1               # another camera
smart-glasses --camera clip.mp4        # a video file
smart-glasses --headless -v            # no windows, verbose logs
```

## Voice commands

| Say | Then |
|---|---|
| اغراض | Live detection. Press **Esc** to go back. |
| اشخاص → جديد | Say the person's name and look at the camera while it captures 20 face samples. |
| اشخاص → قديم | Look at the camera and it announces the name. |
| نصوص | Press **Space** to read the current frame, **Esc** to go back. |
| 0 | Exit |

Anything else gets a "please repeat" prompt.

## Your data

Everything personal is kept in `data/`, which git ignores: `people.csv`, `names/`, `dataset/` (face crops) and `recognizer.yml` (the trained model).

## Settings

Thresholds and paths are all in [`smart_glasses/config.py`](../smart_glasses/config.py), for example the announcement interval, how many objects to announce, or how strict face matching is.

## Limitations

- No accuracy numbers are claimed. Quality depends on the YOLO model and on ArabicOcr.
- Only left and right, no "ahead" or distance. More directions need new recorded audio.
- Speech recognition uses Google's web API, so voice input needs internet. So does speaking a new person's name.
- Face recognition is classic LBPH. Fine for a few known people, not for security.

## Ideas

Offline speech recognition (Vosk or Whisper), distance estimation, an "ahead" direction, currency and colour recognition, an ARM/Jetson build.
