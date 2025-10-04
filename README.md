![Static Badge](https://img.shields.io/badge/Made_with-Python-blue?style=flat&link=https%3A%2F%2Fwww.python.org%2F)
# Audio Distortion Report
A Python script for Multi-channel, multithreaded distortion report between a reference ("before") and a processed ("after") audio file.

Per-channel it computes:
  - Alignment (shared shift determined on mono mix for stability)
  - Gain matching
  - Residual & SDR
  - Spectral FFT
  - THD / SINAD / ENOB (if single-tone; or with --f0 hint)

Also computes an overall mono summary.

## Installation
Install libraries in Python:
```bash
pip install numpy scipy soundfile matplotlib soxr
```

## Usage
```python
# Multi-channel report (auto detection of dominant tone)
python audio_distortion_report_mc.py "before.wav" "after.wav" --out report_mc

# Force the fundamental (useful if the signal is not a pure dominant tone)
python audio_distortion_report_mc.py "before.wav" "after.flac" --out report_mc --f0 1000

# Limit the X-axis of the spectra (zoom) and adjust the number of workers
python audio_distortion_report_mc.py "before.wav" "after.wav" --out report_mc --fft_xlim 5000 --workers 8
```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[GPL-03](https://choosealicense.com/licenses/gpl-3.0/)
