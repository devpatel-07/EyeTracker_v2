#!/bin/zsh
# Double-click in Finder. Resolve this script's folder even when its path has spaces.
cd -- "${0:A:h}" || exit 1
if [[ ! -x .venv/bin/python ]]; then
  print 'The project Python environment is missing. Restore .venv or install requirements first.'
  read -r '?Press Return to close.'
  exit 1
fi
print 'Opening Blink Review in your browser. Keep this terminal open while reviewing.'
.venv/bin/python -B blink_validation.py
status_code=$?
if (( status_code != 0 )); then
  print 'Blink Review stopped with an error. The explanation is above; your saved labels remain on disk.'
  read -r '?Press Return to close.'
fi
