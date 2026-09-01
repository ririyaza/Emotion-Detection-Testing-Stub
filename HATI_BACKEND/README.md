---
title: Hati Backend
emoji: 🐸
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
---

# HATI Backend

Flask API for the HATI app: scenario dialogue FSM (`scenario_engine.py`),
text/audio emotion classification, and Whisper-based speech-to-text.

## Required Space secrets

Set these under Settings → Variables and secrets on this Space (never commit
them to the repo):

- `FIREBASE_CREDENTIALS_JSON` — the full contents of your Firebase service
  account JSON key, pasted as one secret value.
- `FIREBASE_PROJECT_ID` — e.g. `hati-25259`.
