# Deployment Guide - Multi-Agent Research System

This guide explains how to run or deploy the multi-agent research demo locally or on Streamlit Cloud.

## Step 1: API keys

### Anthropic API key

1. Go to `https://console.anthropic.com/`
2. Create an API key
3. Store it as `ANTHROPIC_API_KEY`

### Tavily API key

1. Go to `https://tavily.com/`
2. Create an API key
3. Store it as `TAVILY_API_KEY`

The app also runs in demo mode without API keys.

---

## Step 2: Local run

```bash
pip install -r requirements.txt
streamlit run app.py
```

Open:

```text
http://localhost:8501
```

---

## Step 3: Live mode with environment variables

```bash
export ANTHROPIC_API_KEY="sk-ant-your-key"
export TAVILY_API_KEY="tvly-your-key"
streamlit run app.py
```

---

## Step 4: Streamlit Cloud deployment

1. Go to `https://share.streamlit.io/`
2. Create a new app
3. Select this repository
4. Set main file path to `app.py`
5. Add secrets:

```toml
ANTHROPIC_API_KEY = "sk-ant-your-key"
TAVILY_API_KEY = "tvly-your-key"
```

6. Deploy

---

## Troubleshooting

### Module not found

Check that `requirements.txt` is in the repository root and Streamlit Cloud installed dependencies successfully.

### API key not found

Check that secrets are named exactly:

```text
ANTHROPIC_API_KEY
TAVILY_API_KEY
```

### App stays in demo mode

Verify secrets/environment variables are configured and restart the app.

### App crashes on startup

Check Streamlit logs and verify dependency versions.

---

## Cost notes

Live mode uses paid model/search APIs. Demo mode is available for local testing without external API calls.

---

## Production hardening notes

For a production deployment, add:

- authentication
- persistent state store
- request limits
- cost tracking
- retry/backoff
- structured logging
- OpenTelemetry traces
- eval checks before final report generation
- deployment-specific secrets management
