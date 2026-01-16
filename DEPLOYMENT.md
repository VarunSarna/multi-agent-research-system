# 🚀 Deployment Guide - Multi-Agent Research System

## Step 1: Get Your API Keys

### Anthropic API Key (for Claude LLM)

1. Go to: **https://console.anthropic.com/**
2. Sign up or log in
3. Click **"Get API Keys"** in the left sidebar
4. Click **"Create Key"**
5. Copy the key (starts with `sk-ant-...`)
6. **Cost**: Pay-as-you-go, ~$0.003 per 1K input tokens, ~$0.015 per 1K output tokens
7. **Free credits**: New accounts often get $5 free credits

### Tavily API Key (for Web Search)

1. Go to: **https://tavily.com/**
2. Click **"Get Started"** or **"Sign Up"**
3. Verify your email
4. Go to Dashboard → API Keys
5. Copy your API key (starts with `tvly-...`)
6. **Free tier**: 1,000 searches/month FREE! 🎉

---

## Step 2: Push to GitHub

### Option A: Using GitHub Web Interface

1. Go to **https://github.com/new**
2. Name: `multi-agent-research-system`
3. Keep it **Public** (required for free Streamlit Cloud)
4. Click **"Create repository"**
5. Click **"uploading an existing file"**
6. Drag and drop ALL these files:
   - `app.py`
   - `requirements.txt`
   - `README.md`
   - `.gitignore`
   - `.streamlit/config.toml`
   - `secrets.toml.example`
7. Click **"Commit changes"**

### Option B: Using Git Command Line

```bash
# 1. Create a new folder and copy files there
mkdir multi-agent-research-system
cd multi-agent-research-system

# 2. Copy all downloaded files here

# 3. Initialize git
git init
git add .
git commit -m "Initial commit: Multi-agent research system"

# 4. Create repo on GitHub, then:
git remote add origin https://github.com/YOUR_USERNAME/multi-agent-research-system.git
git branch -M main
git push -u origin main
```

---

## Step 3: Deploy to Streamlit Cloud

1. Go to: **https://share.streamlit.io/**

2. Click **"New app"**

3. Connect your GitHub if not already done

4. Select:
   - **Repository**: `YOUR_USERNAME/multi-agent-research-system`
   - **Branch**: `main`
   - **Main file path**: `app.py`

5. Click **"Advanced settings"** → **"Secrets"**

6. Paste this (with YOUR real keys):
   ```toml
   ANTHROPIC_API_KEY = "sk-ant-your-actual-key-here"
   TAVILY_API_KEY = "tvly-your-actual-key-here"
   ```

7. Click **"Deploy!"**

8. Wait 2-3 minutes for deployment

9. 🎉 Your app is live at: `https://your-app-name.streamlit.app`

---

## Step 4: Share in Your Interview

Your live URL will look like:
```
https://multi-agent-research-system-yourusername.streamlit.app
```

**Pro tips for the interview:**
- Open the app before the interview to "warm it up"
- Have a few example queries ready
- Show the agent logs to demonstrate observability
- Mention you can switch between demo and live mode

---

## Troubleshooting

### "Module not found" error
→ Check `requirements.txt` is in the repo root

### "API key not found" error
→ Check Streamlit Cloud Secrets are set correctly

### App shows "Demo Mode" even with keys
→ Verify your secrets are named exactly:
  - `ANTHROPIC_API_KEY`
  - `TAVILY_API_KEY`

### App crashes on startup
→ Check the Streamlit Cloud logs (click "Manage app" → "Logs")

---

## Cost Estimates

| Usage | Anthropic Cost | Tavily Cost |
|-------|---------------|-------------|
| 10 queries/day | ~$0.30/day | FREE (1000/month) |
| Demo in interview | ~$0.05 | FREE |
| Heavy testing | ~$1-2/day | FREE |

**Tip**: The demo mode works without any API keys, so you can always fall back to that!
