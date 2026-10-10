# Put Feelable online (free)

**Recommended: Render.** It's free and builds straight from this GitHub repo, redeploying on every push to `main`.

1. Sign in at https://render.com with **GitHub** (account TYK3).
2. Open https://render.com/deploy?repo=https://github.com/TYK3/feelable. It reads `render.yaml` and sets up one free Docker web service called `feelable`. Click **Deploy** (Render may ask to connect the repo first).
3. The first build takes about 5–10 minutes. The public link is `https://feelable.onrender.com` (or similar, shown on the service page).
4. Optional: in the service's **Environment** tab, set `GEMINI_API_KEY` for AI descriptions.

Notes:
- Free services sleep after about 15 minutes idle, and the first visit after that takes about 50 seconds to wake. Open the link a minute before you demo.
- The free plan has 512 MB of RAM. The Docker image sets `FEELABLE_ISOLATE=1`, so heavy jobs run in one warm worker process (recycled every 20 jobs); the peak measured ~480 MB across all samples. `FEELABLE_LAZY_STL=1` builds the 3D model only when it's first opened.
- Jobs are stored in `/tmp` and disappear when the service restarts. That's fine for a demo.

**Why not Hugging Face?** Docker Spaces now need a paid PRO plan; free Spaces are static only. If you have PRO, the same `Dockerfile` works there, and `.github/workflows/deploy-hf.yml` pushes to it once `HF_TOKEN` (secret) and `HF_SPACE` (variable) are set in the repo settings.

**For development:** use GitHub Codespaces (repo page → Code → Codespaces). It installs everything and starts the app on port 8000.
