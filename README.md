# FinSight Frontend (Minimal Template)

## Quick start
```bash
# 1) In this folder:
pnpm i

# 2) set backend endpoint (optional; default already localhost):
echo 'VITE_API_BASE="http://localhost:8000/api/v1"' > .env.local

# 3) run
pnpm dev

# 4) quick restart
rm -rf node_modules pnpm-lock.yaml package-lock.json # optional clean reboot
pnpm i 
pnpm dev 
```



## Pages
- `/` Dashboard
- `/news` News browser (GET /news)



