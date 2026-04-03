# Public Deployment

## Target

- OS: Ubuntu 22.04 LTS
- Reverse proxy: Nginx
- App server: Gunicorn
- Frontend: static files from `dashboard/frontend/dist`
- Backend: Flask app from `dashboard/backend/wsgi.py`
- Database: SQLite

## Constraints

- Deploy as a single backend instance.
- Keep Gunicorn worker count at `1`.
- Public deployment keeps telemetry in `replay` mode only.
- PX4 / blind-test are disabled in production.

## Suggested layout

```text
/srv/uav-dashboard/
  dashboard/
    backend/
    frontend/
  models/
  knowledge_base/
  runtime/
```

## Backend env file

Create `/etc/uav-dashboard.env`:

```env
APP_ENV=production
FLASK_DEBUG=False
SECRET_KEY=replace-with-random-secret
JWT_SECRET_KEY=replace-with-random-jwt-secret
CORS_ORIGINS=https://your-domain.example

DB_TYPE=sqlite
DB_PATH=/srv/uav-dashboard/runtime/dashboard.db
RUNTIME_DIR=/srv/uav-dashboard/runtime
UPLOAD_DIR=/srv/uav-dashboard/runtime/uploads
DERIVED_DIR=/srv/uav-dashboard/runtime/derived
RUNS_DIR=/srv/uav-dashboard/runtime/runs
REPORTS_DIR=/srv/uav-dashboard/runtime/reports
AI_RUNTIME_DIR=/srv/uav-dashboard/runtime/ai

MODEL_ROOT=/srv/uav-dashboard/models
OFFLINE_NORM_STATS=/srv/uav-dashboard/models/transfer3_cnnlstm_normalproto/norm_stats.json
DEFAULT_OFFLINE_MODEL=transfer3_cnnlstm_normalproto
DEFAULT_ONLINE_MODEL=online_detector_cnn1d_rfly

ALLOW_AUTO_BOOTSTRAP_ADMIN=False
ENABLE_SETUP_API=False
ENABLE_PUBLIC_REGISTER=False
ENABLE_TELEMETRY=True
ENABLE_BLIND_TEST=False
TELEMETRY_ALLOWED_BACKENDS=replay

ENABLE_AI=False
AI_ENABLE_KNOWLEDGE_BASE=False
EXPOSE_SERVER_PATHS=False
TASK_MAX_WORKERS=2
```

## Build frontend

```bash
cd /srv/uav-dashboard/dashboard/frontend
npm ci
npm run build
```

## Install backend

```bash
cd /srv/uav-dashboard
python3 -m venv .venv
.venv/bin/pip install -r dashboard/backend/requirements.txt
```

## Start backend

Use the provided systemd unit in `deploy/systemd/uav-dashboard.service`.

## Nginx

Use the provided site config in `deploy/nginx/uav-dashboard.conf`.

## Notes

- If AI is enabled later, the server must have outbound access to the configured provider.
- If telemetry replay is enabled, uploaded CSV files and generated session CSVs must be stored on persistent disk.
- Do not enable multi-worker Gunicorn until task execution and telemetry state are moved out of process memory.
