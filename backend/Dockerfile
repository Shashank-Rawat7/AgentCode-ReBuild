FROM python:3.12-slim

# Preserve the same relative depth as the repo (repo_root/backend/...), because
# backend/utils/file_ops.py computes its workspace path as two directories above
# itself. Here that's /app/backend/utils/file_ops.py, so WORKSPACE resolves to
# /app/codebase -- matching the eventual named-volume mount point.
WORKDIR /app/backend

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
