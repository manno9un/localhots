#!/usr/bin/env bash
# (Re)start the localhots website on the LinuxONE server.
# Runs app/server.py inside IBM's pinned s390x JupyterLab image (has numpy/pandas/scikit-learn),
# with the model project and the app both mounted read-only. Public on port 80, plus 127.0.0.1:8080 for SSH tunnels.
set -euo pipefail

IMAGE=registry.linuxone.cloud.marist.edu/l1cc/jupyterlab-image-s390x@sha256:d2da8a7b0c88c3b468e29014b2ada9ef8274f98c976ec456904c5e2bae5aa54c
MODEL_DIR=/home/linux1/wildfire-outlook
APP_DIR=/home/linux1/wildfire-site/app

sudo docker rm -f wildfire-outlook >/dev/null 2>&1 || true
sudo docker run -d --name wildfire-outlook --restart unless-stopped \
  --cpus 3 --memory 2g \
  -e OMP_NUM_THREADS=3 -e OPENBLAS_NUM_THREADS=3 -e PYTHONDONTWRITEBYTECODE=1 \
  -v "$MODEL_DIR":/work:ro -v "$APP_DIR":/app:ro \
  -p 80:8080 -p 127.0.0.1:8080:8080 \
  --entrypoint /opt/venv/bin/python \
  "$IMAGE" -u /app/server.py

echo "Waiting for the model to load..."
for _ in $(seq 1 60); do
  if curl -sf http://127.0.0.1:8080/api/health >/dev/null; then
    echo "Up: http://127.0.0.1:8080"
    exit 0
  fi
  sleep 2
done
sudo docker logs --tail 40 wildfire-outlook
exit 1
