# INTENTIONALLY VULNERABLE lab web app. Isolated, authorized use ONLY.
# Build locally; it is meant to run only on an --internal (no-internet) network.
#
#   docker build -t nemesis-sbx/vulnerable-web nemesis/sandbox/targets/vulnerable_web
#
# stdlib only -> no pip install, no registry access needed at runtime.
FROM python:3.12.7-slim

# Drop root: the app binds :8080 (an unprivileged port), so it needs no root.
RUN useradd --create-home --uid 10001 labuser
USER labuser
WORKDIR /home/labuser

COPY app.py ./app.py

EXPOSE 8080

# HEALTHCHECK powers both `docker inspect`-based health and the sandbox
# `service_health` probe. Uses stdlib so the slim image needs no curl.
HEALTHCHECK --interval=10s --timeout=3s --start-period=5s --retries=3 \
    CMD ["python", "-c", "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8080/health', timeout=2).status==200 else 1)"]

CMD ["python", "-u", "app.py"]
