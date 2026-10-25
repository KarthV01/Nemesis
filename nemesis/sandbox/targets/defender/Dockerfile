# DEFENSIVE monitoring sidecar for the isolated lab. Build locally.
#
#   docker build -t nemesis-sbx/defender nemesis/sandbox/targets/defender
#
# stdlib only -> no pip install, no registry access needed at runtime.
FROM python:3.12.7-slim

RUN useradd --create-home --uid 10002 defender
USER defender
WORKDIR /home/defender

COPY monitor.py ./monitor.py

EXPOSE 9090

# Healthy whenever the monitor's own HTTP endpoint answers.
HEALTHCHECK --interval=10s --timeout=3s --start-period=5s --retries=3 \
    CMD ["python", "-c", "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:9090/health', timeout=2).status==200 else 1)"]

CMD ["python", "-u", "monitor.py"]
