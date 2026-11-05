FROM python:3.10-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/opt/nemesis/external/cage-challenge-4

WORKDIR /opt/nemesis

RUN apt-get update \
    && apt-get install --no-install-recommends -y git build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY external/cage-challenge-4/Requirements.txt /tmp/cc4-requirements.txt
RUN pip install --no-cache-dir \
        torch==2.2.0 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r /tmp/cc4-requirements.txt

COPY external/cage-challenge-4 /opt/nemesis/external/cage-challenge-4
RUN pip install --no-cache-dir \
        -e /opt/nemesis/external/cage-challenge-4 \
        --config-settings editable_mode=compat

COPY nemesis /opt/nemesis/nemesis
COPY modal_app /opt/nemesis/modal_app
COPY scripts /opt/nemesis/scripts
COPY web /opt/nemesis/web

RUN groupadd --gid 10001 nemesis \
    && useradd --uid 10001 --gid nemesis --create-home nemesis \
    && mkdir -p /artifacts \
    && chown nemesis:nemesis /artifacts

USER nemesis

ENTRYPOINT ["python"]
CMD ["scripts/train_selfplay.py", "--artifacts", "/artifacts/selfplay"]
