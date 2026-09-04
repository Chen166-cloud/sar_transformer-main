FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /workspace

COPY docker/requirements.txt /tmp/sar-requirements.txt

RUN python -m pip install --upgrade "pip<25" && \
    python -m pip install -r /tmp/sar-requirements.txt && \
    python -m pip install --no-deps mmcv==1.7.2 && \
    python -m pip check

CMD ["bash"]
