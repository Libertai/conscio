FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /opt/conscio

RUN useradd --create-home --home-dir /home/conscio conscio

COPY pyproject.toml README.md ./
COPY src ./src
COPY docs ./docs

RUN pip install --no-cache-dir -e .

USER conscio
ENV HOME=/home/conscio

EXPOSE 8765
CMD ["conscio", "service", "start"]
