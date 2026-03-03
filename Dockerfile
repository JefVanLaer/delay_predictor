# AIS Stream Mock
#
# Replays AIS data from a Parquet file as newline-delimited JSON on stdout.
#
# Build:
#   docker build -t ais-stream-mock .
#
# Run (mount the directory containing your Parquet file at /data):
#   docker run --rm \
#     -v /path/to/your/data:/data \
#     ais-stream-mock --parquet /data/ais-2025-01-01.parquet --speed-factor 60
#
# The container writes one JSON object per line to stdout until the file
# is exhausted.

FROM python:3.12-slim

WORKDIR /app

# libgomp1 is required by pyarrow on Debian/Ubuntu slim images
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/*

# Install only what the mock streamer needs
RUN pip install --no-cache-dir pandas pyarrow

# Copy the source package
COPY src/ ./src/

# Mount point for Parquet data files (parquet files are git-ignored and
# must be supplied at runtime via a volume or bind mount)
VOLUME ["/data"]

ENTRYPOINT ["python", "src/ais_stream_mock.py"]
CMD ["--parquet", "/data/ais.parquet"]
