FROM mcr.microsoft.com/devcontainers/python:1-3.12

# Remove Yarn's APT repo — its signing key has expired and breaks
# apt-get update for every feature installed afterward. Not needed:
# pnpm is handled via corepack by the Node feature instead.
RUN rm -f /etc/apt/sources.list.d/yarn.list

# Install PostgreSQL client (has to be version 16)
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl ca-certificates gnupg \
    && curl -fsSL https://www.postgresql.org/media/keys/ACCC4CF8.asc \
        | gpg --dearmor -o /usr/share/keyrings/postgresql.gpg \
    && echo "deb [signed-by=/usr/share/keyrings/postgresql.gpg] http://apt.postgresql.org/pub/repos/apt bookworm-pgdg main" \
        > /etc/apt/sources.list.d/pgdg.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends postgresql-client-16 \
    && rm -rf /var/lib/apt/lists/*