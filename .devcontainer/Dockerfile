FROM mcr.microsoft.com/devcontainers/python:1-3.12

# Remove Yarn's APT repo — its signing key has expired and breaks
# apt-get update for every feature installed afterward. Not needed:
# pnpm is handled via corepack by the Node feature instead.
RUN rm -f /etc/apt/sources.list.d/yarn.list