# PeatLearn frontend — builds the Vite/React SPA to static files, then serves
# them with Caddy. Caddy also reverse-proxies /api to the api container and
# terminates TLS (automatic HTTPS) in production. See Caddyfile.
#
# Build:  docker build -t peatlearn-web ./web
# (usually built via docker-compose alongside the api service)

# --- Stage 1: build the static bundle ---
FROM node:24-alpine AS build
WORKDIR /app
COPY package.json package-lock.json* ./
RUN npm install
COPY . .
RUN npm run build

# --- Stage 2: serve with Caddy ---
FROM caddy:2-alpine
COPY Caddyfile /etc/caddy/Caddyfile
COPY --from=build /app/dist /srv
EXPOSE 80 443
