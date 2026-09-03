# Local dev (docker-compose): deps only; nextjs/ mounted at /usr/src/app.
FROM node:25-alpine AS base

RUN apk add --no-cache openssl

WORKDIR /usr/src/app

COPY ./nextjs/package.json ./nextjs/yarn.lock ./
RUN yarn install --frozen-lockfile


FROM node:25-alpine AS builder

RUN apk add --no-cache openssl

WORKDIR /app

COPY ./nextjs/package.json ./nextjs/yarn.lock ./
RUN yarn install --frozen-lockfile

COPY ./nextjs .
RUN yarn run lint && yarn run build


FROM node:25-alpine AS production

RUN apk add --no-cache openssl

WORKDIR /app
ENV NODE_ENV=production
ENV HOSTNAME=0.0.0.0
ENV PORT=3000

COPY --from=builder /app/.next/standalone ./
COPY --from=builder /app/.next/static ./.next/static
COPY --from=builder /app/public ./public

EXPOSE 3000
CMD ["node", "server.js"]
