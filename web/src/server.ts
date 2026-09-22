import { readFileSync } from "node:fs"
import path from "node:path"
import { fileURLToPath } from "node:url"

import { serve } from "@hono/node-server"
import { Hono } from "hono"

const apiBase = (process.env.ROOMIND_API ?? "http://127.0.0.1:8000").replace(/\/$/, "")
const port = Number(process.env.PORT ?? 8080)
const publicDir = path.join(path.dirname(fileURLToPath(import.meta.url)), "../public")
const tmpDir = path.resolve(publicDir, "../../test/tmp")
const artifactName = /^[A-Za-z0-9][A-Za-z0-9._-]*$/
const artifactType: Record<string, string> = {
  ".jpg": "image/jpeg",
  ".jpeg": "image/jpeg",
  ".png": "image/png",
  ".json": "application/json",
}

const app = new Hono()

app.get("/health", async (c) => {
  try {
    const upstream = await fetch(`${apiBase}/openapi.json`, {
      signal: AbortSignal.timeout(2000),
    })
    return c.json({ web: "ok", perception: upstream.ok ? "ok" : "down", api: apiBase })
  } catch {
    return c.json({ web: "ok", perception: "down", api: apiBase })
  }
})

app.get("/api/artifacts/:name", (c) => {
  const name = c.req.param("name")
  if (!artifactName.test(name)) {
    return c.json({ detail: "invalid artifact name" }, 400)
  }
  const filePath = path.resolve(tmpDir, name)
  if (path.dirname(filePath) !== tmpDir) {
    return c.json({ detail: "invalid artifact name" }, 400)
  }
  const type = artifactType[path.extname(filePath).toLowerCase()]
  if (!type) return c.json({ detail: "artifact not found" }, 404)
  try {
    const body = readFileSync(filePath)
    c.header("content-type", type)
    c.header("cache-control", "no-store")
    return c.body(body)
  } catch {
    return c.json({ detail: "artifact not found" }, 404)
  }
})

app.all("/api/*", async (c) => {
  const incoming = new URL(c.req.url)
  const target = `${apiBase}${incoming.pathname}${incoming.search}`
  const headers = new Headers()
  const contentType = c.req.header("content-type")
  if (contentType) headers.set("content-type", contentType)
  const hasBody = c.req.method !== "GET" && c.req.method !== "HEAD"
  try {
    const upstream = await fetch(target, {
      method: c.req.method,
      headers,
      body: hasBody ? c.req.raw.body : undefined,
      duplex: hasBody ? "half" : undefined,
      signal: c.req.raw.signal,
    } as RequestInit)
    const out = new Headers()
    for (const name of ["content-type", "cache-control"]) {
      const value = upstream.headers.get(name)
      if (value) out.set(name, value)
    }
    return new Response(upstream.body, { status: upstream.status, headers: out })
  } catch {
    return c.json({ detail: `感知服务不可达：${apiBase}` }, 502)
  }
})

function file(name: string, type: string) {
  return (c: { header: (k: string, v: string) => void; body: (b: Buffer) => Response }) => {
    c.header("content-type", type)
    c.header("cache-control", "no-cache")
    return c.body(readFileSync(path.join(publicDir, name)))
  }
}

app.get("/", file("index.html", "text/html; charset=utf-8"))
app.get("/styles.css", file("styles.css", "text/css; charset=utf-8"))
app.get("/app.js", file("app.js", "text/javascript; charset=utf-8"))

serve({ fetch: app.fetch, port }, (info) => {
  console.log(`RoomMind web http://127.0.0.1:${info.port} -> ${apiBase}`)
})
