import { readFileSync } from "node:fs"
import path from "node:path"
import { fileURLToPath } from "node:url"

import { serve } from "@hono/node-server"
import { Hono } from "hono"
import { Agent, fetch as undiciFetch } from "undici"

// 整条流水线要等 VLM 逐个描述，常常超过 Node fetch 默认的 5 分钟。
// 到点会报「感知服务不可达」，服务端其实还在跑。
const perceptionAgent = new Agent({
  connectTimeout: 10_000,
  headersTimeout: 0,
  bodyTimeout: 0,
})

const apiBase = (process.env.ROOMIND_API ?? "http://127.0.0.1:8000").replace(/\/$/, "")
const agentBase = (process.env.ROOMIND_AGENT ?? "http://127.0.0.1:8001").replace(/\/$/, "")
const port = Number(process.env.PORT ?? 8080)
const publicDir = path.join(path.dirname(fileURLToPath(import.meta.url)), "../public")
const tmpDir = path.resolve(publicDir, "../../tests/tmp")
const artifactName = /^[A-Za-z0-9][A-Za-z0-9._-]*$/
const artifactType: Record<string, string> = {
  ".jpg": "image/jpeg",
  ".jpeg": "image/jpeg",
  ".png": "image/png",
  ".json": "application/json",
}

const dev = process.env.NODE_ENV !== "production"
const app = new Hono()

if (dev) {
  app.use("*", async (c, next) => {
    const started = Date.now()
    await next()
    const path = c.req.path
    if (path === "/health" || path.startsWith("/api/scene/progress")) return
    if (!path.startsWith("/api/") && !path.startsWith("/agent-api/")) return
    console.log(`${c.req.method} ${path} ${c.res.status} ${Date.now() - started}ms`)
  })
}

async function reach(url: string) {
  try {
    const upstream = await fetch(url, { signal: AbortSignal.timeout(2000) })
    return upstream.ok ? "ok" : "down"
  } catch {
    return "down"
  }
}

app.get("/health", async (c) => {
  const [perception, agent] = await Promise.all([
    reach(`${apiBase}/openapi.json`),
    reach(`${agentBase}/health`),
  ])
  return c.json({ web: "ok", perception, agent, api: apiBase, agentApi: agentBase, dev })
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

app.all("/agent-api/*", async (c) => {
  const incoming = new URL(c.req.url)
  const suffix = incoming.pathname.replace(/^\/agent-api/, "") || "/"
  const target = `${agentBase}${suffix}${incoming.search}`
  const headers = new Headers()
  const contentType = c.req.header("content-type")
  if (contentType) headers.set("content-type", contentType)
  const hasBody = c.req.method !== "GET" && c.req.method !== "HEAD"
  try {
    const upstream = await undiciFetch(target, {
      method: c.req.method,
      headers,
      body: hasBody ? await c.req.raw.arrayBuffer() : undefined,
      signal: c.req.raw.signal,
      dispatcher: perceptionAgent,
    })
    const out = new Headers()
    const type = upstream.headers.get("content-type")
    if (type) out.set("content-type", type)
    return new Response(upstream.body, { status: upstream.status, headers: out })
  } catch (err) {
    console.error("agent proxy failed", err)
    return c.json({ detail: `Agent 不可达：${agentBase}` }, 502)
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
    const upstream = await undiciFetch(target, {
      method: c.req.method,
      headers,
      body: hasBody ? c.req.raw.body : undefined,
      duplex: hasBody ? "half" : undefined,
      signal: c.req.raw.signal,
      dispatcher: perceptionAgent,
    })
    const out = new Headers()
    for (const name of ["content-type", "cache-control"]) {
      const value = upstream.headers.get(name)
      if (value) out.set(name, value)
    }
    return new Response(upstream.body, { status: upstream.status, headers: out })
  } catch (err) {
    console.error("perception proxy failed", err)
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
app.get("/memory", file("memory.html", "text/html; charset=utf-8"))
app.get("/agent", file("agent.html", "text/html; charset=utf-8"))
for (const name of ["sam", "dino", "dinov3", "vlm"]) {
  app.get(`/${name}`, file("bench.html", "text/html; charset=utf-8"))
}
app.get("/styles.css", file("styles.css", "text/css; charset=utf-8"))
app.get("/memory.css", file("memory.css", "text/css; charset=utf-8"))
app.get("/app.js", file("app.js", "text/javascript; charset=utf-8"))
app.get("/memory.js", file("memory.js", "text/javascript; charset=utf-8"))
app.get("/nav.js", file("nav.js", "text/javascript; charset=utf-8"))
app.get("/bench.js", file("bench.js", "text/javascript; charset=utf-8"))
app.get("/camera.js", file("camera.js", "text/javascript; charset=utf-8"))
app.get("/agent.js", file("agent.js", "text/javascript; charset=utf-8"))

serve({ fetch: app.fetch, hostname: "0.0.0.0", port }, (info) => {
  const role = dev ? "dev" : "web"
  console.log(`RoomMind ${role} http://127.0.0.1:${info.port}`)
  console.log(`  perception ${apiBase}`)
  console.log(`  agent      ${agentBase}`)
})
