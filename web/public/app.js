const $ = (id) => document.getElementById(id)

const STAGES = [
  { id: "sam", label: "分割", hint: "SAM" },
  { id: "siglip", label: "分类", hint: "SigLIP" },
  { id: "dinov3", label: "向量", hint: "DINOv3" },
  { id: "vlm", label: "描述", hint: "VLM" },
]

let file = null
let pollTimer = 0
let latest = {
  running: false,
  stage: "idle",
  detail: "",
  index: null,
  total: null,
  error: "",
  details: {},
  seconds: {},
}

function idleProgress() {
  return {
    running: false,
    stage: "idle",
    detail: "",
    index: null,
    total: null,
    error: "",
    details: {},
    seconds: {},
  }
}

function formatSeconds(value) {
  if (typeof value !== "number" || Number.isNaN(value)) return ""
  return `${value.toFixed(2)} 秒`
}

function stageRowState(id, progress) {
  const order = STAGES.map((item) => item.id)
  const current = progress.stage
  if (current === "done") return "done"
  if (current === "idle") return "pending"
  const at = order.indexOf(current)
  const index = order.indexOf(id)
  if (at < 0 || index < 0) return "pending"
  if (index < at) return "done"
  if (index > at) return "pending"
  return progress.error ? "error" : "active"
}

function statusText(progress) {
  if (progress.error) return progress.error
  if (progress.stage === "sam") return "正在分割画面"
  if (progress.stage === "siglip") return progress.detail ? `正在分类，${progress.detail}` : "正在分类"
  if (progress.stage === "dinov3") return progress.detail ? `正在计算向量，${progress.detail}` : "正在计算向量"
  if (progress.stage === "vlm") {
    if (progress.total) return `正在描述物体 ${progress.index + 1}/${progress.total}`
    return progress.detail ? `正在描述，${progress.detail}` : "正在描述物体"
  }
  if (progress.stage === "done") return progress.detail ? `处理完成，${progress.detail}` : "处理完成"
  return file ? "可以开始处理" : "等待图片"
}

function paintProgress(progress) {
  latest = progress
  $("phase-status").textContent = statusText(progress)
  $("phase-status").className = progress.error ? "status error" : "status"
  for (const item of STAGES) {
    const row = document.querySelector(`[data-stage="${item.id}"]`)
    if (!row) continue
    const state = stageRowState(item.id, progress)
    row.className = state
    row.querySelector(".phase-state").textContent = {
      pending: "等待",
      active: "进行中",
      done: "完成",
      error: "失败",
    }[state]
    const detail = (progress.details && progress.details[item.id]) || ""
    row.querySelector(".phase-detail").textContent = detail
    const seconds = progress.seconds && progress.seconds[item.id]
    row.querySelector(".phase-time").textContent = state === "pending" ? "" : formatSeconds(seconds)
  }
}

function buildPhases() {
  const list = $("phases")
  list.replaceChildren()
  for (const item of STAGES) {
    const row = document.createElement("li")
    row.dataset.stage = item.id
    row.className = "pending"
    const name = document.createElement("span")
    name.className = "phase-name"
    name.textContent = `${item.label} · ${item.hint}`
    const detail = document.createElement("span")
    detail.className = "phase-detail"
    const time = document.createElement("span")
    time.className = "phase-time"
    const state = document.createElement("span")
    state.className = "phase-state"
    state.textContent = "等待"
    row.append(name, detail, time, state)
    list.append(row)
  }
}

async function refreshHealth() {
  const node = $("link-status")
  try {
    const res = await fetch("/health")
    const body = await res.json()
    if (body.perception === "ok") {
      node.textContent = "感知服务已连接"
      node.className = "link ok"
    } else {
      node.textContent = `感知服务未连接（${body.api}）`
      node.className = "link down"
    }
  } catch {
    node.textContent = "页面服务异常"
    node.className = "link down"
  }
}

function useFile(next) {
  if (!next || !next.type.startsWith("image/")) return
  file = next
  $("preview").src = URL.createObjectURL(next)
  $("preview-stage").hidden = false
  $("drop-label").textContent = next.name
  $("run").disabled = false
  if (latest.stage === "idle") paintProgress(latest)
}

async function errorText(res) {
  const text = await res.text()
  try {
    const body = JSON.parse(text)
    if (typeof body.detail === "string") return body.detail
    if (Array.isArray(body.detail)) {
      return body.detail.map((item) => item.msg || JSON.stringify(item)).join("；")
    }
  } catch {
    /* 非 JSON */
  }
  return text || `请求失败 ${res.status}`
}

function artifactUrl(filePath) {
  if (!filePath) return ""
  const name = String(filePath).split(/[/\\]/).pop() || ""
  if (!/^[A-Za-z0-9][A-Za-z0-9._-]*$/.test(name)) return ""
  return `/api/artifacts/${encodeURIComponent(name)}`
}

function stopPoll() {
  window.clearInterval(pollTimer)
  pollTimer = 0
}

async function pullProgress() {
  try {
    const res = await fetch("/api/scene/progress")
    if (!res.ok) return
    paintProgress(await res.json())
  } catch {
    /* 下一轮再试 */
  }
}

function startPoll() {
  stopPoll()
  pullProgress()
  pollTimer = window.setInterval(pullProgress, 700)
}

function chip(text) {
  const span = document.createElement("span")
  span.textContent = text
  return span
}

function longValue(summary, text) {
  const details = document.createElement("details")
  const title = document.createElement("summary")
  title.textContent = summary
  const pre = document.createElement("pre")
  pre.textContent = text
  details.append(title, pre)
  return details
}

function field(label, value) {
  const wrap = document.createElement("div")
  const dt = document.createElement("dt")
  dt.textContent = label
  const dd = document.createElement("dd")
  if (value instanceof Node) dd.append(value)
  else dd.textContent = value == null || value === "" ? "—" : String(value)
  wrap.append(dt, dd)
  return wrap
}

function renderObjects(data) {
  const objects = data.objects || []
  const meta = $("frame-meta")
  meta.replaceChildren(
    chip(data.filename || "未命名"),
    chip(`${data.width ?? "—"} × ${data.height ?? "—"}`),
    chip(data.timestamp || "—"),
    chip(`${objects.length} 个物体`),
  )
  const src = artifactUrl(data.image_path)
  $("result-image").hidden = !src
  $("result-missing").hidden = Boolean(src)
  if (src) {
    $("result-image").src = src
    $("result-caption").textContent = data.filename ? `${data.filename} 的结果图` : "结果图"
  } else {
    $("result-image").removeAttribute("src")
  }
  $("object-empty").hidden = objects.length > 0
  const list = $("object-list")
  list.replaceChildren()
  for (const obj of objects) {
    const card = document.createElement("article")
    card.className = "object"
    const classes = obj.class || obj.object_class || []
    const classText = classes.length
      ? classes.map((item) => `${item.name} ${Number(item.score).toFixed(3)}`).join("、")
      : "—"
    const mask = obj.mask || {}
    const counts = Array.isArray(mask.counts) ? mask.counts.join(", ") : ""
    const embedding = Array.isArray(obj.embedding) ? obj.embedding : []
    const crop = document.createElement("img")
    crop.alt = `物体 ${obj.id} 的裁剪`
    if (obj.crop) crop.src = `data:image/jpeg;base64,${obj.crop}`
    const shot = document.createElement("div")
    shot.className = "object-shot"
    shot.append(crop)
    const fields = document.createElement("dl")
    fields.append(
      field("编号", obj.id),
      field("类别", classText),
      field("描述", obj.description || "—"),
      field("框", (obj.bounding_box || []).join(", ")),
      field(
        "掩码",
        longValue(
          mask.size ? `${mask.size[0]} × ${mask.size[1]}` : "无",
          counts || "无",
        ),
      ),
      field(
        "向量",
        longValue(
          embedding.length ? `${embedding.length} 维` : "无",
          embedding.join(", ") || "无",
        ),
      ),
    )
    card.append(shot, fields)
    list.append(card)
  }
  $("results").hidden = false
}

async function run() {
  if (!file) return
  $("results").hidden = true
  $("run").disabled = true
  $("run").textContent = "处理中…"
  paintProgress({ ...idleProgress(), running: true, stage: "sam" })
  startPoll()
  try {
    const form = new FormData()
    form.append("image", file, file.name || "upload.jpg")
    const res = await fetch("/api/scene", { method: "POST", body: form })
    stopPoll()
    await pullProgress()
    if (!res.ok) {
      paintProgress({
        ...latest,
        running: false,
        error: await errorText(res),
        stage: latest.stage === "idle" || latest.stage === "done" ? "error" : latest.stage,
      })
      return
    }
    const data = await res.json()
    paintProgress({
      ...latest,
      running: false,
      stage: "done",
      error: "",
      detail: `${(data.objects || []).length} 个物体`,
    })
    renderObjects(data)
  } catch (err) {
    stopPoll()
    paintProgress({
      ...latest,
      running: false,
      error: err instanceof Error ? err.message : "处理失败",
      stage: latest.stage === "idle" || latest.stage === "done" ? "error" : latest.stage,
    })
  } finally {
    $("run").textContent = "开始处理"
    $("run").disabled = !file
  }
}

buildPhases()
paintProgress(idleProgress())
refreshHealth()
setInterval(refreshHealth, 5000)

$("file").addEventListener("change", () => useFile($("file").files[0]))
const drop = $("drop")
drop.addEventListener("dragover", (event) => {
  event.preventDefault()
  drop.classList.add("hot")
})
drop.addEventListener("dragleave", () => drop.classList.remove("hot"))
drop.addEventListener("drop", (event) => {
  event.preventDefault()
  drop.classList.remove("hot")
  useFile(event.dataTransfer.files[0])
})
$("run").addEventListener("click", run)
