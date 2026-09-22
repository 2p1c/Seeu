const $ = (id) => document.getElementById(id)

let service = "vlm"
let file = null
let cameraOpen = false
let infoTimer = 0

function updatePointTotal() {
  const n = Math.max(1, Number($("points-crop").value) || 1)
  $("point-total").textContent = String(n * n)
}

function updateRunState() {
  const needsVlm = service === "vlm" || service === "all"
  $("run").disabled = !file || cameraOpen || (needsVlm && !$("vlm-objects").value.trim())
}

function showService(name) {
  service = name
  for (const button of document.querySelectorAll(".svc")) {
    button.setAttribute("aria-pressed", String(button.dataset.service === name))
  }
  $("fields-vlm").hidden = name !== "vlm" && name !== "all"
  $("fields-dino").hidden = name !== "dino" && name !== "all"
  $("fields-sam").hidden = name !== "sam" && name !== "all"
  const hints = {
    vlm: "VLM 按你写下的物体生成文字描述，不返回标注图。",
    dino: "DINO 按英文短语找物体，返回检测框和带框图片。",
    sam: "SAM 自动分割。每边点数决定采样密度，每批点数决定一次送进显存的点数。",
    all: "按 VLM → DINO → SAM 依次请求同一张图。某一步失败后不再继续。",
  }
  $("run-hint").textContent = hints[name]
  updateRunState()
}

function setSource(mode) {
  const camera = mode === "camera"
  $("tab-camera").setAttribute("aria-selected", String(camera))
  $("tab-upload").setAttribute("aria-selected", String(!camera))
  $("pane-camera").hidden = !camera
  $("pane-upload").hidden = camera
  if (!camera) closeCamera()
  updateRunState()
}

function localDateTimeValue() {
  const now = new Date()
  const pad = (n) => String(n).padStart(2, "0")
  return `${now.getFullYear()}-${pad(now.getMonth() + 1)}-${pad(now.getDate())}T${pad(now.getHours())}:${pad(now.getMinutes())}`
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

async function loadDevices() {
  const select = $("camera-source")
  select.replaceChildren()
  try {
    const res = await fetch("/api/camera/devices")
    if (!res.ok) throw new Error(await errorText(res))
    const devices = await res.json()
    $("camera-note").textContent = "实时预览只显示画面，不运行推理。"
    if (!devices.length) {
      const option = document.createElement("option")
      option.value = "0"
      option.textContent = "未发现设备，仍可尝试 0"
      select.append(option)
      return
    }
    for (const device of devices) {
      const option = document.createElement("option")
      option.value = device.source
      option.textContent = `${device.source}  ${device.name}`
      select.append(option)
    }
  } catch (err) {
    const option = document.createElement("option")
    option.value = "0"
    option.textContent = "设备列表获取失败"
    select.append(option)
    const message = err instanceof Error ? err.message : "无法列出摄像头"
    $("camera-note").textContent = `设备列表暂不可用：${message}`
  }
}

function paintCameraInfo(info) {
  if (!info?.open) {
    $("camera-stats").hidden = true
    return
  }
  $("camera-stats").hidden = false
  $("stat-size").textContent = `${info.width} × ${info.height}`
  $("stat-fps").textContent = info.fps ? `${info.fps} fps` : "—"
  $("stat-measured").textContent = info.measured_fps ? `${info.measured_fps} fps` : "统计中"
  $("stat-fourcc").textContent = info.fourcc || "—"
  $("stat-source").textContent = info.source || "—"
  $("stat-cap").textContent = `${info.max_fps} fps`
  if (info.width !== info.requested_width || info.height !== info.requested_height) {
    $("stat-size").textContent = `${info.width} × ${info.height}（请求 ${info.requested_width}×${info.requested_height}）`
  }
}

async function pollCameraInfo() {
  try {
    const res = await fetch("/api/camera/info")
    paintCameraInfo(await res.json())
  } catch {
    paintCameraInfo(null)
  }
}

function closeCamera() {
  cameraOpen = false
  $("camera-view").removeAttribute("src")
  $("camera-toggle").textContent = "打开预览"
  $("camera-empty").hidden = false
  $("camera-stats").hidden = true
  window.clearInterval(infoTimer)
  updateRunState()
}

async function openCamera() {
  const params = new URLSearchParams({
    source: $("camera-source").value || "0",
    width: $("camera-width").value || "1920",
    height: $("camera-height").value || "1080",
    max_fps: "10",
  })
  $("camera-note").textContent = "正在打开摄像头…"
  $("camera-view").src = `/api/camera/stream?${params}`
  cameraOpen = true
  $("camera-toggle").textContent = "关闭预览"
  $("camera-empty").hidden = true
  updateRunState()
  window.clearInterval(infoTimer)
  infoTimer = window.setInterval(pollCameraInfo, 1000)
  setTimeout(pollCameraInfo, 400)
}

function useFile(next) {
  if (!next || !next.type.startsWith("image/")) return
  file = next
  $("upload-view").src = URL.createObjectURL(next)
  $("upload-stage").hidden = false
  $("drop-label").textContent = next.name
  updateRunState()
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

function chip(text) {
  const span = document.createElement("span")
  span.textContent = text
  return span
}

function figure(src, caption) {
  const fig = document.createElement("figure")
  const img = document.createElement("img")
  img.src = src
  img.alt = caption
  const cap = document.createElement("figcaption")
  cap.textContent = caption
  fig.append(img, cap)
  return fig
}

function rawDetails(data) {
  const details = document.createElement("details")
  const summary = document.createElement("summary")
  summary.textContent = "原始返回"
  const pre = document.createElement("pre")
  pre.textContent = JSON.stringify(data, null, 2)
  details.append(summary, pre)
  return details
}

function resultCard(title) {
  const card = document.createElement("article")
  card.className = "result"
  const heading = document.createElement("h2")
  heading.textContent = title
  card.append(heading)
  $("results").append(card)
  return card
}

function renderVlm(card, data) {
  const meta = document.createElement("div")
  meta.className = "meta"
  meta.append(
    chip(data.location || "—"),
    chip(data.time || "—"),
    chip((data.objects || []).join("、") || "无物体"),
  )
  const text = document.createElement("p")
  text.textContent = data.description || "（没有描述）"
  card.append(meta, text, rawDetails(data))
}

function renderDino(card, data) {
  const meta = document.createElement("div")
  meta.className = "meta"
  meta.append(
    chip(`${data.detection_count ?? 0} 个框`),
    chip(`${data.inference_ms ?? "—"} ms`),
    chip(`模型显存 ${data.vram_model_mb ?? "—"} MB`),
    chip(`峰值 ${data.vram_peak_mb ?? "—"} MB`),
    chip(`${data.device || "—"} / ${data.dtype || "—"}`),
  )
  card.append(meta)
  const count = Number(data.detection_count ?? (data.objects || []).length)
  const src = artifactUrl(data.annotated_path)
  if (count > 0 && src) {
    const shots = document.createElement("div")
    shots.className = "shots"
    shots.append(figure(src, "带检测框的图片"))
    card.append(shots)
  }
  const table = document.createElement("table")
  const head = document.createElement("tr")
  for (const name of ["物体", "置信度", "框"]) {
    const th = document.createElement("th")
    th.textContent = name
    head.append(th)
  }
  table.append(head)
  for (const obj of data.objects || []) {
    const row = document.createElement("tr")
    for (const value of [obj.label, obj.score, (obj.bbox || []).join(", ")]) {
      const td = document.createElement("td")
      td.textContent = String(value ?? "")
      row.append(td)
    }
    table.append(row)
  }
  card.append(table, rawDetails(data))
}

function renderSam(card, data) {
  const crop = Number(data.points_per_crop) || 0
  const meta = document.createElement("div")
  meta.className = "meta"
  meta.append(
    chip(`${data.mask_count ?? 0} 个 mask`),
    chip(`每边 ${data.points_per_crop ?? "—"} 点`),
    chip(`采样 ${crop * crop} 点`),
    chip(`每批 ${data.points_per_batch ?? "—"} 点`),
    chip(`${data.inference_ms ?? "—"} ms`),
    chip(`模型显存 ${data.vram_model_mb ?? "—"} MB`),
    chip(`峰值 ${data.vram_peak_mb ?? "—"} MB`),
  )
  card.append(meta)
  const shots = document.createElement("div")
  shots.className = "shots"
  const overlay = artifactUrl(data.overlay_path)
  const masks = artifactUrl(data.masks_path)
  if (overlay) shots.append(figure(overlay, "叠加图"))
  if (masks) shots.append(figure(masks, "mask 图"))
  if (shots.childElementCount) card.append(shots)
  card.append(rawDetails(data))
}

async function postService(name) {
  const form = new FormData()
  form.append("image", file, file.name || "upload.jpg")
  if (name === "vlm") {
    form.append("prompt", $("vlm-prompt").value.trim())
    form.append("time", $("vlm-time").value || localDateTimeValue())
    form.append("location", $("vlm-location").value.trim())
    form.append("objects", $("vlm-objects").value.trim())
    return fetch("/api/vlm/analyze", { method: "POST", body: form })
  }
  if (name === "dino") {
    form.append("prompt", $("dino-prompt").value.trim())
    return fetch("/api/dino/detect", { method: "POST", body: form })
  }
  const query = new URLSearchParams({
    points_per_crop: $("points-crop").value || "16",
    points_per_batch: $("points-batch").value || "8",
  })
  return fetch(`/api/sam/segment?${query}`, { method: "POST", body: form })
}

async function run() {
  const steps = service === "all" ? ["vlm", "dino", "sam"] : [service]
  const titles = { vlm: "VLM", dino: "DINO", sam: "SAM" }
  $("results").replaceChildren()
  $("run").disabled = true
  $("run").textContent = "推理中…"
  try {
    for (const step of steps) {
      const card = resultCard(`${titles[step]} 推理中`)
      const res = await postService(step)
      if (!res.ok) {
        card.classList.add("error")
        card.querySelector("h2").textContent = `${titles[step]} 失败`
        const text = document.createElement("p")
        text.textContent = await errorText(res)
        card.append(text)
        break
      }
      const data = await res.json()
      card.querySelector("h2").textContent = titles[step]
      if (step === "vlm") renderVlm(card, data)
      if (step === "dino") renderDino(card, data)
      if (step === "sam") renderSam(card, data)
    }
  } finally {
    $("run").textContent = "运行"
    updateRunState()
  }
}

$("vlm-time").value = localDateTimeValue()
updatePointTotal()
showService("vlm")
setSource("camera")
loadDevices()
refreshHealth()
setInterval(refreshHealth, 5000)

$("tab-camera").addEventListener("click", () => setSource("camera"))
$("tab-upload").addEventListener("click", () => setSource("upload"))
$("camera-refresh").addEventListener("click", loadDevices)
$("camera-toggle").addEventListener("click", () => {
  if (cameraOpen) {
    closeCamera()
    $("camera-note").textContent = "实时预览只显示画面，不运行推理。"
    return
  }
  openCamera()
})
$("camera-view").addEventListener("error", () => {
  if (!cameraOpen) return
  closeCamera()
  $("camera-note").textContent = "预览中断。摄像头可能被占用，或感知服务没有打开它。"
})
$("vlm-objects").addEventListener("input", updateRunState)
$("points-crop").addEventListener("input", updatePointTotal)
for (const button of document.querySelectorAll(".svc")) {
  button.addEventListener("click", () => showService(button.dataset.service))
}
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
