const $ = (id) => document.getElementById(id)

const services = {
  sam: {
    title: "SAM 分割",
    note: "只跑自动分割。每批点数越小越省显存。",
    fields: [
      { name: "points_per_batch", label: "每批点数", value: "8" },
      { name: "points_per_crop", label: "每边点数", value: "16" },
      { name: "max_size", label: "最长边", value: "1024" },
    ],
    async submit(file, values) {
      const query = new URLSearchParams(values)
      const body = new FormData()
      body.append("image", file)
      return fetch(`/api/sam/segment?${query}`, { method: "POST", body })
    },
    images(data) {
      return [
        { src: artifactUrl(data.overlay_path), caption: "叠加图" },
        { src: artifactUrl(data.masks_path), caption: "mask" },
      ].filter((item) => item.src)
    },
  },
  dino: {
    title: "Grounding DINO",
    note: "开放词汇检测。提示词用英文短语，用句号分开。",
    fields: [
      { name: "prompt", label: "提示词", value: "a chair. a sofa.", text: true },
      { name: "box_threshold", label: "框阈值", value: "0.4" },
      { name: "text_threshold", label: "文本阈值", value: "0.3" },
      { name: "max_size", label: "最长边", value: "800" },
    ],
    async submit(file, values) {
      const query = new URLSearchParams({
        box_threshold: values.box_threshold,
        text_threshold: values.text_threshold,
        max_size: values.max_size,
      })
      const body = new FormData()
      body.append("image", file)
      body.append("prompt", values.prompt)
      return fetch(`/api/dino/detect?${query}`, { method: "POST", body })
    },
    images(data) {
      const src = artifactUrl(data.annotated_path)
      return src ? [{ src, caption: "标注图" }] : []
    },
  },
  dinov3: {
    title: "DINOv3 相似度",
    note: "上传两张图，比较它们向量的余弦相似度。越接近 1，画面越像。",
    fields: [],
    pair: true,
    async submit(_file, _values, files) {
      const body = new FormData()
      body.append("image_a", files.a)
      body.append("image_b", files.b)
      return fetch("/api/dinov3/compare", { method: "POST", body })
    },
    images() {
      return []
    },
  },
  vlm: {
    title: "VLM 描述",
    note: "只跑看图描述，需要本机 Ollama 里的 Qwen3-VL。",
    fields: [
      { name: "objects", label: "物体", value: "椅子、桌子", text: true },
      { name: "location", label: "地点", value: "客厅" },
      { name: "time", label: "时间", value: "" },
      { name: "prompt", label: "提示词", value: "描述这些物体在画面中的位置和状态。", text: true },
    ],
    async submit(file, values) {
      const body = new FormData()
      body.append("image", file)
      body.append("prompt", values.prompt)
      body.append("time", values.time)
      body.append("location", values.location)
      body.append("objects", values.objects)
      return fetch("/api/vlm/analyze", { method: "POST", body })
    },
    images() {
      return []
    },
  },
}

function artifactUrl(filePath) {
  if (!filePath) return ""
  const name = String(filePath).split(/[/\\]/).pop() || ""
  if (!/^[A-Za-z0-9][A-Za-z0-9._-]*$/.test(name)) return ""
  return `/api/artifacts/${encodeURIComponent(name)}`
}

function serviceOf(path) {
  const name = path.replace(/^\//, "").split("/")[0]
  return services[name] ? name : ""
}

async function errorText(res) {
  const text = await res.text()
  try {
    const body = JSON.parse(text)
    if (typeof body.detail === "string") return body.detail
  } catch {
    /* 非 JSON */
  }
  return text || `请求失败 ${res.status}`
}

const name = serviceOf(location.pathname)
const service = services[name]
let file = null
let fileA = null
let fileB = null
const paired = Boolean(service && service.pair)

if (paired) {
  $("drop").hidden = true
  $("pair").hidden = false
  $("status").textContent = "需要两张图片"
}

if (!service) {
  $("status").textContent = "未知页面"
} else {
  document.title = `RoomMind ${service.title}`
  $("page-name").textContent = service.title
  $("title").textContent = service.title
  $("note").textContent = service.note
  if (name === "vlm") {
    const time = service.fields.find((item) => item.name === "time")
    if (time) time.value = new Date().toISOString().slice(0, 16)
  }
  for (const field of service.fields) {
    const label = document.createElement("label")
    label.textContent = field.label
    const input = document.createElement(field.text ? "textarea" : "input")
    input.id = `field-${field.name}`
    input.value = field.value
    if (field.text) input.rows = 2
    label.append(input)
    $("fields").append(label)
  }
}

function values() {
  const found = {}
  for (const field of service.fields) found[field.name] = $(`field-${field.name}`).value.trim()
  return found
}

function ready() {
  return paired ? Boolean(fileA && fileB) : Boolean(file)
}

function showFile(next) {
  file = next
  saveNamedFile(`roomind.bench.file.${name}`, next)
  $("drop-label").textContent = next ? next.name : "选择或拖入一张图片"
  $("run").disabled = !next
  $("preview-stage").hidden = !next
  if (next) $("preview").src = URL.createObjectURL(next)
  $("status").textContent = next ? "可以运行" : "等待图片"
}

function showPair(which, next) {
  if (which === "a") fileA = next
  else fileB = next
  saveNamedFile(`roomind.bench.${which}.${name}`, next)
  const label = which === "a" ? $("drop-label-a") : $("drop-label-b")
  label.textContent = next ? next.name : which === "a" ? "第一张图片" : "第二张图片"
  $("run").disabled = !ready()
  $("status").textContent = ready() ? "可以比较" : "需要两张图片"
}

$("file-a").addEventListener("change", () => showPair("a", $("file-a").files[0] || null))
$("file-b").addEventListener("change", () => showPair("b", $("file-b").files[0] || null))

$("file").addEventListener("change", () => showFile($("file").files[0] || null))
$("drop").addEventListener("dragover", (event) => {
  event.preventDefault()
  $("drop").classList.add("hot")
})
$("drop").addEventListener("dragleave", () => $("drop").classList.remove("hot"))
$("drop").addEventListener("drop", (event) => {
  event.preventDefault()
  $("drop").classList.remove("hot")
  showFile(event.dataTransfer.files[0] || null)
})

$("run").addEventListener("click", async () => {
  if (!ready() || !service) return
  $("run").disabled = true
  $("status").textContent = "运行中，请稍等"
  $("status").className = "status"
  $("shots").replaceChildren()
  $("json").hidden = true
  try {
    const res = await service.submit(file, values(), { a: fileA, b: fileB })
    if (!res.ok) throw new Error(await errorText(res))
    const data = await res.json()
    for (const item of service.images(data)) {
      const figure = document.createElement("figure")
      const img = document.createElement("img")
      img.src = item.src
      img.alt = item.caption
      const caption = document.createElement("figcaption")
      caption.textContent = item.caption
      figure.append(img, caption)
      $("shots").append(figure)
    }
    $("json").textContent = JSON.stringify(data, null, 2)
    $("json").hidden = false
    if (typeof data.cosine_similarity === "number") {
      $("status").textContent = `余弦相似度 ${data.cosine_similarity}`
    } else {
      $("status").textContent = "完成"
    }
  } catch (err) {
    $("status").textContent = err instanceof Error ? err.message : "请求失败"
    $("status").className = "status error"
  } finally {
    $("run").disabled = !ready()
    saveBenchView()
  }
})

function saveBenchView() {
  const images = [...$("shots").querySelectorAll("figure")].map((figure) => ({
    src: figure.querySelector("img")?.getAttribute("src") || "",
    caption: figure.querySelector("figcaption")?.textContent || "",
  }))
  try {
    sessionStorage.setItem(`roomind.bench.view.${name}`, JSON.stringify({
      status: $("status").textContent,
      failed: $("status").classList.contains("error"),
      json: $("json").hidden ? "" : $("json").textContent,
      images,
    }))
  } catch {
    /* 结果太大时不记下 */
  }
}

function restoreBenchView() {
  let saved = null
  try {
    saved = JSON.parse(sessionStorage.getItem(`roomind.bench.view.${name}`) || "null")
  } catch {
    saved = null
  }
  if (!saved) return
  $("shots").replaceChildren()
  for (const item of saved.images || []) {
    if (!item.src) continue
    const figure = document.createElement("figure")
    const img = document.createElement("img")
    img.src = item.src
    img.alt = item.caption || "结果"
    const caption = document.createElement("figcaption")
    caption.textContent = item.caption || ""
    figure.append(img, caption)
    $("shots").append(figure)
  }
  if (saved.json) {
    $("json").textContent = saved.json
    $("json").hidden = false
  }
  $("status").textContent = saved.status || $("status").textContent
  $("status").className = saved.failed ? "status error" : "status"
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
      node.textContent = "感知服务未连接"
      node.className = "link down"
    }
  } catch {
    node.textContent = "页面服务异常"
    node.className = "link down"
  }
}

if (paired) {
  const keptA = loadNamedFile(`roomind.bench.a.${name}`)
  const keptB = loadNamedFile(`roomind.bench.b.${name}`)
  if (keptA) showPair("a", keptA)
  if (keptB) showPair("b", keptB)
} else {
  const kept = loadNamedFile(`roomind.bench.file.${name}`)
  if (kept) showFile(kept)
}
restoreBenchView()
refreshHealth()
mountCamera((shot) => {
  if (paired) showPair("a", shot)
  else showFile(shot)
})
