const $ = (id) => document.getElementById(id)

function chip(text) {
  const span = document.createElement("span")
  span.textContent = text
  return span
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

function setStatus(text, failed) {
  const node = $("status")
  node.textContent = text
  node.className = failed ? "link down" : "link ok"
}

function classText(classes) {
  if (!Array.isArray(classes) || !classes.length) return "—"
  return classes.map((item) => `${item.name} ${Number(item.score).toFixed(3)}`).join("、")
}

function renderObjects(frame) {
  const objects = frame.objects || []
  $("frame-meta").replaceChildren(
    chip(frame.filename || "未命名"),
    chip(`${frame.width} × ${frame.height}`),
    chip(frame.captured_at || "—"),
    chip(`${objects.length} 个物体`),
  )
  $("object-empty").hidden = objects.length > 0
  $("object-empty").textContent = "这个画面没有物体。"
  const list = $("objects")
  list.replaceChildren()
  for (const obj of objects) {
    const card = document.createElement("article")
    card.className = "object"
    const shot = document.createElement("div")
    shot.className = "object-shot"
    if (obj.crop) {
      const img = document.createElement("img")
      img.alt = `物体 ${obj.id} 的裁剪`
      img.src = `data:image/jpeg;base64,${obj.crop}`
      shot.append(img)
    }
    const fields = document.createElement("dl")
    const mask = Array.isArray(obj.mask_size) ? obj.mask_size.join(" × ") : "—"
    for (const [label, value] of [
      ["编号", obj.id],
      ["类别", obj.label || "—"],
      ["候选", classText(obj.classes)],
      ["描述", obj.description || "—"],
      ["框", (obj.bounding_box || []).join(", ")],
      ["掩码", mask],
      ["向量", obj.embedding_dim ? `${obj.embedding_dim} 维` : "—"],
    ]) {
      const wrap = document.createElement("div")
      const dt = document.createElement("dt")
      dt.textContent = label
      const dd = document.createElement("dd")
      dd.textContent = String(value)
      wrap.append(dt, dd)
      fields.append(wrap)
    }
    card.append(shot, fields)
    list.append(card)
  }
}

async function openFrame(id, button) {
  try {
    sessionStorage.setItem("roomind.memory.frame", String(id))
  } catch {
    /* 忽略 */
  }
  for (const item of document.querySelectorAll(".frames button")) {
    item.setAttribute("aria-pressed", String(item === button))
  }
  setStatus("正在读取画面", false)
  const res = await fetch(`/api/memory/frames/${id}`)
  if (!res.ok) {
    setStatus(await errorText(res), true)
    return
  }
  renderObjects(await res.json())
  setStatus("已连接记忆库", false)
}

async function loadFrames() {
  const res = await fetch("/api/memory/frames")
  if (!res.ok) {
    setStatus(await errorText(res), true)
    return
  }
  const frames = await res.json()
  const list = $("frames")
  list.replaceChildren()
  $("frame-empty").hidden = frames.length > 0
  if (!frames.length) {
    setStatus("库里还没有画面", false)
    return
  }
  for (const frame of frames) {
    const item = document.createElement("li")
    const button = document.createElement("button")
    button.type = "button"
    button.dataset.id = String(frame.id)
    button.setAttribute("aria-pressed", "false")
    const name = document.createElement("strong")
    name.textContent = frame.filename || `画面 ${frame.id}`
    const meta = document.createElement("span")
    meta.textContent = `${frame.captured_at} · ${frame.object_count} 个物体 · ${frame.width}×${frame.height}`
    button.append(name, meta)
    button.addEventListener("click", () => openFrame(frame.id, button))
    item.append(button)
    list.append(item)
  }
  setStatus(`${frames.length} 个画面`, false)
  const saved = sessionStorage.getItem("roomind.memory.frame")
  const buttons = [...list.querySelectorAll("button")]
  const picked = buttons.find((button) => button.dataset.id === saved) || buttons[0]
  picked.click()
}

loadFrames().catch((err) => {
  setStatus(err instanceof Error ? err.message : "读取失败", true)
})
