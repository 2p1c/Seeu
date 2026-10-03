const CAPTURE_KEY = "roomind.capture"

function fileFromDataUrl(dataUrl, name) {
  const match = /^data:(.*?);base64,(.*)$/.exec(dataUrl)
  if (!match) return null
  const binary = atob(match[2])
  const bytes = new Uint8Array(binary.length)
  for (let i = 0; i < binary.length; i += 1) bytes[i] = binary.charCodeAt(i)
  return new File([bytes], name, { type: match[1] || "image/jpeg" })
}

function readCapture() {
  try {
    const raw = sessionStorage.getItem(CAPTURE_KEY)
    if (!raw) return null
    const saved = JSON.parse(raw)
    return fileFromDataUrl(saved.dataUrl, saved.name || "capture.jpg")
  } catch {
    return null
  }
}

function saveNamedFile(key, file) {
  if (!file) {
    sessionStorage.removeItem(key)
    return
  }
  const reader = new FileReader()
  reader.onload = () => {
    try {
      sessionStorage.setItem(key, JSON.stringify({ name: file.name, dataUrl: reader.result }))
    } catch {
      /* 图片太大时不记下 */
    }
  }
  reader.readAsDataURL(file)
}

function loadNamedFile(key) {
  try {
    const raw = sessionStorage.getItem(key)
    if (!raw) return null
    const saved = JSON.parse(raw)
    return fileFromDataUrl(saved.dataUrl, saved.name || "image.jpg")
  } catch {
    return null
  }
}

function writeCapture(file, dataUrl) {
  try {
    sessionStorage.setItem(CAPTURE_KEY, JSON.stringify({ name: file.name, dataUrl }))
  } catch {
    /* 截图太大时只留在当前页 */
  }
}

function mountCamera(onUse) {
  const root = document.getElementById("camera")
  if (!root || root.dataset.ready === "1") return
  root.dataset.ready = "1"
  root.className = "camera"
  root.innerHTML = `
    <div class="row">
      <select id="camera-device" aria-label="摄像头"></select>
      <button type="button" class="text-btn" id="camera-toggle">打开预览</button>
      <button type="button" class="text-btn" id="camera-shot" disabled>截一张</button>
      <button type="button" class="text-btn" id="camera-use" disabled>用作本页输入</button>
    </div>
    <div class="camera-view" id="camera-view" hidden>
      <img id="camera-live" alt="摄像头预览">
    </div>
    <p class="camera-meta" id="camera-meta"></p>
  `

  const device = root.querySelector("#camera-device")
  const toggle = root.querySelector("#camera-toggle")
  const shotBtn = root.querySelector("#camera-shot")
  const useBtn = root.querySelector("#camera-use")
  const view = root.querySelector("#camera-view")
  const live = root.querySelector("#camera-live")
  const meta = root.querySelector("#camera-meta")
  let streaming = false
  let shot = readCapture()
  let infoTimer = 0
  toggle.disabled = true

  function setMeta(text) {
    meta.textContent = text
  }

  function formatInfo(info) {
    const rate = info.measured_fps > 0 ? info.measured_fps : info.fps
    return `${info.width}×${info.height} · ${rate} fps · ${info.fourcc || "unknown"}`
  }

  function remember(next) {
    shot = next
    useBtn.disabled = !shot
  }

  if (shot) {
    view.hidden = false
    live.src = URL.createObjectURL(shot)
    useBtn.disabled = false
    setMeta("已有截图，可当作本页输入")
  }

  async function loadDevices() {
    device.replaceChildren()
    try {
      const res = await fetch("/api/camera/devices")
      if (!res.ok) throw new Error("列不出摄像头")
      const devices = await res.json()
      if (!devices.length) {
        const option = document.createElement("option")
        option.textContent = "没有摄像头"
        device.append(option)
        toggle.disabled = true
        return
      }
      for (const item of devices) {
        const option = document.createElement("option")
        option.value = item.source
        option.textContent = item.name && item.name !== "camera" ? `${item.name} (${item.source})` : item.source
        device.append(option)
      }
      toggle.disabled = false
    } catch {
      const option = document.createElement("option")
      option.textContent = "摄像头不可用"
      device.append(option)
      toggle.disabled = true
      if (!shot) setMeta("感知服务未连接，无法打开摄像头")
    }
  }

  function stopStream() {
    streaming = false
    window.clearInterval(infoTimer)
    infoTimer = 0
    live.removeAttribute("src")
    shotBtn.disabled = true
    toggle.textContent = "打开预览"
  }

  async function pullInfo() {
    if (!streaming) return
    try {
      const res = await fetch("/api/camera/info")
      if (!res.ok) return
      const info = await res.json()
      if (info.open) setMeta(formatInfo(info))
    } catch {
      /* 下一轮再读 */
    }
  }

  function openPreview() {
    const source = device.value
    if (!source) return
    streaming = true
    view.hidden = false
    shotBtn.disabled = false
    toggle.textContent = "关闭预览"
    setMeta("正在打开摄像头")
    live.src = `/api/camera/stream?source=${encodeURIComponent(source)}`
    pullInfo()
    infoTimer = window.setInterval(pullInfo, 1000)
  }

  toggle.addEventListener("click", () => {
    if (streaming) {
      stopStream()
      view.hidden = !shot
      if (shot) {
        live.src = URL.createObjectURL(shot)
        setMeta("预览已关闭")
      } else {
        setMeta("")
      }
      return
    }
    openPreview()
  })

  live.addEventListener("error", () => {
    if (!streaming) return
    stopStream()
    view.hidden = !shot
    setMeta("打不开摄像头，可能正被占用")
  })

  shotBtn.addEventListener("click", () => {
    const width = live.naturalWidth
    const height = live.naturalHeight
    if (!streaming || !width || !height) {
      setMeta("还没有画面")
      return
    }
    const canvas = document.createElement("canvas")
    canvas.width = width
    canvas.height = height
    canvas.getContext("2d").drawImage(live, 0, 0)
    canvas.toBlob((blob) => {
      if (!blob) {
        setMeta("截图失败")
        return
      }
      const file = new File([blob], "capture.jpg", { type: "image/jpeg" })
      const dataUrl = canvas.toDataURL("image/jpeg", 0.92)
      writeCapture(file, dataUrl)
      remember(file)
      stopStream()
      live.src = dataUrl
      view.hidden = false
      setMeta(`${width}×${height} · 已截图，尚未运行`)
    }, "image/jpeg", 0.92)
  })

  useBtn.addEventListener("click", () => {
    const next = shot || readCapture()
    if (!next) return
    remember(next)
    onUse(next)
    setMeta("已用作本页输入，需要运行时再点本页按钮")
  })

  window.addEventListener("pagehide", stopStream)
  loadDevices()
}

window.mountCamera = mountCamera
window.saveNamedFile = saveNamedFile
window.loadNamedFile = loadNamedFile
