const $ = (id) => document.getElementById(id)

const CHAT_KEY = "roomind.agent.messages"
const messages = []
let busy = false

function persistChat() {
  try {
    sessionStorage.setItem(CHAT_KEY, JSON.stringify(messages))
  } catch {
    /* 忽略 */
  }
}

function setBusy(next) {
  busy = next
  $("send").disabled = next
  $("input").disabled = next
  $("compact").disabled = next || messages.length === 0
  $("send").textContent = next ? "思考中…" : "发送"
}

function hideEmpty() {
  const empty = $("empty")
  if (empty) empty.remove()
}

function addBubble(role, text, usage) {
  hideEmpty()
  const node = document.createElement("article")
  node.className = `bubble ${role}`
  node.textContent = text
  if (usage) {
    const meta = document.createElement("p")
    meta.className = "usage"
    meta.textContent = `本次 ${usage.total_tokens} tokens（输入 ${usage.prompt_tokens}，输出 ${usage.completion_tokens}）`
    node.append(meta)
  }
  $("transcript").append(node)
  node.scrollIntoView({ block: "nearest" })
}

function addNotice(text) {
  hideEmpty()
  const node = document.createElement("p")
  node.className = "notice"
  node.textContent = text
  $("transcript").append(node)
}

function addError(text) {
  hideEmpty()
  const node = document.createElement("p")
  node.className = "bubble error"
  node.textContent = text
  $("transcript").append(node)
}

function renderAll() {
  $("transcript").replaceChildren()
  if (messages.length === 0) {
    const empty = document.createElement("p")
    empty.className = "empty"
    empty.id = "empty"
    empty.textContent = "问当前空间里有什么。Agent 通过感知服务读取最新画面，库是空的就会如实说明。"
    $("transcript").append(empty)
    return
  }
  for (const msg of messages) {
    const text = typeof msg.content === "string" ? msg.content : ""
    if (msg.role === "system") addNotice(text || "上下文摘要")
    else if (msg.role === "user" || msg.role === "assistant") addBubble(msg.role, text)
  }
}

async function readBody(res) {
  const text = await res.text()
  try {
    return JSON.parse(text)
  } catch {
    return { detail: text || `请求失败 ${res.status}` }
  }
}

async function send(text) {
  messages.push({ role: "user", content: text })
  addBubble("user", text)
  setBusy(true)
  try {
    const res = await fetch("/agent-api/complete", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ messages }),
    })
    const body = await readBody(res)
    if (!res.ok) {
      messages.pop()
      addError(body.detail || body.error || "请求失败")
      return
    }
    messages.push({ role: "assistant", content: body.content || "" })
    addBubble("assistant", body.content || "", body.usage)
  } catch (err) {
    messages.pop()
    addError(err instanceof Error ? err.message : "请求失败")
  } finally {
    setBusy(false)
    persistChat()
  }
}

async function compact() {
  setBusy(true)
  try {
    const res = await fetch("/agent-api/compact", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ messages }),
    })
    const body = await readBody(res)
    if (!res.ok) {
      addError(body.detail || body.error || "压缩失败")
      return
    }
    messages.splice(0, messages.length, ...(body.compacted || []))
    if (!body.skipped) renderAll()
    addNotice(body.notice || (body.skipped ? "无需压缩" : "上下文已压缩"))
  } catch (err) {
    addError(err instanceof Error ? err.message : "压缩失败")
  } finally {
    setBusy(false)
    persistChat()
  }
}

try {
  const saved = JSON.parse(sessionStorage.getItem(CHAT_KEY) || "[]")
  if (Array.isArray(saved)) {
    for (const msg of saved) {
      if (!msg || (msg.role !== "user" && msg.role !== "assistant" && msg.role !== "system")) continue
      messages.push({ role: msg.role, content: typeof msg.content === "string" ? msg.content : "" })
    }
  }
} catch {
  /* 坏数据就从空对话开始 */
}
if (messages.length) {
  renderAll()
  setBusy(false)
}

$("composer").addEventListener("submit", (event) => {
  event.preventDefault()
  const text = $("input").value.trim()
  if (!text || busy) return
  $("input").value = ""
  send(text)
})

$("input").addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault()
    $("composer").requestSubmit()
  }
})

$("compact").addEventListener("click", compact)

async function refreshHealth() {
  const node = $("link-status")
  try {
    const res = await fetch("/agent-api/health")
    if (!res.ok) throw new Error()
    node.textContent = "Agent 已连接"
    node.className = "link ok"
  } catch {
    node.textContent = "Agent 未连接"
    node.className = "link down"
  }
}

refreshHealth()
