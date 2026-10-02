const pages = [
  ["/", "流水线"],
  ["/yolo", "YOLO"],
  ["/sam", "SAM"],
  ["/dino", "DINO"],
  ["/dinov3", "DINOv3"],
  ["/vlm", "VLM"],
  ["/memory", "记忆库"],
  ["/agent", "Agent"],
]

const nav = document.getElementById("nav")
if (nav) {
  const here = location.pathname.replace(/\/$/, "") || "/"
  for (const [href, label] of pages) {
    const link = document.createElement("a")
    link.href = href
    link.textContent = label
    if (href === here) link.setAttribute("aria-current", "page")
    nav.append(link)
  }
}
