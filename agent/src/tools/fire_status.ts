import type { Tool } from "./types.js";

export class FireStatusTool implements Tool {
  name = "fire_status";
  description = "Read the live fire/smoke monitor status and recent risk alerts. Call before answering fire safety monitoring questions. A stopped, stale or failed monitor cannot establish that the home is safe. Detections are suspected fire/smoke, not confirmed fires.";
  parameters = { type: "object" as const, properties: {} };

  async execute(): Promise<string> {
    const base = (process.env.ROOMIND_API ?? "http://127.0.0.1:8000").replace(/\/$/, "");
    try {
      const response = await fetch(`${base}/api/fire/status`, { signal: AbortSignal.timeout(5000) });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const raw: unknown = await response.json();
      if (!raw || typeof raw !== "object" || !("state" in raw)) throw new Error("Invalid monitor response");
      const data = raw as Record<string, unknown>;
      const events = Array.isArray(data.events) ? data.events : [];
      // Bound context and avoid recursively embedding previous Agent reviews.
      data.events = events.slice(-10).map(({ agent_review, ...event }: Record<string, unknown>) => event);
      return JSON.stringify(data);
    } catch (error) {
      return JSON.stringify({ available: false, reason: "monitor_unavailable", detail: String(error) });
    }
  }
}
