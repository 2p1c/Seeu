import type { SpaceStatusStore } from "../db/types.js";
import type { Tool } from "./types.js";

export class StatusTool implements Tool {
  name = "status";

  description =
    "Query the current household space status snapshot from the database. Returns detection time (detected_at) and objects (id, location, description). Call this before answering questions about what is in the space, where an object is, or when it was last detected. If no snapshot exists, say so; do not invent objects.";

  parameters = {
    type: "object" as const,
    properties: {},
  };

  constructor(private store: SpaceStatusStore) {}

  async execute(_args: Record<string, unknown>): Promise<string> {
    void _args;
    try {
      const snapshot = await this.store.getStatus();
      if (snapshot == null) {
        return JSON.stringify({ available: false, reason: "no_snapshot" });
      }
      return JSON.stringify({
        available: true,
        detected_at: snapshot.detected_at,
        objects: snapshot.objects,
      });
    } catch (e) {
      return `Error running status: ${e instanceof Error ? e.message : String(e)}`;
    }
  }
}
