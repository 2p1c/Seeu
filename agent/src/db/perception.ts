import type { SpaceStatus, SpaceStatusStore } from "./types.js";

type LatestFrame = {
  captured_at: string;
  objects: { id: number; label: string; position: string; description: string }[];
};

/** 物体记忆库归感知服务所有，这里通过 GET /api/memory/latest 读最新画面。 */
export class PerceptionSpaceStatusStore implements SpaceStatusStore {
  constructor(
    private apiBase = process.env.ROOMIND_API ?? "http://127.0.0.1:8000",
    private fetchImpl: typeof fetch = fetch,
  ) {}

  async getStatus(): Promise<SpaceStatus | null> {
    const url = `${this.apiBase.replace(/\/$/, "")}/api/memory/latest`;
    const res = await this.fetchImpl(url, { signal: AbortSignal.timeout(5000) });
    if (res.status === 404) return null;
    if (!res.ok) throw new Error(`perception ${res.status} ${url}`);
    const frame = (await res.json()) as LatestFrame;
    return {
      detected_at: frame.captured_at,
      objects: frame.objects.map((o) => ({
        id: `${o.label}-${o.id}`,
        location: o.position,
        description: o.description,
      })),
    };
  }
}
