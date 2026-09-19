import type { SpaceStatus, SpaceStatusStore } from "./types.js";

export class InMemorySpaceStatusStore implements SpaceStatusStore {
  constructor(private snapshot: SpaceStatus | null = null) {}

  async getStatus(): Promise<SpaceStatus | null> {
    return this.snapshot;
  }

  set(snapshot: SpaceStatus | null): void {
    this.snapshot = snapshot;
  }
}
