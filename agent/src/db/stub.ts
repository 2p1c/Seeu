import type { SpaceStatus, SpaceStatusStore } from "./types.js";

/** Placeholder until the database is designed. Always reports no snapshot. */
export class StubSpaceStatusStore implements SpaceStatusStore {
  async getStatus(): Promise<SpaceStatus | null> {
    return null;
  }
}
