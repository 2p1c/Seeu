import { StubSpaceStatusStore } from "../db/stub.js";
import type { SpaceStatusStore } from "../db/types.js";
import { StatusTool } from "./status.js";
import type { Tool } from "./types.js";

export function createTools(store: SpaceStatusStore = new StubSpaceStatusStore()): Tool[] {
  return [new StatusTool(store)];
}
