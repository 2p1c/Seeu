import { PerceptionSpaceStatusStore } from "../db/perception.js";
import type { SpaceStatusStore } from "../db/types.js";
import { StatusTool } from "./status.js";
import type { Tool } from "./types.js";

export function createTools(store: SpaceStatusStore = new PerceptionSpaceStatusStore()): Tool[] {
  return [new StatusTool(store)];
}
