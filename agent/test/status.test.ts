import assert from "node:assert/strict";
import { test } from "node:test";
import { InMemorySpaceStatusStore } from "../src/db/memory.js";
import { StubSpaceStatusStore } from "../src/db/stub.js";
import { StatusTool } from "../src/tools/status.js";
import { createTools } from "../src/tools/index.js";
import { makeAgent } from "./helpers.js";

test("stub store reports no snapshot", async () => {
  const tool = new StatusTool(new StubSpaceStatusStore());
  const raw = await tool.execute({});
  assert.deepEqual(JSON.parse(raw), { available: false, reason: "no_snapshot" });
});

test("status returns detected_at and objects from the store", async () => {
  const store = new InMemorySpaceStatusStore({
    detected_at: "2026-09-19T10:00:00.000Z",
    objects: [
      {
        id: "cup-1",
        location: "dining table",
        description: "white ceramic mug",
      },
    ],
  });
  const tool = new StatusTool(store);
  const raw = await tool.execute({});
  assert.deepEqual(JSON.parse(raw), {
    available: true,
    detected_at: "2026-09-19T10:00:00.000Z",
    objects: [
      {
        id: "cup-1",
        location: "dining table",
        description: "white ceramic mug",
      },
    ],
  });
});

test("status store errors are returned as text, not thrown", async () => {
  const tool = new StatusTool({
    async getStatus() {
      throw new Error("db down");
    },
  });
  const raw = await tool.execute({});
  assert.match(raw, /Error running status: db down/);
});

test("createTools registers only status", () => {
  const tools = createTools();
  assert.deepEqual(tools.map((t) => t.name), ["status"]);
});

test("agent answers from status snapshot and does not invent missing objects", async () => {
  const store = new InMemorySpaceStatusStore({
    detected_at: "2026-09-19T10:00:00.000Z",
    objects: [
      { id: "cup-1", location: "dining table", description: "white ceramic mug" },
    ],
  });
  const events: string[] = [];
  const agent = makeAgent({
    tools: createTools(store),
    turns: [
      { tool_calls: [{ id: "c1", name: "status", arguments: "{}" }] },
      { content: "餐桌上有一只白色陶瓷杯（cup-1），没有查到钥匙。" },
    ],
  });
  const outcome = await agent.runWithMessages(
    [{ role: "user", content: "杯子在哪？钥匙呢？" }],
    (evt) => events.push(evt.type),
  );
  assert.equal(outcome.type, "final");
  if (outcome.type !== "final") return;
  assert.match(outcome.content, /陶瓷杯/);
  assert.match(outcome.content, /没有查到钥匙/);
  assert.deepEqual(events, ["llm", "tool_result", "final"]);
});

test("agent reports no snapshot when the database is empty", async () => {
  const agent = makeAgent({
    tools: createTools(new StubSpaceStatusStore()),
    turns: [
      { tool_calls: [{ id: "c1", name: "status", arguments: "{}" }] },
      { content: "还没有空间状态快照，无法确认当前有哪些物体。" },
    ],
  });
  const outcome = await agent.runWithMessages([
    { role: "user", content: "客厅现在有什么？" },
  ]);
  assert.equal(outcome.type, "final");
  if (outcome.type !== "final") return;
  assert.match(outcome.content, /没有/);
});
