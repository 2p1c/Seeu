import assert from "node:assert/strict";
import type { AddressInfo } from "node:net";
import { test } from "node:test";
import { createApp } from "../src/server.js";
import { FireStatusTool } from "../src/tools/fire_status.js";
import { makeAgent } from "./helpers.js";

test("fire status preserves stale/failure state and bounds prior reviews", async () => {
  const original = globalThis.fetch;
  globalThis.fetch = async () => new Response(JSON.stringify({
    state: "error", stale: true, error: "camera disconnected",
    events: Array.from({ length: 15 }, (_, id) => ({ id, type: "fire", agent_review: "old text" })),
  }), { status: 200 });
  try {
    const result = JSON.parse(await new FireStatusTool().execute());
    assert.equal(result.stale, true);
    assert.equal(result.state, "error");
    assert.equal(result.events.length, 10);
    assert.equal(result.events[0].id, 5);
    assert.equal(result.events[0].agent_review, undefined);
  } finally { globalThis.fetch = original; }
});

test("fire status network failure is unavailable, not safe", async () => {
  const original = globalThis.fetch;
  globalThis.fetch = async () => { throw new Error("offline"); };
  try {
    const result = JSON.parse(await new FireStatusTool().execute());
    assert.equal(result.available, false);
    assert.equal(result.reason, "monitor_unavailable");
  } finally { globalThis.fetch = original; }
});

test("proactive review validates observations and returns model analysis", async () => {
  const server = createApp(makeAgent({ turns: [{ content: "厨房疑似烟雾，持续5秒，待确认。" }] })).listen(0, "127.0.0.1");
  await new Promise<void>(resolve => server.once("listening", resolve));
  const url = `http://127.0.0.1:${(server.address() as AddressInfo).port}/fire/review`;
  try {
    for (const invalid of [{}, { semantic: "烟雾", events: [] }, { semantic: "烟雾", events: [null] }]) {
      const response = await fetch(url, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(invalid) });
      assert.equal(response.status, 400);
    }
    const response = await fetch(url, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({
      semantic: "灶台附近有疑似烟雾", events: [{ type: "smoke", location: "厨房", duration_seconds: 5 }], history: [],
    }) });
    assert.equal(response.status, 200);
    const result = await response.json() as { content: string };
    assert.match(result.content, /厨房疑似烟雾/);
  } finally {
    await new Promise<void>((resolve, reject) => server.close(error => error ? reject(error) : resolve()));
  }
});
