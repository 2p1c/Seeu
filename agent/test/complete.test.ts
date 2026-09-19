import assert from "node:assert/strict";
import type { AddressInfo } from "node:net";
import { test } from "node:test";
import { addUsage, emptyUsage, readUsage, Agent } from "../src/agent.js";
import { createApp } from "../src/server.js";
import { createTools } from "../src/tools/index.js";
import { InMemorySpaceStatusStore } from "../src/db/memory.js";
import { makeAgent, scriptedClient } from "./helpers.js";
import type { LlmClient } from "../src/agent.js";

async function listen(agent: Agent): Promise<{ base: string; close: () => Promise<void> }> {
  const app = createApp(agent);
  const server = app.listen(0, "127.0.0.1");
  await new Promise<void>((resolve) => server.once("listening", () => resolve()));
  const { port } = server.address() as AddressInfo;
  return {
    base: `http://127.0.0.1:${port}`,
    close: () => new Promise((resolve, reject) => server.close((e) => (e ? reject(e) : resolve()))),
  };
}

test("readUsage treats missing usage as zeros", () => {
  assert.deepEqual(readUsage({}), emptyUsage());
  assert.deepEqual(readUsage({ usage: null }), emptyUsage());
});

test("addUsage sums each field", () => {
  assert.deepEqual(
    addUsage(
      { prompt_tokens: 10, completion_tokens: 2, total_tokens: 12 },
      { prompt_tokens: 3, completion_tokens: 4, total_tokens: 7 },
    ),
    { prompt_tokens: 13, completion_tokens: 6, total_tokens: 19 },
  );
});

test("GET /health", async () => {
  const { base, close } = await listen(makeAgent({ turns: [{ content: "unused" }] }));
  try {
    const res = await fetch(`${base}/health`);
    assert.equal(res.status, 200);
    assert.deepEqual(await res.json(), { status: "ok" });
  } finally {
    await close();
  }
});

test("POST /complete rejects missing messages", async () => {
  const { base, close } = await listen(makeAgent({ turns: [{ content: "unused" }] }));
  try {
    const res = await fetch(`${base}/complete`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({}),
    });
    assert.equal(res.status, 400);
    const body = (await res.json()) as { error: string };
    assert.equal(body.error, "bad_request");
  } finally {
    await close();
  }
});

test("POST /complete JSON includes this run's usage", async () => {
  const agent = makeAgent({
    turns: [
      {
        content: "hello",
        usage: { prompt_tokens: 11, completion_tokens: 5, total_tokens: 16 },
      },
    ],
  });
  const { base, close } = await listen(agent);
  try {
    const res = await fetch(`${base}/complete`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ messages: [{ role: "user", content: "hi" }] }),
    });
    assert.equal(res.status, 200);
    assert.deepEqual(await res.json(), {
      role: "assistant",
      content: "hello",
      usage: { prompt_tokens: 11, completion_tokens: 5, total_tokens: 16 },
    });
  } finally {
    await close();
  }
});

test("POST /complete runs status then answers from the snapshot", async () => {
  const store = new InMemorySpaceStatusStore({
    detected_at: "2026-09-19T10:00:00.000Z",
    objects: [{ id: "cup-1", location: "dining table", description: "white ceramic mug" }],
  });
  const agent = makeAgent({
    tools: createTools(store),
    turns: [
      {
        tool_calls: [{ id: "c1", name: "status", arguments: "{}" }],
        usage: { prompt_tokens: 100, completion_tokens: 20, total_tokens: 120 },
      },
      {
        content: "餐桌上有一只白色陶瓷杯。",
        usage: { prompt_tokens: 140, completion_tokens: 10, total_tokens: 150 },
      },
    ],
  });
  const { base, close } = await listen(agent);
  try {
    const res = await fetch(`${base}/complete`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ messages: [{ role: "user", content: "杯子在哪？" }] }),
    });
    assert.equal(res.status, 200);
    assert.deepEqual(await res.json(), {
      role: "assistant",
      content: "餐桌上有一只白色陶瓷杯。",
      usage: { prompt_tokens: 240, completion_tokens: 30, total_tokens: 270 },
    });
  } finally {
    await close();
  }
});

test("unknown tool result is fed back instead of crashing", async () => {
  const agent = makeAgent({
    turns: [
      { tool_calls: [{ id: "c1", name: "nope", arguments: "{}" }] },
      { content: "I cannot use that tool." },
    ],
  });
  const outcome = await agent.runWithMessages([{ role: "user", content: "hi" }]);
  assert.equal(outcome.type, "final");
  if (outcome.type !== "final") return;
  assert.equal(outcome.content, "I cannot use that tool.");
});

test("aborted signal before loop does not call LLM", async () => {
  let calls = 0;
  const inner = scriptedClient([{ content: "should not run" }]);
  const client: LlmClient = {
    chat: {
      completions: {
        async create(body, options) {
          calls += 1;
          return inner.chat.completions.create(body, options);
        },
      },
    },
  };
  const ac = new AbortController();
  ac.abort();
  const agent = new Agent("test-model", [], 10, undefined, "", { client });
  const outcome = await agent.runWithMessages(
    [{ role: "user", content: "hi" }],
    undefined,
    "",
    ac.signal,
  );
  assert.equal(outcome.type, "cancelled");
  assert.equal(calls, 0);
});
