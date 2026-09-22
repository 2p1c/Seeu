import dotenv from "dotenv";
import { fileURLToPath } from "node:url";
import path from "node:path";
import express from "express";
import type { Request, Response } from "express";

import { Agent } from "./agent.js";
import type { LoopListener, RunOutcome, TokenUsage } from "./agent.js";
import { createTools } from "./tools/index.js";

dotenv.config({
  path: path.join(path.dirname(fileURLToPath(import.meta.url)), "../.env"),
});

const isProd = process.env.NODE_ENV === "production";
const LOOP_LOG =
  process.env.AGENT_LOOP_LOG === "1" ||
  (process.env.AGENT_LOOP_LOG !== "0" && !isProd);

function classifyError(e: unknown): { status: number; body: { error: string; detail: string } } {
  const detail = e instanceof Error ? e.message : String(e);
  if (e && typeof e === "object" && "status" in e) {
    const status = (e as { status?: number }).status;
    if (typeof status === "number") {
      return { status: 502, body: { error: "llm_error", detail: `${status}: ${detail}` } };
    }
  }
  if (/timeout|timed out|aborted/i.test(detail)) {
    return { status: 504, body: { error: "timeout", detail } };
  }
  return { status: 500, body: { error: "internal", detail } };
}

function parseBody(
  req: Request,
): { ok: true; messages: unknown[]; identity?: string } | { ok: false; detail: string } {
  const body = req.body;
  if (!body || typeof body !== "object") return { ok: false, detail: "request body must be a JSON object" };
  const messages = (body as { messages?: unknown }).messages;
  if (!Array.isArray(messages)) return { ok: false, detail: "messages must be an array" };
  const identity = (body as { identity?: unknown }).identity;
  if (identity !== undefined && typeof identity !== "string") {
    return { ok: false, detail: "identity must be a string" };
  }
  return { ok: true, messages, identity };
}

function requestSignal(req: Request, res: Response): AbortSignal {
  const ac = new AbortController();
  res.on("close", () => {
    if (!res.writableEnded && !ac.signal.aborted) ac.abort();
  });
  return ac.signal;
}

function usagePayload(usage: TokenUsage): TokenUsage {
  return {
    prompt_tokens: usage.prompt_tokens,
    completion_tokens: usage.completion_tokens,
    total_tokens: usage.total_tokens,
  };
}

function sendJsonOutcome(res: Response, outcome: RunOutcome): void {
  if (outcome.type === "cancelled") {
    res.status(499).json({ error: "cancelled", usage: usagePayload(outcome.usage) });
    return;
  }
  res.json({
    role: "assistant",
    content: outcome.content,
    usage: usagePayload(outcome.usage),
  });
}

function loopLogger(): LoopListener | undefined {
  if (!LOOP_LOG) return undefined;
  return (evt) => {
    console.log("[loop]", JSON.stringify(evt));
  };
}

export function createApp(agent: Agent): express.Express {
  const app = express();
  app.use(express.json({ limit: "1mb" }));

  app.get("/health", (_req: Request, res: Response) => {
    res.json({ status: "ok" });
  });

  app.post("/compact", async (req: Request, res: Response) => {
    const parsed = parseBody(req);
    if (!parsed.ok) {
      res.status(400).json({ error: "bad_request", detail: parsed.detail });
      return;
    }
    try {
      const result = await agent.compact(
        parsed.messages as never,
        requestSignal(req, res),
      );
      res.json({
        compacted: result.compacted,
        skipped: result.skipped,
        notice: result.notice,
        usage: usagePayload(result.usage),
      });
    } catch (e) {
      const { status, body } = classifyError(e);
      res.status(status).json(body);
    }
  });

  app.post("/complete", async (req: Request, res: Response) => {
    const parsed = parseBody(req);
    if (!parsed.ok) {
      res.status(400).json({ error: "bad_request", detail: parsed.detail });
      return;
    }
    try {
      const outcome = await agent.runWithMessages(
        parsed.messages as never,
        loopLogger(),
        parsed.identity,
        requestSignal(req, res),
      );
      sendJsonOutcome(res, outcome);
    } catch (e) {
      const { status, body } = classifyError(e);
      res.status(status).json(body);
    }
  });

  const publicDir = path.join(path.dirname(fileURLToPath(import.meta.url)), "../public");
  app.use(express.static(publicDir));

  return app;
}

function isDirectRun(metaUrl: string): boolean {
  const self = fileURLToPath(metaUrl);
  const argv1 = process.argv[1];
  if (!argv1) return false;
  return path.resolve(argv1) === self;
}

if (isDirectRun(import.meta.url)) {
  const model = process.env.MODEL;
  if (!model) {
    console.error("MODEL environment variable is not set.");
    process.exit(1);
  }
  const tools = createTools();
  const agent = new Agent(model, tools, undefined, process.env.OPENAI_BASE_URL);
  const PORT = Number(process.env.PORT) || 8001;
  createApp(agent).listen(PORT, () => {
    console.log(`Agent HTTP server listening on :${PORT}`);
  });
}
