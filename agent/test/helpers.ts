import type OpenAI from "openai";
import { Agent, type LlmClient, type TokenUsage } from "../src/agent.js";
import type { Tool } from "../src/tools/types.js";

export type ScriptTurn =
  | { content: string; usage?: TokenUsage }
  | {
      content?: string | null;
      tool_calls: { id: string; name: string; arguments: string }[];
      usage?: TokenUsage;
    };

async function abortableDelay(ms: number, signal?: AbortSignal): Promise<void> {
  if (signal?.aborted) {
    const err = new Error("This operation was aborted");
    err.name = "AbortError";
    throw err;
  }
  if (ms <= 0) return;
  await new Promise<void>((resolve, reject) => {
    const t = setTimeout(() => {
      signal?.removeEventListener("abort", onAbort);
      resolve();
    }, ms);
    const onAbort = () => {
      clearTimeout(t);
      const err = new Error("This operation was aborted");
      err.name = "AbortError";
      reject(err);
    };
    signal?.addEventListener("abort", onAbort, { once: true });
  });
}

export function scriptedClient(turns: ScriptTurn[], delayMs = 0): LlmClient {
  let i = 0;
  return {
    chat: {
      completions: {
        async create(_body, options) {
          await abortableDelay(delayMs, options?.signal);
          const turn = turns[i++];
          if (!turn) throw new Error("scripted LLM has no remaining turns");
          if ("tool_calls" in turn && turn.tool_calls) {
            return {
              choices: [
                {
                  message: {
                    content: turn.content ?? null,
                    tool_calls: turn.tool_calls.map((tc) => ({
                      id: tc.id,
                      type: "function" as const,
                      function: { name: tc.name, arguments: tc.arguments },
                    })),
                  },
                },
              ],
              usage: turn.usage,
            };
          }
          return { choices: [{ message: { content: turn.content } }], usage: turn.usage };
        },
      },
    },
  };
}

export function makeAgent(opts: {
  turns: ScriptTurn[];
  tools?: Tool[];
  maxSteps?: number;
  delayMs?: number;
}): Agent {
  return new Agent(
    "test-model",
    opts.tools ?? [],
    opts.maxSteps ?? 10,
    undefined,
    "",
    {
      client: scriptedClient(opts.turns, opts.delayMs ?? 0),
    },
  );
}

export type ChatMessage = OpenAI.ChatCompletionMessageParam;
