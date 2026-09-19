import OpenAI from "openai";
import { IDENTITY, SYSTEM_PROMPT } from "./prompts.js";
import type { Tool } from "./tools/types.js";
import {
  compactPromptMessages,
  compactedMessages,
  NOTICE_COMPACTED,
  NOTICE_SKIPPED,
  partitionForCompact,
  stripTrailingCompact,
} from "./compact.js";

export type TokenUsage = {
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens: number;
};

export function emptyUsage(): TokenUsage {
  return { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 };
}

function asNonNegInt(value: unknown): number {
  if (typeof value !== "number" || !Number.isFinite(value)) return 0;
  return Math.max(0, Math.trunc(value));
}

export function readUsage(response: {
  usage?: {
    prompt_tokens?: number;
    completion_tokens?: number;
    total_tokens?: number;
  } | null;
}): TokenUsage {
  const u = response.usage;
  if (!u) return emptyUsage();
  const prompt_tokens = asNonNegInt(u.prompt_tokens);
  const completion_tokens = asNonNegInt(u.completion_tokens);
  const total_tokens = asNonNegInt(u.total_tokens) || prompt_tokens + completion_tokens;
  return { prompt_tokens, completion_tokens, total_tokens };
}

export function addUsage(a: TokenUsage, b: TokenUsage): TokenUsage {
  return {
    prompt_tokens: a.prompt_tokens + b.prompt_tokens,
    completion_tokens: a.completion_tokens + b.completion_tokens,
    total_tokens: a.total_tokens + b.total_tokens,
  };
}

type ChatMessage = OpenAI.ChatCompletionMessageParam;

export type LoopEvent =
  | {
      type: "llm";
      step: number;
      contentPreview: string;
      toolCalls: { id: string; name: string; arguments: string }[];
    }
  | {
      type: "tool_result";
      step: number;
      toolCallId: string;
      name: string;
      resultPreview: string;
    }
  | { type: "final"; step: number; content: string }
  | { type: "max_steps"; step: number; content: string };

export type LoopListener = (event: LoopEvent) => void;

export type RunOutcome =
  | { type: "final"; content: string; usage: TokenUsage }
  | { type: "max_steps"; content: string; usage: TokenUsage }
  | { type: "cancelled"; usage: TokenUsage };

export type LlmClient = {
  chat: {
    completions: {
      create: (
        body: {
          model: string;
          messages: ChatMessage[];
          tools?: OpenAI.ChatCompletionTool[];
          tool_choice?: "auto";
        },
        options?: { signal?: AbortSignal },
      ) => Promise<{
        choices: Array<{
          message: {
            content?: string | null;
            tool_calls?: OpenAI.ChatCompletionMessageToolCall[];
          };
        }>;
        usage?: {
          prompt_tokens?: number;
          completion_tokens?: number;
          total_tokens?: number;
        } | null;
      }>;
    };
  };
};

function isAbortError(e: unknown): boolean {
  return !!e && typeof e === "object" && (e as { name?: string }).name === "AbortError";
}

export type AgentDeps = {
  client?: LlmClient;
};

function preview(text: string, max = 500): string {
  if (!text) return "";
  return text.length > max ? `${text.slice(0, max)}…` : text;
}

function emitLoop(onEvent: LoopListener | undefined, event: LoopEvent): void {
  if (!onEvent) return;
  try {
    onEvent(event);
  } catch {
    // observers must not break the loop
  }
}

const IDENTITY_MAX_LENGTH = 4000;

function normalizeIdentity(
  identity: string | undefined,
  fallback: string,
): string | undefined {
  const raw = identity === undefined ? fallback : identity;
  const trimmed = raw.trim();
  if (!trimmed) return undefined;
  return trimmed.length > IDENTITY_MAX_LENGTH
    ? trimmed.slice(0, IDENTITY_MAX_LENGTH)
    : trimmed;
}

function assembleMessages(
  messages: ChatMessage[],
  identity: string | undefined,
  fallbackIdentity: string,
): ChatMessage[] {
  const assembled: ChatMessage[] = [{ role: "system", content: SYSTEM_PROMPT }];
  const id = normalizeIdentity(identity, fallbackIdentity);
  if (id) {
    assembled.push({ role: "system", content: id });
  }
  assembled.push(...messages);
  return assembled;
}

export class Agent {
  private client: LlmClient;
  private tools: Map<string, Tool>;
  private toolSchemas: OpenAI.ChatCompletionTool[];

  constructor(
    private model: string,
    tools: Tool[] = [],
    private maxSteps = 10,
    baseURL?: string,
    private identity = IDENTITY,
    deps: AgentDeps = {},
  ) {
    this.client =
      deps.client ??
      new OpenAI({
        apiKey: process.env.OPENAI_API_KEY,
        baseURL,
      });
    this.tools = new Map(tools.map((t) => [t.name, t]));
    this.toolSchemas = tools.map((t) => ({
      type: "function",
      function: {
        name: t.name,
        description: t.description,
        parameters: t.parameters,
      },
    }));
  }

  async run(task: string, onEvent?: LoopListener, signal?: AbortSignal): Promise<string> {
    const outcome = await this.runLoop(
      assembleMessages([{ role: "user", content: task }], undefined, this.identity),
      onEvent,
      signal,
    );
    if (outcome.type === "cancelled") return "Cancelled.";
    return outcome.content;
  }

  async runWithMessages(
    messages: ChatMessage[],
    onEvent?: LoopListener,
    identity?: string,
    signal?: AbortSignal,
  ): Promise<RunOutcome> {
    return this.runLoop(
      assembleMessages(messages, identity, this.identity),
      onEvent,
      signal,
    );
  }

  async compact(
    messages: ChatMessage[],
    signal?: AbortSignal,
  ): Promise<{
    compacted: ChatMessage[];
    skipped: boolean;
    notice: string;
    usage: TokenUsage;
  }> {
    const source = stripTrailingCompact(messages);
    const { old, recent, skipped } = partitionForCompact(source);
    if (skipped) {
      return {
        compacted: source,
        skipped: true,
        notice: NOTICE_SKIPPED,
        usage: emptyUsage(),
      };
    }
    let response: Awaited<ReturnType<LlmClient["chat"]["completions"]["create"]>>;
    try {
      response = await this.client.chat.completions.create(
        {
          model: this.model,
          messages: compactPromptMessages(old),
        },
        { signal },
      );
    } catch (e) {
      if (signal?.aborted || isAbortError(e)) {
        const err = new Error("This operation was aborted");
        err.name = "AbortError";
        throw err;
      }
      throw e;
    }
    const summary = response.choices[0]?.message?.content ?? "";
    return {
      compacted: compactedMessages(summary, recent),
      skipped: false,
      notice: NOTICE_COMPACTED,
      usage: readUsage(response),
    };
  }

  private cancelled(usage: TokenUsage = emptyUsage()): RunOutcome {
    return { type: "cancelled", usage };
  }

  private async runLoop(
    messages: ChatMessage[],
    onEvent?: LoopListener,
    signal?: AbortSignal,
  ): Promise<RunOutcome> {
    const emit = (event: LoopEvent) => emitLoop(onEvent, event);
    let usage = emptyUsage();

    for (let step = 0; step < this.maxSteps; step++) {
      if (signal?.aborted) return this.cancelled(usage);
      const stepNum = step + 1;
      let response: Awaited<ReturnType<LlmClient["chat"]["completions"]["create"]>>;
      try {
        response = await this.client.chat.completions.create(
          {
            model: this.model,
            messages,
            ...(this.toolSchemas.length > 0
              ? { tools: this.toolSchemas, tool_choice: "auto" as const }
              : {}),
          },
          { signal },
        );
      } catch (e) {
        if (signal?.aborted || isAbortError(e)) return this.cancelled(usage);
        throw e;
      }
      usage = addUsage(usage, readUsage(response));

      const msg = response.choices[0].message;

      if (msg.tool_calls && msg.tool_calls.length > 0) {
        emit({
          type: "llm",
          step: stepNum,
          contentPreview: preview(msg.content ?? ""),
          toolCalls: msg.tool_calls
            .filter((call) => call.type === "function")
            .map((call) => ({
              id: call.id,
              name: call.function.name,
              arguments: preview(call.function.arguments || ""),
            })),
        });

        messages.push({
          role: "assistant",
          content: msg.content,
          tool_calls: msg.tool_calls,
        });

        for (const call of msg.tool_calls) {
          if (call.type !== "function") continue;

          const tool = this.tools.get(call.function.name);
          if (!tool) {
            const unknown = `Unknown tool: ${call.function.name}`;
            messages.push({
              role: "tool",
              tool_call_id: call.id,
              content: unknown,
            });
            emit({
              type: "tool_result",
              step: stepNum,
              toolCallId: call.id,
              name: call.function.name,
              resultPreview: preview(unknown),
            });
            continue;
          }

          let args: Record<string, unknown> = {};
          try {
            args = JSON.parse(call.function.arguments || "{}");
          } catch {
            args = {};
          }

          let result: string;
          try {
            result = await tool.execute(args);
          } catch (e) {
            result = `Error executing tool "${tool.name}": ${
              e instanceof Error ? e.message : String(e)
            }`;
          }
          messages.push({ role: "tool", tool_call_id: call.id, content: result });
          emit({
            type: "tool_result",
            step: stepNum,
            toolCallId: call.id,
            name: tool.name,
            resultPreview: preview(result),
          });
        }
      } else {
        const content = msg.content ?? "No answer produced.";
        emit({ type: "final", step: stepNum, content });
        return { type: "final", content, usage };
      }
    }

    const exhausted = "Could not solve task: Maximum number of steps exceeded.";
    emit({ type: "max_steps", step: this.maxSteps, content: exhausted });
    return { type: "max_steps", content: exhausted, usage };
  }
}
