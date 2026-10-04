# Agent

RoomMind 的对话进程。它看不到画面，也不能直接跑 DINO、SAM。用户问空间里有什么时，它只能调用工具，再根据工具返回的文字作答。

环境变量和启动命令见仓库根目录 README 的「Agent 环境变量」。在本目录执行：

```bash
npm install
npm test
npm run server    # http://127.0.0.1:8001
```

`npm test` 不访问真实模型，用脚本把模型回复写死。

## 一次请求怎么跑

`POST /complete` 的 body 是 `{ "messages": [...], "identity"?: "..." }`。`messages` 只保留对话，不要自己塞系统提示。

`server.ts` 把请求交给 `Agent.runWithMessages`。组装后的消息是：

1. `prompts.ts` 里的 `SYSTEM_PROMPT`：什么时候该调工具、什么时候停。
2. 身份。请求里的 `identity` 优先；没传就用 `IDENTITY`。空字符串表示不要身份。
3. 调用方传来的 `messages`。

然后进入循环，最多 10 步（构造 `Agent` 时的 `maxSteps`）：

1. 把已注册工具的 JSON Schema 和 `tool_choice: "auto"` 一起发给模型。
2. 模型返回 `tool_calls`：按顺序执行，把结果以 `role: "tool"` 追加进本次上下文，再进入下一步。
3. 模型只返回文字：循环结束，这就是最终回答。
4. 10 步仍在调工具：返回 `Could not solve task: Maximum number of steps exceeded.`

工具名不在注册表里时，不会抛错，而是把 `Unknown tool: ...` 交回模型。`execute` 抛错时同样变成一段错误文本交回模型。客户端断开连接会中止循环，HTTP 状态是 `499`。

`/complete` 的响应只有最终文字和这一轮的 token 用量。页面下一轮只带上用户和助手的文字，工具调用过程不会留在页面历史里。终端里的 `[loop]` 日志能看到每一步；`AGENT_LOOP_LOG=0` 关掉。

`POST /compact` 在用户轮次超过 10 轮时，把更早的对话总结成一条 system 消息，最近 10 轮原样保留。不足 10 轮则跳过，不调用模型。

`GET /health` 只表示进程在听。

## 目录

```text
src/server.ts          HTTP。启动时 createTools()，默认接 StubSpaceStatusStore
src/agent.ts           循环、拼消息、压缩
src/prompts.ts         系统提示和身份
src/compact.ts         保留最近 10 轮，更早的交给模型总结
src/tools/types.ts     Tool 接口
src/tools/index.ts     工具注册表
src/tools/status.ts    目前唯一的工具
src/db/types.ts        空间快照的形状
src/db/stub.ts         正式进程用的空实现，永远没有快照
src/db/memory.ts       测试用的内存实现
public/                对话页
test/                  用 scriptedClient 假装模型
```

## 现有工具

`status` 没有参数。它调用 `SpaceStatusStore.getStatus()`：

| 返回 | 含义 |
| --- | --- |
| `null` | `{ "available": false, "reason": "no_snapshot" }`。当前正式进程一直是这个结果。 |
| 快照 | `{ "available": true, "detected_at", "objects" }`。物体含 `id`、`location`、`description`。 |
| 抛错 | 文本 `Error running status: ...`，不把异常抛出循环。 |

身份提示要求：问空间事实之前必须先调 `status`；没有快照或列表里没有该物体时如实说，不要编造。

数据库接上之后，实现 `SpaceStatusStore`，在 `server.ts` 里传给 `createTools(store)`。不要改 `status` 的返回形状，除非同时改 `IDENTITY` 和测试。

## 添加工具

工具是一个类，实现 `src/tools/types.ts` 的 `Tool`：

```ts
import type { Tool } from "./types.js";

export class ClockTool implements Tool {
  name = "clock";
  description = "Return the current time as an ISO string. Call this when the user asks what time it is.";
  parameters = {
    type: "object" as const,
    properties: {},
  };

  async execute(_args: Record<string, unknown>): Promise<string> {
    return new Date().toISOString();
  }
}
```

需要参数时写进 `properties`，并列出 `required`。这里的 schema 只支持 `type` 和 `description`，不是完整 JSON Schema。`execute` 必须返回字符串；结构化结果用 `JSON.stringify`。

然后在 `createTools` 里放进数组：

```ts
return [new StatusTool(store), new ClockTool()];
```

模型是否会调用，取决于 `description` 写不写清楚。如果这是回答某类问题的必经步骤，再在 `prompts.ts` 的 `IDENTITY` 里写明，和 `status` 一样。只改描述、不改身份，模型可能不用它。

加一个测试，放在 `test/`。用 `test/helpers.ts` 的 `scriptedClient` 规定模型先返回 `tool_calls`，再返回最终文字，不要在测试里请求真实接口：

```ts
const agent = makeAgent({
  tools: createTools(),
  turns: [
    { tool_calls: [{ id: "c1", name: "clock", arguments: "{}" }] },
    { content: "现在是测试时间。" },
  ],
});
```

`npm test` 通过后再跑 `npm run server`，在页面上问一句该触发新工具的话，看终端 `[loop]` 里有没有 `tool_result`。
