import { describe, expect, it } from "vitest";
import {
  convertOpenAIUsage,
  createOpenAIToClaudeStreamTransform,
  stripVolatileCch,
  transformClaudeRequestToOpenAI,
  transformOpenAIResponseToClaude,
} from "./opencode-go-converter";

async function readStream(stream: ReadableStream<Uint8Array>): Promise<string> {
  return await new Response(stream).text();
}

describe("OpenCode Go converter", () => {
  describe("Claude request conversion", () => {
    it("removes a billing-only system block and rewrites historical system messages", () => {
      const input = {
        model: "claude-sonnet-4-5",
        system: [
          {
            type: "text",
            text: "x-anthropic-billing-header: cc_version=2.1.177.c0b; cc_entrypoint=cli; cch=a1b2;",
          },
          { type: "text", text: "You are a coding assistant." },
        ],
        messages: [
          { role: "user", content: "first" },
          { role: "SYSTEM", content: "The date changed." },
        ],
        max_tokens: 4096,
        stream: true,
      };

      const output = transformClaudeRequestToOpenAI(input);

      expect(output).toMatchObject({
        model: "claude-sonnet-4-5",
        max_tokens: 4096,
        stream: true,
        stream_options: { include_usage: true },
        messages: [
          { role: "system", content: "You are a coding assistant." },
          { role: "user", content: "first" },
          { role: "user", content: "The date changed." },
        ],
      });
      expect(input.messages[1].role).toBe("SYSTEM");
    });

    it("strips only volatile cch values from mixed system content", () => {
      const first = transformClaudeRequestToOpenAI({
        model: "model",
        system:
          "x-anthropic-billing-header: cc_version=2.1.177; cc_entrypoint=cli; cch=aaa;\nStable prompt",
        messages: [],
      });
      const second = transformClaudeRequestToOpenAI({
        model: "model",
        system:
          "x-anthropic-billing-header: cc_version=2.1.177; cc_entrypoint=cli; cch=bbb;\nStable prompt",
        messages: [],
      });

      expect(first.messages).toEqual(second.messages);
      expect(JSON.stringify(first.messages)).not.toContain("cch=");
      expect(JSON.stringify(first.messages)).toContain("cc_version=2.1.177");
      expect(stripVolatileCch("a; cch=token; b")).toBe("a; b");
    });

    it("converts images, tools, tool calls, and tool results", () => {
      const output = transformClaudeRequestToOpenAI({
        model: "model",
        messages: [
          {
            role: "user",
            content: [
              { type: "text", text: "inspect" },
              {
                type: "image",
                source: { type: "base64", media_type: "image/png", data: "AAAA" },
              },
            ],
          },
          {
            role: "assistant",
            content: [
              { type: "thinking", thinking: "I should inspect it." },
              {
                type: "tool_use",
                id: "toolu_1",
                name: "view",
                input: { path: "a.png" },
                thinking: "Use the image viewer.",
              },
            ],
          },
          {
            role: "user",
            content: [{ type: "tool_result", tool_use_id: "toolu_1", content: "ok" }],
          },
        ],
        tools: [
          {
            name: "view",
            description: "View a file",
            input_schema: { type: "object", properties: { path: { type: "string" } } },
          },
        ],
        tool_choice: { type: "tool", name: "view", disable_parallel_tool_use: true },
      });

      expect(output.messages).toEqual([
        {
          role: "user",
          content: [
            { type: "text", text: "inspect" },
            {
              type: "image_url",
              image_url: { url: "data:image/png;base64,AAAA", detail: "auto" },
            },
          ],
        },
        {
          role: "assistant",
          content: null,
          reasoning_content: "I should inspect it.Use the image viewer.",
          tool_calls: [
            {
              id: "toolu_1",
              type: "function",
              function: { name: "view", arguments: '{"path":"a.png"}' },
            },
          ],
        },
        { role: "tool", content: "ok", tool_call_id: "toolu_1" },
      ]);
      expect(output.tools).toEqual([
        {
          type: "function",
          function: {
            name: "view",
            description: "View a file",
            parameters: { type: "object", properties: { path: { type: "string" } } },
          },
        },
      ]);
      expect(output.tool_choice).toEqual({ type: "function", function: { name: "view" } });
      expect(output.parallel_tool_calls).toBe(false);
    });
  });

  describe("OpenAI response conversion", () => {
    it("converts text, reasoning, tool calls, stop reason, and cache usage", () => {
      const output = transformOpenAIResponseToClaude({
        id: "chatcmpl-123",
        model: "kimi-k2.5",
        choices: [
          {
            finish_reason: "tool_calls",
            message: {
              role: "assistant",
              content: "I will inspect it.",
              reasoning_content: "Need a tool.",
              tool_calls: [
                {
                  id: "call_1",
                  type: "function",
                  function: { name: "read", arguments: '{"path":"a.ts"}' },
                },
              ],
            },
          },
        ],
        usage: {
          prompt_tokens: 120,
          completion_tokens: 9,
          prompt_tokens_details: { cached_tokens: 80, cache_write_tokens: 10 },
        },
      });

      expect(output).toMatchObject({
        id: "msg_chatcmpl-123",
        type: "message",
        role: "assistant",
        model: "kimi-k2.5",
        stop_reason: "tool_use",
        usage: {
          input_tokens: 30,
          output_tokens: 9,
          cache_creation_input_tokens: 10,
          cache_read_input_tokens: 80,
        },
      });
      expect(output.content).toEqual([
        { type: "thinking", thinking: "Need a tool.", signature: "opencode-go" },
        { type: "text", text: "I will inspect it." },
        { type: "tool_use", id: "call_1", name: "read", input: { path: "a.ts" } },
      ]);
    });

    it("supports OpenCode cache hit and miss usage fields", () => {
      expect(
        convertOpenAIUsage({
          prompt_tokens: 100,
          completion_tokens: 5,
          prompt_cache_hit_tokens: 70,
          prompt_cache_miss_tokens: 20,
        })
      ).toEqual({
        input_tokens: 10,
        output_tokens: 5,
        cache_creation_input_tokens: 20,
        cache_read_input_tokens: 70,
      });
    });
  });

  describe("stream conversion", () => {
    it("buffers split frames and emits Claude thinking, text, tools, and final usage", async () => {
      const upstreamChunks: Uint8Array[] = [];
      const transform = createOpenAIToClaudeStreamTransform("fallback", (chunk) => {
        upstreamChunks.push(chunk);
      });
      const writer = transform.writable.getWriter();
      const outputPromise = readStream(transform.readable);
      const encoder = new TextEncoder();
      const frames = [
        {
          id: "chatcmpl-stream",
          model: "kimi-k2.5",
          choices: [
            { delta: { role: "assistant", reasoning_content: "plan" }, finish_reason: null },
          ],
        },
        {
          id: "chatcmpl-stream",
          choices: [{ delta: { content: "done" }, finish_reason: null }],
        },
        {
          id: "chatcmpl-stream",
          choices: [
            {
              delta: {
                tool_calls: [
                  {
                    index: 0,
                    id: "call_1",
                    function: { name: "read", arguments: '{"path"' },
                  },
                ],
              },
              finish_reason: null,
            },
          ],
        },
        {
          id: "chatcmpl-stream",
          choices: [
            {
              delta: { tool_calls: [{ index: 0, function: { arguments: ':"a.ts"}' } }] },
              finish_reason: "tool_calls",
            },
          ],
        },
        {
          id: "chatcmpl-stream",
          choices: [],
          usage: {
            prompt_tokens: 100,
            completion_tokens: 12,
            prompt_tokens_details: { cached_tokens: 75 },
          },
        },
      ];
      const source = `${frames.map((frame) => `data: ${JSON.stringify(frame)}\n\n`).join("")}data: [DONE]\n\n`;
      const split = Math.floor(source.length / 2) + 3;

      await writer.write(encoder.encode(source.slice(0, split)));
      await writer.write(encoder.encode(source.slice(split)));
      await writer.close();
      const output = await outputPromise;

      expect(upstreamChunks).toHaveLength(2);
      expect(output).toContain("event: message_start");
      expect(output).toContain('"type":"thinking_delta","thinking":"plan"');
      expect(output).toContain('"type":"signature_delta","signature":"opencode-go"');
      expect(output).toContain('"type":"text_delta","text":"done"');
      expect(output).toContain('"type":"tool_use","id":"call_1","name":"read"');
      expect(output).toContain('"partial_json":"{\\"path\\""');
      expect(output).toContain('"partial_json":":\\"a.ts\\"}"');
      expect(output).toContain('"stop_reason":"tool_use"');
      expect(output).toContain(
        '"input_tokens":25,"output_tokens":12,"cache_creation_input_tokens":0,"cache_read_input_tokens":75'
      );
      expect(output.match(/event: message_stop/g)).toHaveLength(1);
      expect(output).not.toContain("[DONE]");
    });

    it("converts an upstream streaming error to a Claude error event", async () => {
      const transform = createOpenAIToClaudeStreamTransform();
      const writer = transform.writable.getWriter();
      const outputPromise = readStream(transform.readable);

      await writer.write(
        new TextEncoder().encode(
          'data: {"error":{"type":"rate_limit_error","message":"slow down"}}\n\n'
        )
      );
      await writer.close();

      const output = await outputPromise;
      expect(output).toContain("event: error");
      expect(output).toContain('"type":"rate_limit_error","message":"slow down"');
      expect(output).not.toContain("event: message_stop");
    });
  });
});
