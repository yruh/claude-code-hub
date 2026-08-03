import { describe, expect, it } from "vitest";
import {
  convertAnthropicUsage,
  createClaudeToOpenAIStreamTransform,
  stripVolatileCch,
  transformClaudeResponseToOpenAI,
  transformOpenAIRequestToClaude,
} from "./opencode-go-converter";

async function readStream(stream: ReadableStream<Uint8Array>): Promise<string> {
  return await new Response(stream).text();
}

describe("OpenCode Go converter", () => {
  describe("OpenAI request conversion", () => {
    it("stabilizes leading system content and appends later system reminders as user messages", () => {
      const input = {
        model: "claude-sonnet-4-5",
        messages: [
          {
            role: "system",
            content:
              "x-anthropic-billing-header: cc_version=2.1.177; cc_entrypoint=cli; cch=a1b2;\nYou are a coding assistant.",
          },
          { role: "user", content: "first" },
          { role: "SYSTEM", content: "The date changed." },
        ],
        max_completion_tokens: 4096,
        stream: true,
      };

      const output = transformOpenAIRequestToClaude(input);

      expect(output).toMatchObject({
        model: "claude-sonnet-4-5",
        max_tokens: 4096,
        stream: true,
        system: [
          {
            type: "text",
            text: "x-anthropic-billing-header: cc_version=2.1.177; cc_entrypoint=cli\nYou are a coding assistant.",
          },
        ],
        messages: [
          { role: "user", content: [{ type: "text", text: "first" }] },
          { role: "user", content: [{ type: "text", text: "The date changed." }] },
        ],
      });
      expect(input.messages[2].role).toBe("SYSTEM");
    });

    it("produces the same system prefix when only cch changes", () => {
      const convert = (cch: string) =>
        transformOpenAIRequestToClaude({
          model: "model",
          messages: [
            {
              role: "system",
              content: `x-anthropic-billing-header: cc_version=2.1.177; cc_entrypoint=cli; cch=${cch};\nStable`,
            },
            { role: "user", content: "hello" },
          ],
        });

      expect(convert("aaa").system).toEqual(convert("bbb").system);
      expect(JSON.stringify(convert("aaa").system)).not.toContain("cch=");
      expect(stripVolatileCch("a; cch=token; b")).toBe("a; b");
    });

    it("converts images, tools, tool calls, results, and tool choice", () => {
      const output = transformOpenAIRequestToClaude({
        model: "model",
        messages: [
          {
            role: "user",
            content: [
              { type: "text", text: "inspect" },
              { type: "image_url", image_url: { url: "data:image/png;base64,AAAA" } },
            ],
          },
          {
            role: "assistant",
            content: null,
            tool_calls: [
              {
                id: "call_1",
                type: "function",
                function: { name: "view", arguments: '{"path":"a.png"}' },
              },
            ],
          },
          { role: "tool", tool_call_id: "call_1", content: "ok" },
        ],
        tools: [
          {
            type: "function",
            function: {
              name: "view",
              description: "View a file",
              parameters: { type: "object", properties: { path: { type: "string" } } },
            },
          },
        ],
        tool_choice: { type: "function", function: { name: "view" } },
        parallel_tool_calls: false,
      });

      expect(output.messages).toEqual([
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
          content: [{ type: "tool_use", id: "call_1", name: "view", input: { path: "a.png" } }],
        },
        {
          role: "user",
          content: [
            {
              type: "tool_result",
              tool_use_id: "call_1",
              content: [{ type: "text", text: "ok" }],
            },
          ],
        },
      ]);
      expect(output.tools).toEqual([
        {
          name: "view",
          description: "View a file",
          input_schema: { type: "object", properties: { path: { type: "string" } } },
        },
      ]);
      expect(output.tool_choice).toEqual({
        type: "tool",
        name: "view",
        disable_parallel_tool_use: true,
      });
    });
  });

  describe("Anthropic response conversion", () => {
    it("converts text, thinking, tool use, stop reason, and cache usage", () => {
      const output = transformClaudeResponseToOpenAI({
        id: "msg_123",
        type: "message",
        model: "claude-sonnet-4-5",
        content: [
          { type: "thinking", thinking: "Need a tool." },
          { type: "text", text: "I will inspect it." },
          { type: "tool_use", id: "toolu_1", name: "read", input: { path: "a.ts" } },
        ],
        stop_reason: "tool_use",
        usage: {
          input_tokens: 30,
          output_tokens: 9,
          cache_creation_input_tokens: 10,
          cache_read_input_tokens: 80,
        },
      });

      expect(output).toMatchObject({
        id: "msg_123",
        object: "chat.completion",
        model: "claude-sonnet-4-5",
        choices: [
          {
            finish_reason: "tool_calls",
            message: {
              role: "assistant",
              content: "I will inspect it.",
              reasoning_content: "Need a tool.",
              tool_calls: [
                {
                  id: "toolu_1",
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
          total_tokens: 129,
          prompt_tokens_details: { cached_tokens: 80, cache_write_tokens: 10 },
        },
      });
    });

    it("converts disjoint Anthropic cache buckets to OpenAI prompt token totals", () => {
      expect(
        convertAnthropicUsage({
          input_tokens: 10,
          output_tokens: 5,
          cache_creation_input_tokens: 20,
          cache_read_input_tokens: 70,
        })
      ).toEqual({
        prompt_tokens: 100,
        completion_tokens: 5,
        total_tokens: 105,
        prompt_tokens_details: { cached_tokens: 70, cache_write_tokens: 20 },
      });
    });
  });

  describe("stream conversion", () => {
    it("buffers split frames and emits OpenAI thinking, text, tools, finish, and usage", async () => {
      const upstreamChunks: Uint8Array[] = [];
      const transform = createClaudeToOpenAIStreamTransform("fallback", (chunk) => {
        upstreamChunks.push(chunk);
      });
      const writer = transform.writable.getWriter();
      const outputPromise = readStream(transform.readable);
      const frames = [
        {
          type: "message_start",
          message: {
            id: "msg_stream",
            model: "claude-sonnet-4-5",
            usage: {
              input_tokens: 25,
              output_tokens: 0,
              cache_read_input_tokens: 75,
              cache_creation_input_tokens: 10,
            },
          },
        },
        {
          type: "content_block_delta",
          index: 0,
          delta: { type: "thinking_delta", thinking: "plan" },
        },
        {
          type: "content_block_delta",
          index: 1,
          delta: { type: "text_delta", text: "done" },
        },
        {
          type: "content_block_start",
          index: 2,
          content_block: { type: "tool_use", id: "toolu_1", name: "read", input: {} },
        },
        {
          type: "content_block_delta",
          index: 2,
          delta: { type: "input_json_delta", partial_json: '{"path":"a.ts"}' },
        },
        {
          type: "message_delta",
          delta: { stop_reason: "tool_use" },
          usage: { output_tokens: 12 },
        },
        { type: "message_stop" },
      ];
      const source = frames
        .map((frame) => `event: ${frame.type}\ndata: ${JSON.stringify(frame)}\n\n`)
        .join("");
      const split = Math.floor(source.length / 2) + 3;

      await writer.write(new TextEncoder().encode(source.slice(0, split)));
      await writer.write(new TextEncoder().encode(source.slice(split)));
      await writer.close();
      const output = await outputPromise;

      expect(upstreamChunks).toHaveLength(2);
      expect(output).toContain('"role":"assistant","content":""');
      expect(output).toContain('"reasoning_content":"plan"');
      expect(output).toContain('"content":"done"');
      expect(output).toContain('"id":"toolu_1","type":"function"');
      expect(output).toContain('"arguments":"{\\"path\\":\\"a.ts\\"}"');
      expect(output).toContain('"finish_reason":"tool_calls"');
      expect(output).toContain('"prompt_tokens":110,"completion_tokens":12,"total_tokens":122');
      expect(output.match(/data: \[DONE\]/g)).toHaveLength(1);
    });

    it("converts an Anthropic streaming error to an OpenAI error object", async () => {
      const transform = createClaudeToOpenAIStreamTransform();
      const writer = transform.writable.getWriter();
      const outputPromise = readStream(transform.readable);

      await writer.write(
        new TextEncoder().encode(
          'event: error\ndata: {"type":"error","error":{"type":"rate_limit_error","message":"slow down"}}\n\n'
        )
      );
      await writer.close();

      const output = await outputPromise;
      expect(output).toContain('"type":"rate_limit_error"');
      expect(output).toContain('"message":"slow down"');
      expect(output).not.toContain("[DONE]");
    });
  });
});
