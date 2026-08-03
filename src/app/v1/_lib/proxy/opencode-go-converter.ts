type JsonObject = Record<string, unknown>;

interface ClaudeUsage {
  input_tokens: number;
  output_tokens: number;
  cache_creation_input_tokens: number;
  cache_read_input_tokens: number;
}

interface StreamToolState {
  blockIndex: number;
  id: string;
  name: string;
  started: boolean;
  pendingArguments: string;
}

const BILLING_HEADER_ONLY = /^\s*x-anthropic-billing-header\s*:[^\r\n]*\s*$/i;
const THINKING_SIGNATURE = "opencode-go";

function isRecord(value: unknown): value is JsonObject {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function firstNumber(...values: unknown[]): number {
  for (const value of values) {
    if (typeof value === "number" && Number.isFinite(value)) {
      return value;
    }
  }
  return 0;
}

function safeJsonStringify(value: unknown): string {
  try {
    return JSON.stringify(value) ?? "{}";
  } catch {
    return "{}";
  }
}

function parseToolArguments(value: unknown): JsonObject {
  if (isRecord(value)) {
    return value;
  }
  if (typeof value !== "string" || value.length === 0) {
    return {};
  }

  try {
    const parsed = JSON.parse(value) as unknown;
    return isRecord(parsed) ? parsed : { value: parsed };
  } catch {
    return {};
  }
}

export function stripVolatileCch(text: string): string {
  return text
    .replace(/(^|;)([ \t]*)cch\s*=\s*[^;\s\r\n]+(?:[ \t]*;)?/gim, (_match, lead) =>
      lead === ";" ? ";" : ""
    )
    .replace(/;[ \t]*;/g, ";")
    .replace(/:[ \t]*;/g, ":")
    .replace(/;[ \t]*(?=\r?$)/gm, "");
}

function normalizeSystemText(system: unknown): string | null {
  if (typeof system === "string") {
    if (BILLING_HEADER_ONLY.test(system)) {
      return null;
    }
    const stable = stripVolatileCch(system);
    return stable.length > 0 ? stable : null;
  }

  if (!Array.isArray(system)) {
    return null;
  }

  const parts: string[] = [];
  for (const rawPart of system) {
    if (!isRecord(rawPart) || rawPart.type !== "text" || typeof rawPart.text !== "string") {
      continue;
    }
    if (BILLING_HEADER_ONLY.test(rawPart.text)) {
      continue;
    }
    const stable = stripVolatileCch(rawPart.text);
    if (stable.length > 0) {
      parts.push(stable);
    }
  }

  return parts.length > 0 ? parts.join("\n\n") : null;
}

function normalizeRole(role: unknown): string {
  if (typeof role !== "string") {
    return "user";
  }
  return role.trim().toLowerCase() === "system" ? "user" : role.trim().toLowerCase();
}

function convertImageBlock(block: JsonObject): JsonObject | null {
  const source = isRecord(block.source) ? block.source : null;
  if (!source || typeof source.type !== "string") {
    return null;
  }

  let url = "";
  if (source.type === "base64" && typeof source.data === "string") {
    const mediaType =
      typeof source.media_type === "string" ? source.media_type : "application/octet-stream";
    url = `data:${mediaType};base64,${source.data}`;
  } else if (source.type === "url" && typeof source.url === "string") {
    url = source.url;
  }

  return url
    ? {
        type: "image_url",
        image_url: { url, detail: "auto" },
      }
    : null;
}

function toolResultText(content: unknown): string {
  if (typeof content === "string") {
    return content;
  }
  if (!Array.isArray(content)) {
    return content === undefined ? "" : safeJsonStringify(content);
  }

  return content
    .map((item) => {
      if (typeof item === "string") {
        return item;
      }
      if (isRecord(item) && typeof item.text === "string") {
        return item.text;
      }
      return safeJsonStringify(item);
    })
    .join("");
}

function appendUserMessage(output: JsonObject[], role: string, content: unknown): void {
  if (typeof content === "string") {
    output.push({ role, content });
    return;
  }

  if (!Array.isArray(content)) {
    output.push({ role, content: "" });
    return;
  }

  const contentParts: JsonObject[] = [];
  const toolResults: JsonObject[] = [];
  for (const rawBlock of content) {
    if (!isRecord(rawBlock)) {
      continue;
    }

    if (rawBlock.type === "tool_result") {
      toolResults.push({
        role: "tool",
        content: toolResultText(rawBlock.content),
        tool_call_id: typeof rawBlock.tool_use_id === "string" ? rawBlock.tool_use_id : "",
      });
      continue;
    }

    if (rawBlock.type === "image") {
      const image = convertImageBlock(rawBlock);
      if (image) {
        contentParts.push(image);
      }
      continue;
    }

    if (rawBlock.type === "text" && typeof rawBlock.text === "string") {
      contentParts.push({ type: "text", text: rawBlock.text });
    }
  }

  output.push(...toolResults);
  if (contentParts.length > 0 || toolResults.length === 0) {
    output.push({ role, content: contentParts });
  }
}

function appendAssistantMessage(output: JsonObject[], content: unknown): void {
  if (typeof content === "string") {
    output.push({ role: "assistant", content });
    return;
  }

  if (!Array.isArray(content)) {
    output.push({ role: "assistant", content: "" });
    return;
  }

  const textParts: string[] = [];
  const thinkingParts: string[] = [];
  const toolCalls: JsonObject[] = [];

  for (const rawBlock of content) {
    if (!isRecord(rawBlock)) {
      continue;
    }
    if (rawBlock.type === "text" && typeof rawBlock.text === "string") {
      textParts.push(rawBlock.text);
    } else if (rawBlock.type === "thinking" && typeof rawBlock.thinking === "string") {
      thinkingParts.push(rawBlock.thinking);
    } else if (rawBlock.type === "tool_use") {
      if (typeof rawBlock.thinking === "string" && rawBlock.thinking.length > 0) {
        thinkingParts.push(rawBlock.thinking);
      }
      toolCalls.push({
        id: typeof rawBlock.id === "string" ? rawBlock.id : "",
        type: "function",
        function: {
          name: typeof rawBlock.name === "string" ? rawBlock.name : "",
          arguments: safeJsonStringify(rawBlock.input ?? {}),
        },
      });
    }
  }

  output.push({
    role: "assistant",
    content: textParts.length > 0 ? textParts.join("") : null,
    ...(thinkingParts.length > 0 ? { reasoning_content: thinkingParts.join("") } : {}),
    ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}),
  });
}

export function transformClaudeRequestToOpenAI(request: JsonObject): JsonObject {
  const messages: JsonObject[] = [];
  const systemText = normalizeSystemText(request.system);
  if (systemText !== null) {
    messages.push({ role: "system", content: systemText });
  }

  if (Array.isArray(request.messages)) {
    for (const rawMessage of request.messages) {
      if (!isRecord(rawMessage)) {
        continue;
      }
      const role = normalizeRole(rawMessage.role);
      if (role === "assistant") {
        appendAssistantMessage(messages, rawMessage.content);
      } else {
        appendUserMessage(messages, "user", rawMessage.content);
      }
    }
  }

  const output: JsonObject = {
    model: typeof request.model === "string" ? request.model : "",
    messages,
    stream: request.stream === true,
  };

  if (Array.isArray(request.tools)) {
    const tools = request.tools.flatMap((rawTool) => {
      if (!isRecord(rawTool) || typeof rawTool.name !== "string") {
        return [];
      }
      if (typeof rawTool.type === "string" && rawTool.type.startsWith("web_search_")) {
        return [];
      }
      return [
        {
          type: "function",
          function: {
            name: rawTool.name,
            ...(typeof rawTool.description === "string"
              ? { description: rawTool.description }
              : {}),
            parameters: isRecord(rawTool.input_schema)
              ? rawTool.input_schema
              : { type: "object", properties: {} },
          },
        },
      ];
    });
    if (tools.length > 0) {
      output.tools = tools;
    }
  }

  if (typeof request.tool_choice === "string") {
    output.tool_choice = request.tool_choice;
  } else if (isRecord(request.tool_choice)) {
    const type = request.tool_choice.type;
    if (type === "auto") {
      output.tool_choice = "auto";
    } else if (type === "any") {
      output.tool_choice = "required";
    } else if (type === "tool" && typeof request.tool_choice.name === "string") {
      output.tool_choice = {
        type: "function",
        function: { name: request.tool_choice.name },
      };
    }
    if (request.tool_choice.disable_parallel_tool_use === true) {
      output.parallel_tool_calls = false;
    }
  }

  if (typeof request.max_tokens === "number") {
    output.max_tokens = request.max_tokens;
  }
  if (typeof request.temperature === "number") {
    output.temperature = request.temperature;
  }
  if (typeof request.top_p === "number") {
    output.top_p = request.top_p;
  }
  if (Array.isArray(request.stop_sequences) && request.stop_sequences.length > 0) {
    output.stop = request.stop_sequences;
  }
  if (request.stream === true) {
    output.stream_options = { include_usage: true };
  }

  return output;
}

export function convertOpenAIUsage(value: unknown): ClaudeUsage {
  const usage = isRecord(value) ? value : {};
  const details = isRecord(usage.prompt_tokens_details) ? usage.prompt_tokens_details : {};
  const promptTokens = firstNumber(usage.prompt_tokens, usage.input_tokens);
  const cacheRead = firstNumber(
    details.cached_tokens,
    usage.prompt_cache_hit_tokens,
    usage.cache_read_input_tokens
  );
  const cacheWrite = firstNumber(
    details.cache_write_tokens,
    usage.prompt_cache_miss_tokens,
    usage.cache_creation_input_tokens
  );

  return {
    input_tokens: Math.max(0, promptTokens - cacheRead - cacheWrite),
    output_tokens: firstNumber(usage.completion_tokens, usage.output_tokens),
    cache_creation_input_tokens: cacheWrite,
    cache_read_input_tokens: cacheRead,
  };
}

function extractTextContent(value: unknown): string {
  if (typeof value === "string") {
    return value;
  }
  if (!Array.isArray(value)) {
    return "";
  }
  return value
    .map((part) => {
      if (!isRecord(part)) {
        return "";
      }
      if ((part.type === "text" || part.type === "output_text") && typeof part.text === "string") {
        return part.text;
      }
      return "";
    })
    .join("");
}

function toClaudeMessageId(value: unknown): string {
  if (typeof value === "string" && value.startsWith("msg_")) {
    return value;
  }
  const suffix = typeof value === "string" ? value.replace(/[^a-zA-Z0-9_-]/g, "_") : "opencode_go";
  return `msg_${suffix || "opencode_go"}`;
}

function mapStopReason(value: unknown, hasToolCalls = false): string {
  if (value === "length") {
    return "max_tokens";
  }
  if (value === "tool_calls" || value === "function_call" || hasToolCalls) {
    return "tool_use";
  }
  return "end_turn";
}

export function transformOpenAIResponseToClaude(value: unknown, fallbackModel = ""): JsonObject {
  if (!isRecord(value)) {
    throw new Error("OpenCode Go returned a non-object response");
  }
  if (isRecord(value.error)) {
    return {
      type: "error",
      error: {
        type: typeof value.error.type === "string" ? value.error.type : "api_error",
        message:
          typeof value.error.message === "string"
            ? value.error.message
            : "OpenCode Go upstream error",
      },
    };
  }

  const choice = Array.isArray(value.choices) && isRecord(value.choices[0]) ? value.choices[0] : {};
  const message = isRecord(choice.message) ? choice.message : {};
  const content: JsonObject[] = [];
  const reasoning = firstString(message.reasoning_content, message.reasoning, message.thinking);
  if (reasoning) {
    content.push({ type: "thinking", thinking: reasoning, signature: THINKING_SIGNATURE });
  }

  const text = extractTextContent(message.content);
  if (text) {
    content.push({ type: "text", text });
  }

  const rawToolCalls = Array.isArray(message.tool_calls) ? message.tool_calls : [];
  for (const [index, rawToolCall] of rawToolCalls.entries()) {
    if (!isRecord(rawToolCall)) {
      continue;
    }
    const fn = isRecord(rawToolCall.function) ? rawToolCall.function : {};
    content.push({
      type: "tool_use",
      id:
        typeof rawToolCall.id === "string" && rawToolCall.id
          ? rawToolCall.id
          : `toolu_opencode_go_${index}`,
      name: typeof fn.name === "string" ? fn.name : "",
      input: parseToolArguments(fn.arguments),
    });
  }

  return {
    id: toClaudeMessageId(value.id),
    type: "message",
    role: "assistant",
    model: typeof value.model === "string" ? value.model : fallbackModel,
    content,
    stop_reason: mapStopReason(choice.finish_reason, rawToolCalls.length > 0),
    stop_sequence: null,
    usage: convertOpenAIUsage(value.usage),
  };
}

function firstString(...values: unknown[]): string {
  for (const value of values) {
    if (typeof value === "string" && value.length > 0) {
      return value;
    }
  }
  return "";
}

function formatSse(event: string, data: JsonObject): string {
  return `event: ${event}\ndata: ${JSON.stringify(data)}\n\n`;
}

class OpenAIToClaudeStreamState {
  private started = false;
  private stopped = false;
  private nextBlockIndex = 0;
  private activeNarrative: { type: "text" | "thinking"; index: number } | null = null;
  private readonly tools = new Map<number, StreamToolState>();
  private messageId = "msg_opencode_go";
  private model: string;
  private finishReason: unknown = null;
  private usage: ClaudeUsage = convertOpenAIUsage(undefined);

  constructor(fallbackModel: string) {
    this.model = fallbackModel;
  }

  consume(data: JsonObject): string[] {
    if (this.stopped) {
      return [];
    }
    if (isRecord(data.error)) {
      this.stopped = true;
      return [
        formatSse("error", {
          type: "error",
          error: {
            type: typeof data.error.type === "string" ? data.error.type : "api_error",
            message:
              typeof data.error.message === "string"
                ? data.error.message
                : "OpenCode Go upstream error",
          },
        }),
      ];
    }

    const output: string[] = [];
    if (typeof data.id === "string") {
      this.messageId = toClaudeMessageId(data.id);
    }
    if (typeof data.model === "string") {
      this.model = data.model;
    }
    if (isRecord(data.usage)) {
      this.usage = convertOpenAIUsage(data.usage);
    }
    this.ensureStarted(output);

    const choice =
      Array.isArray(data.choices) && isRecord(data.choices[0]) ? data.choices[0] : null;
    if (!choice) {
      return output;
    }
    const delta = isRecord(choice.delta) ? choice.delta : {};

    const reasoning = firstString(delta.reasoning_content, delta.reasoning, delta.thinking);
    if (reasoning) {
      this.appendNarrative("thinking", reasoning, output);
    }

    const text = extractTextContent(delta.content);
    if (text) {
      this.appendNarrative("text", text, output);
    }

    if (Array.isArray(delta.tool_calls)) {
      this.closeNarrative(output);
      for (const [position, rawToolCall] of delta.tool_calls.entries()) {
        if (!isRecord(rawToolCall)) {
          continue;
        }
        const upstreamIndex = typeof rawToolCall.index === "number" ? rawToolCall.index : position;
        const fn = isRecord(rawToolCall.function) ? rawToolCall.function : {};
        let tool = this.tools.get(upstreamIndex);
        if (!tool) {
          tool = {
            blockIndex: this.nextBlockIndex++,
            id:
              typeof rawToolCall.id === "string" && rawToolCall.id
                ? rawToolCall.id
                : `toolu_opencode_go_${upstreamIndex}`,
            name: "",
            started: false,
            pendingArguments: "",
          };
          this.tools.set(upstreamIndex, tool);
        }

        if (typeof rawToolCall.id === "string" && rawToolCall.id) {
          tool.id = rawToolCall.id;
        }
        if (typeof fn.name === "string" && fn.name) {
          tool.name = mergeStreamFragment(tool.name, fn.name);
        }
        const argumentsDelta = typeof fn.arguments === "string" ? fn.arguments : "";
        if (!tool.started) {
          tool.pendingArguments += argumentsDelta;
          if (tool.name) {
            output.push(
              formatSse("content_block_start", {
                type: "content_block_start",
                index: tool.blockIndex,
                content_block: { type: "tool_use", id: tool.id, name: tool.name, input: {} },
              })
            );
            tool.started = true;
            if (tool.pendingArguments) {
              output.push(this.toolArgumentsDelta(tool, tool.pendingArguments));
              tool.pendingArguments = "";
            }
          }
        } else if (argumentsDelta) {
          output.push(this.toolArgumentsDelta(tool, argumentsDelta));
        }
      }
    }

    if (choice.finish_reason !== null && choice.finish_reason !== undefined) {
      this.finishReason = choice.finish_reason;
    }
    return output;
  }

  finish(): string[] {
    if (this.stopped) {
      return [];
    }
    const output: string[] = [];
    this.ensureStarted(output);
    this.closeNarrative(output);
    for (const tool of [...this.tools.values()].sort((a, b) => a.blockIndex - b.blockIndex)) {
      if (!tool.started) {
        output.push(
          formatSse("content_block_start", {
            type: "content_block_start",
            index: tool.blockIndex,
            content_block: { type: "tool_use", id: tool.id, name: tool.name, input: {} },
          })
        );
        tool.started = true;
        if (tool.pendingArguments) {
          output.push(this.toolArgumentsDelta(tool, tool.pendingArguments));
        }
      }
      output.push(
        formatSse("content_block_stop", {
          type: "content_block_stop",
          index: tool.blockIndex,
        })
      );
    }

    output.push(
      formatSse("message_delta", {
        type: "message_delta",
        delta: {
          stop_reason: mapStopReason(this.finishReason, this.tools.size > 0),
          stop_sequence: null,
        },
        usage: this.usage,
      }),
      formatSse("message_stop", { type: "message_stop" })
    );
    this.stopped = true;
    return output;
  }

  private ensureStarted(output: string[]): void {
    if (this.started) {
      return;
    }
    output.push(
      formatSse("message_start", {
        type: "message_start",
        message: {
          id: this.messageId,
          type: "message",
          role: "assistant",
          model: this.model,
          content: [],
          stop_reason: null,
          stop_sequence: null,
          usage: convertOpenAIUsage(undefined),
        },
      })
    );
    this.started = true;
  }

  private appendNarrative(type: "text" | "thinking", value: string, output: string[]): void {
    if (this.activeNarrative?.type !== type) {
      this.closeNarrative(output);
      const index = this.nextBlockIndex++;
      this.activeNarrative = { type, index };
      output.push(
        formatSse("content_block_start", {
          type: "content_block_start",
          index,
          content_block:
            type === "thinking"
              ? { type: "thinking", thinking: "", signature: "" }
              : { type: "text", text: "" },
        })
      );
    }

    output.push(
      formatSse("content_block_delta", {
        type: "content_block_delta",
        index: this.activeNarrative.index,
        delta:
          type === "thinking"
            ? { type: "thinking_delta", thinking: value }
            : { type: "text_delta", text: value },
      })
    );
  }

  private closeNarrative(output: string[]): void {
    if (!this.activeNarrative) {
      return;
    }
    if (this.activeNarrative.type === "thinking") {
      output.push(
        formatSse("content_block_delta", {
          type: "content_block_delta",
          index: this.activeNarrative.index,
          delta: { type: "signature_delta", signature: THINKING_SIGNATURE },
        })
      );
    }
    output.push(
      formatSse("content_block_stop", {
        type: "content_block_stop",
        index: this.activeNarrative.index,
      })
    );
    this.activeNarrative = null;
  }

  private toolArgumentsDelta(tool: StreamToolState, value: string): string {
    return formatSse("content_block_delta", {
      type: "content_block_delta",
      index: tool.blockIndex,
      delta: { type: "input_json_delta", partial_json: value },
    });
  }
}

function mergeStreamFragment(current: string, incoming: string): string {
  if (!current || incoming.startsWith(current)) {
    return incoming;
  }
  if (current.endsWith(incoming)) {
    return current;
  }
  return current + incoming;
}

function findFrameBoundary(buffer: string): { index: number; length: number } | null {
  const match = /\r?\n\r?\n/.exec(buffer);
  return match ? { index: match.index, length: match[0].length } : null;
}

function parseSseData(frame: string): string | null {
  const values = frame
    .split(/\r?\n/)
    .filter((line) => line.startsWith("data:"))
    .map((line) => line.slice(5).trimStart());
  return values.length > 0 ? values.join("\n") : null;
}

export function createOpenAIToClaudeStreamTransform(
  fallbackModel = "",
  onUpstreamChunk?: (chunk: Uint8Array) => void
): TransformStream<Uint8Array, Uint8Array> {
  const decoder = new TextDecoder();
  const encoder = new TextEncoder();
  const state = new OpenAIToClaudeStreamState(fallbackModel);
  let buffer = "";

  const emit = (controller: TransformStreamDefaultController<Uint8Array>, chunks: string[]) => {
    for (const chunk of chunks) {
      controller.enqueue(encoder.encode(chunk));
    }
  };

  const consumeFrame = (
    frame: string,
    controller: TransformStreamDefaultController<Uint8Array>
  ) => {
    const data = parseSseData(frame);
    if (!data) {
      return;
    }
    if (data === "[DONE]") {
      emit(controller, state.finish());
      return;
    }
    try {
      const parsed = JSON.parse(data) as unknown;
      if (isRecord(parsed)) {
        emit(controller, state.consume(parsed));
      }
    } catch {
      // Wait for a complete SSE frame; invalid upstream frames are ignored.
    }
  };

  return new TransformStream<Uint8Array, Uint8Array>({
    transform(chunk, controller) {
      onUpstreamChunk?.(chunk);
      buffer += decoder.decode(chunk, { stream: true });
      let boundary = findFrameBoundary(buffer);
      while (boundary) {
        const frame = buffer.slice(0, boundary.index);
        buffer = buffer.slice(boundary.index + boundary.length);
        consumeFrame(frame, controller);
        boundary = findFrameBoundary(buffer);
      }
    },
    flush(controller) {
      buffer += decoder.decode();
      if (buffer.trim()) {
        consumeFrame(buffer, controller);
      }
      emit(controller, state.finish());
    },
  });
}
