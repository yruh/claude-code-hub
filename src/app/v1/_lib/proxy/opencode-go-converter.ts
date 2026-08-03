type JsonObject = Record<string, unknown>;

interface AnthropicUsage {
  input_tokens: number;
  output_tokens: number;
  cache_creation_input_tokens: number;
  cache_read_input_tokens: number;
}

interface OpenAIUsage {
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens: number;
  prompt_tokens_details: {
    cached_tokens: number;
    cache_write_tokens: number;
  };
}

const BILLING_HEADER_ONLY = /^\s*x-anthropic-billing-header\s*:[^\r\n]*\s*$/i;
const DEFAULT_MAX_TOKENS = 4096;

function isRecord(value: unknown): value is JsonObject {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function firstNumber(...values: unknown[]): number {
  for (const value of values) {
    if (typeof value === "number" && Number.isFinite(value)) {
      return Math.max(0, value);
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

function stableSystemText(value: unknown): string | null {
  if (typeof value === "string") {
    if (BILLING_HEADER_ONLY.test(value)) {
      return null;
    }
    const stable = stripVolatileCch(value);
    return stable.length > 0 ? stable : null;
  }

  if (!Array.isArray(value)) {
    return null;
  }

  const stableParts = value.flatMap((part) => {
    const text =
      typeof part === "string"
        ? part
        : isRecord(part) && typeof part.text === "string"
          ? part.text
          : "";
    if (!text || BILLING_HEADER_ONLY.test(text)) {
      return [];
    }
    const stable = stripVolatileCch(text);
    return stable.length > 0 ? [stable] : [];
  });

  return stableParts.length > 0 ? stableParts.join("\n\n") : null;
}

function parseDataUrl(url: string): { mediaType: string; data: string } | null {
  const match = /^data:([^;,]+);base64,([\s\S]+)$/i.exec(url);
  return match ? { mediaType: match[1], data: match[2] } : null;
}

function convertImageUrl(value: unknown): JsonObject | null {
  const url =
    typeof value === "string"
      ? value
      : isRecord(value) && typeof value.url === "string"
        ? value.url
        : "";
  if (!url) {
    return null;
  }

  const dataUrl = parseDataUrl(url);
  if (dataUrl) {
    const blockType = dataUrl.mediaType === "application/pdf" ? "document" : "image";
    return {
      type: blockType,
      source: {
        type: "base64",
        media_type: dataUrl.mediaType,
        data: dataUrl.data,
      },
    };
  }

  return {
    type: "image",
    source: { type: "url", url },
  };
}

function convertOpenAIContent(value: unknown): JsonObject[] {
  if (typeof value === "string") {
    return value.length > 0 ? [{ type: "text", text: value }] : [];
  }
  if (!Array.isArray(value)) {
    return [];
  }

  const blocks: JsonObject[] = [];
  for (const part of value) {
    if (typeof part === "string") {
      if (part.length > 0) {
        blocks.push({ type: "text", text: part });
      }
      continue;
    }
    if (!isRecord(part)) {
      continue;
    }

    if (
      (part.type === "text" || part.type === "input_text" || part.type === "output_text") &&
      typeof part.text === "string"
    ) {
      if (part.text.length > 0) {
        blocks.push({ type: "text", text: part.text });
      }
      continue;
    }

    if (part.type === "image_url" || part.type === "input_image") {
      const image = convertImageUrl(part.image_url);
      if (image) {
        blocks.push(image);
      }
    }
  }
  return blocks;
}

function appendClaudeMessage(
  messages: JsonObject[],
  role: "user" | "assistant",
  blocks: JsonObject[]
) {
  const content = blocks.length > 0 ? blocks : [{ type: "text", text: "..." }];
  messages.push({ role, content });
}

function convertAssistantMessage(message: JsonObject): JsonObject[] {
  const blocks = convertOpenAIContent(message.content);
  const rawToolCalls = Array.isArray(message.tool_calls) ? message.tool_calls : [];

  for (const [index, rawToolCall] of rawToolCalls.entries()) {
    if (!isRecord(rawToolCall)) {
      continue;
    }
    const fn = isRecord(rawToolCall.function) ? rawToolCall.function : {};
    blocks.push({
      type: "tool_use",
      id:
        typeof rawToolCall.id === "string" && rawToolCall.id
          ? rawToolCall.id
          : `toolu_opencode_go_${index}`,
      name: typeof fn.name === "string" ? fn.name : "",
      input: parseToolArguments(fn.arguments),
    });
  }

  if (isRecord(message.function_call)) {
    blocks.push({
      type: "tool_use",
      id: "toolu_opencode_go_function",
      name: typeof message.function_call.name === "string" ? message.function_call.name : "",
      input: parseToolArguments(message.function_call.arguments),
    });
  }

  return blocks;
}

function convertToolResult(message: JsonObject): JsonObject {
  const content = convertOpenAIContent(message.content);
  return {
    type: "tool_result",
    tool_use_id:
      typeof message.tool_call_id === "string" && message.tool_call_id
        ? message.tool_call_id
        : typeof message.name === "string"
          ? message.name
          : "toolu_opencode_go_unknown",
    content: content.length > 0 ? content : [{ type: "text", text: "" }],
  };
}

function convertTools(value: unknown): JsonObject[] {
  if (!Array.isArray(value)) {
    return [];
  }
  return value.flatMap((rawTool) => {
    if (!isRecord(rawTool)) {
      return [];
    }
    const fn = isRecord(rawTool.function) ? rawTool.function : rawTool;
    if (typeof fn.name !== "string" || !fn.name) {
      return [];
    }
    return [
      {
        name: fn.name,
        ...(typeof fn.description === "string" ? { description: fn.description } : {}),
        input_schema: isRecord(fn.parameters) ? fn.parameters : { type: "object", properties: {} },
      },
    ];
  });
}

function convertToolChoice(value: unknown, disableParallel: boolean): JsonObject | null {
  if (value === "auto") {
    return { type: "auto", ...(disableParallel ? { disable_parallel_tool_use: true } : {}) };
  }
  if (value === "required") {
    return { type: "any", ...(disableParallel ? { disable_parallel_tool_use: true } : {}) };
  }
  if (!isRecord(value)) {
    return disableParallel ? { type: "auto", disable_parallel_tool_use: true } : null;
  }

  const fn = isRecord(value.function) ? value.function : value;
  if (typeof fn.name !== "string" || !fn.name) {
    return null;
  }
  return {
    type: "tool",
    name: fn.name,
    ...(disableParallel ? { disable_parallel_tool_use: true } : {}),
  };
}

function resolveThinking(request: JsonObject, maxTokens: number): JsonObject | null {
  const reasoning = isRecord(request.reasoning) ? request.reasoning : {};
  const effort =
    typeof request.reasoning_effort === "string"
      ? request.reasoning_effort
      : typeof reasoning.effort === "string"
        ? reasoning.effort
        : "";
  const budgets: Record<string, number> = {
    low: 1280,
    medium: 2048,
    high: 4096,
    xhigh: 8192,
  };
  const requestedBudget = firstNumber(
    reasoning.max_tokens,
    reasoning.budget_tokens,
    budgets[effort]
  );
  if (requestedBudget < 1024 || maxTokens <= 1024) {
    return null;
  }
  return { type: "enabled", budget_tokens: Math.min(requestedBudget, maxTokens - 1) };
}

export function transformOpenAIRequestToClaude(request: JsonObject): JsonObject {
  const messages: JsonObject[] = [];
  const system: JsonObject[] = [];
  let conversationStarted = false;

  if (Array.isArray(request.messages)) {
    for (const rawMessage of request.messages) {
      if (!isRecord(rawMessage)) {
        continue;
      }
      const role = typeof rawMessage.role === "string" ? rawMessage.role.toLowerCase() : "user";

      if (role === "system" || role === "developer") {
        const stable = stableSystemText(rawMessage.content);
        if (!stable) {
          continue;
        }
        if (!conversationStarted) {
          system.push({ type: "text", text: stable });
        } else {
          appendClaudeMessage(messages, "user", [{ type: "text", text: stable }]);
        }
        continue;
      }

      conversationStarted = true;
      if (role === "assistant") {
        appendClaudeMessage(messages, "assistant", convertAssistantMessage(rawMessage));
      } else if (role === "tool" || role === "function") {
        appendClaudeMessage(messages, "user", [convertToolResult(rawMessage)]);
      } else {
        appendClaudeMessage(messages, "user", convertOpenAIContent(rawMessage.content));
      }
    }
  }

  if (messages[0]?.role === "assistant") {
    messages.unshift({ role: "user", content: [{ type: "text", text: "..." }] });
  }
  if (messages.length === 0) {
    messages.push({ role: "user", content: [{ type: "text", text: "..." }] });
  }

  const maxTokens = Math.max(
    1,
    firstNumber(request.max_completion_tokens, request.max_tokens) || DEFAULT_MAX_TOKENS
  );
  const output: JsonObject = {
    model: typeof request.model === "string" ? request.model : "",
    messages,
    max_tokens: maxTokens,
    stream: request.stream === true,
  };

  if (system.length > 0) {
    output.system = system;
  }

  const tools = convertTools(request.tools);
  if (tools.length > 0 && request.tool_choice !== "none") {
    output.tools = tools;
    const toolChoice = convertToolChoice(
      request.tool_choice,
      request.parallel_tool_calls === false
    );
    if (toolChoice) {
      output.tool_choice = toolChoice;
    }
  }

  if (typeof request.temperature === "number") {
    output.temperature = request.temperature;
  }
  if (typeof request.top_p === "number") {
    output.top_p = request.top_p;
  }
  if (typeof request.top_k === "number") {
    output.top_k = request.top_k;
  }

  const stop = typeof request.stop === "string" ? [request.stop] : request.stop;
  if (Array.isArray(stop)) {
    const stopSequences = stop.filter((item): item is string => typeof item === "string" && !!item);
    if (stopSequences.length > 0) {
      output.stop_sequences = stopSequences;
    }
  }

  const userId =
    typeof request.user === "string"
      ? request.user
      : isRecord(request.metadata) && typeof request.metadata.user_id === "string"
        ? request.metadata.user_id
        : null;
  if (userId) {
    output.metadata = { user_id: userId };
  }

  const thinking = resolveThinking(request, maxTokens);
  if (thinking) {
    output.thinking = thinking;
    output.temperature = 1;
  }

  return output;
}

export function convertAnthropicUsage(value: unknown): OpenAIUsage {
  const usage = isRecord(value) ? value : {};
  const input = firstNumber(usage.input_tokens);
  const output = firstNumber(usage.output_tokens);
  const cacheRead = firstNumber(usage.cache_read_input_tokens);
  const cacheWrite = firstNumber(usage.cache_creation_input_tokens);
  const promptTokens = input + cacheRead + cacheWrite;

  return {
    prompt_tokens: promptTokens,
    completion_tokens: output,
    total_tokens: promptTokens + output,
    prompt_tokens_details: {
      cached_tokens: cacheRead,
      cache_write_tokens: cacheWrite,
    },
  };
}

function mapAnthropicStopReason(value: unknown): string | null {
  if (value === "max_tokens") {
    return "length";
  }
  if (value === "tool_use") {
    return "tool_calls";
  }
  if (value === "refusal") {
    return "content_filter";
  }
  if (value === "end_turn" || value === "stop_sequence" || value === "pause_turn") {
    return "stop";
  }
  return null;
}

function toOpenAIError(value: JsonObject): JsonObject {
  const error = isRecord(value.error) ? value.error : value;
  return {
    error: {
      message:
        typeof error.message === "string" ? error.message : "Anthropic Messages upstream error",
      type: typeof error.type === "string" ? error.type : "api_error",
      code: typeof error.code === "string" ? error.code : null,
      param: null,
    },
  };
}

export function transformClaudeResponseToOpenAI(value: unknown, fallbackModel = ""): JsonObject {
  if (!isRecord(value)) {
    throw new Error("OpenCode Go returned a non-object Anthropic response");
  }
  if (value.type === "error" || isRecord(value.error)) {
    return toOpenAIError(value);
  }

  const text: string[] = [];
  const reasoning: string[] = [];
  const toolCalls: JsonObject[] = [];
  if (Array.isArray(value.content)) {
    for (const rawBlock of value.content) {
      if (!isRecord(rawBlock)) {
        continue;
      }
      if (rawBlock.type === "text" && typeof rawBlock.text === "string") {
        text.push(rawBlock.text);
      } else if (rawBlock.type === "thinking" && typeof rawBlock.thinking === "string") {
        reasoning.push(rawBlock.thinking);
      } else if (rawBlock.type === "tool_use") {
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
  }

  const message: JsonObject = {
    role: "assistant",
    content: text.length > 0 ? text.join("") : null,
  };
  if (reasoning.length > 0) {
    message.reasoning_content = reasoning.join("");
  }
  if (toolCalls.length > 0) {
    message.tool_calls = toolCalls;
  }

  return {
    id: typeof value.id === "string" ? value.id : "chatcmpl-opencode-go",
    object: "chat.completion",
    created: Math.floor(Date.now() / 1000),
    model: typeof value.model === "string" ? value.model : fallbackModel,
    choices: [
      {
        index: 0,
        message,
        finish_reason: mapAnthropicStopReason(value.stop_reason),
        logprobs: null,
      },
    ],
    usage: convertAnthropicUsage(value.usage),
  };
}

function mergeAnthropicUsage(current: AnthropicUsage, value: unknown): AnthropicUsage {
  const usage = isRecord(value) ? value : {};
  return {
    input_tokens:
      typeof usage.input_tokens === "number"
        ? firstNumber(usage.input_tokens)
        : current.input_tokens,
    output_tokens:
      typeof usage.output_tokens === "number"
        ? firstNumber(usage.output_tokens)
        : current.output_tokens,
    cache_creation_input_tokens:
      typeof usage.cache_creation_input_tokens === "number"
        ? firstNumber(usage.cache_creation_input_tokens)
        : current.cache_creation_input_tokens,
    cache_read_input_tokens:
      typeof usage.cache_read_input_tokens === "number"
        ? firstNumber(usage.cache_read_input_tokens)
        : current.cache_read_input_tokens,
  };
}

function emptyAnthropicUsage(): AnthropicUsage {
  return {
    input_tokens: 0,
    output_tokens: 0,
    cache_creation_input_tokens: 0,
    cache_read_input_tokens: 0,
  };
}

class ClaudeToOpenAIStreamState {
  private readonly toolIndexes = new Map<number, number>();
  private messageId = "chatcmpl-opencode-go";
  private model: string;
  private created = Math.floor(Date.now() / 1000);
  private usage = emptyAnthropicUsage();
  private nextToolIndex = 0;
  private started = false;
  private finishEmitted = false;
  private stopped = false;
  private stopReason: unknown = null;

  constructor(fallbackModel: string) {
    this.model = fallbackModel;
  }

  consume(data: JsonObject): string[] {
    if (this.stopped) {
      return [];
    }
    if (data.type === "error" || isRecord(data.error)) {
      this.stopped = true;
      return [formatOpenAISse(toOpenAIError(data))];
    }

    const output: string[] = [];
    if (data.type === "message_start" && isRecord(data.message)) {
      const message = data.message;
      if (typeof message.id === "string") {
        this.messageId = message.id;
      }
      if (typeof message.model === "string") {
        this.model = message.model;
      }
      this.usage = mergeAnthropicUsage(this.usage, message.usage);
      this.ensureStarted(output);
      return output;
    }

    this.ensureStarted(output);
    if (data.type === "content_block_start" && isRecord(data.content_block)) {
      this.consumeBlockStart(data, data.content_block, output);
    } else if (data.type === "content_block_delta" && isRecord(data.delta)) {
      this.consumeBlockDelta(data, data.delta, output);
    } else if (data.type === "message_delta") {
      if (isRecord(data.delta) && data.delta.stop_reason !== undefined) {
        this.stopReason = data.delta.stop_reason;
      }
      this.usage = mergeAnthropicUsage(this.usage, data.usage);
      this.emitFinish(output);
    } else if (data.type === "message_stop") {
      this.emitFinish(output);
      output.push("data: [DONE]\n\n");
      this.stopped = true;
    }
    return output;
  }

  finish(): string[] {
    if (this.stopped) {
      return [];
    }
    const output: string[] = [];
    this.ensureStarted(output);
    this.emitFinish(output);
    output.push("data: [DONE]\n\n");
    this.stopped = true;
    return output;
  }

  private consumeBlockStart(data: JsonObject, block: JsonObject, output: string[]) {
    if (block.type === "text" && typeof block.text === "string" && block.text) {
      output.push(this.chunk({ content: block.text }));
      return;
    }
    if (block.type === "thinking" && typeof block.thinking === "string" && block.thinking) {
      output.push(this.chunk({ reasoning_content: block.thinking }));
      return;
    }
    if (block.type !== "tool_use") {
      return;
    }

    const blockIndex = typeof data.index === "number" ? data.index : this.nextToolIndex;
    const toolIndex = this.nextToolIndex++;
    this.toolIndexes.set(blockIndex, toolIndex);
    const argumentsText =
      isRecord(block.input) && Object.keys(block.input).length > 0
        ? safeJsonStringify(block.input)
        : "";
    output.push(
      this.chunk({
        tool_calls: [
          {
            index: toolIndex,
            id: typeof block.id === "string" ? block.id : "",
            type: "function",
            function: {
              name: typeof block.name === "string" ? block.name : "",
              arguments: argumentsText,
            },
          },
        ],
      })
    );
  }

  private consumeBlockDelta(data: JsonObject, delta: JsonObject, output: string[]) {
    if (delta.type === "text_delta" && typeof delta.text === "string") {
      output.push(this.chunk({ content: delta.text }));
      return;
    }
    if (delta.type === "thinking_delta" && typeof delta.thinking === "string") {
      output.push(this.chunk({ reasoning_content: delta.thinking }));
      return;
    }
    if (delta.type !== "input_json_delta" || typeof delta.partial_json !== "string") {
      return;
    }

    const blockIndex = typeof data.index === "number" ? data.index : 0;
    const toolIndex = this.toolIndexes.get(blockIndex) ?? blockIndex;
    output.push(
      this.chunk({
        tool_calls: [
          {
            index: toolIndex,
            function: { arguments: delta.partial_json },
          },
        ],
      })
    );
  }

  private ensureStarted(output: string[]) {
    if (this.started) {
      return;
    }
    output.push(this.chunk({ role: "assistant", content: "" }));
    this.started = true;
  }

  private emitFinish(output: string[]) {
    if (this.finishEmitted) {
      return;
    }
    output.push(
      this.chunk(
        {},
        mapAnthropicStopReason(this.stopReason) ?? "stop",
        convertAnthropicUsage(this.usage)
      )
    );
    this.finishEmitted = true;
  }

  private chunk(
    delta: JsonObject,
    finishReason: string | null = null,
    usage?: OpenAIUsage
  ): string {
    return formatOpenAISse({
      id: this.messageId,
      object: "chat.completion.chunk",
      created: this.created,
      model: this.model,
      choices: [{ index: 0, delta, finish_reason: finishReason, logprobs: null }],
      ...(usage ? { usage } : {}),
    });
  }
}

function formatOpenAISse(data: JsonObject): string {
  return `data: ${JSON.stringify(data)}\n\n`;
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

export function createClaudeToOpenAIStreamTransform(
  fallbackModel = "",
  onUpstreamChunk?: (chunk: Uint8Array) => void
): TransformStream<Uint8Array, Uint8Array> {
  const decoder = new TextDecoder();
  const encoder = new TextEncoder();
  const state = new ClaudeToOpenAIStreamState(fallbackModel);
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
      // Invalid upstream frames are ignored after the content gate has committed.
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
