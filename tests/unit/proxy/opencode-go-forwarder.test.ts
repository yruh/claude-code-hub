import { beforeEach, describe, expect, it, vi } from "vitest";
import { resolveEndpointPolicy } from "@/app/v1/_lib/proxy/endpoint-policy";
import { ProxyForwarder } from "@/app/v1/_lib/proxy/forwarder";
import { ProxySession } from "@/app/v1/_lib/proxy/session";
import type { Provider } from "@/types/provider";

vi.mock("@/lib/logger", () => ({
  logger: {
    debug: vi.fn(),
    error: vi.fn(),
    info: vi.fn(),
    trace: vi.fn(),
    warn: vi.fn(),
    fatal: vi.fn(),
  },
}));

vi.mock("@/lib/request-filter-engine", () => ({
  requestFilterEngine: {
    applyFinal: vi.fn(async () => {}),
  },
}));

function createProvider(): Provider {
  return {
    id: 74,
    name: "opencode-go-upstream",
    providerType: "opencode-go",
    url: "https://opencode.example.com/v1",
    key: "upstream-key",
    preserveClientIp: false,
    priority: 0,
    costMultiplier: 1,
    maxRetryAttempts: 1,
    mcpPassthroughType: "none",
    requestTimeoutNonStreamingMs: 0,
    firstByteTimeoutStreamingMs: 0,
  } as unknown as Provider;
}

function createMessage(cch: string): Record<string, unknown> {
  return {
    model: "claude-sonnet-4-5",
    messages: [
      {
        role: "system",
        content: `x-anthropic-billing-header: cc_version=2.1.177; cc_entrypoint=cli; cch=${cch};\nYou are a coding assistant.`,
      },
      { role: "user", content: "first" },
      { role: "SYSTEM", content: "The date changed." },
    ],
    max_tokens: 4096,
    stream: false,
  };
}

function createSession(cch: string): ProxySession {
  const headers = new Headers({
    "content-type": "application/json",
    authorization: "Bearer proxy-user-key",
  });
  const message = createMessage(cch);
  const session = Object.create(ProxySession.prototype);

  Object.assign(session, {
    startTime: Date.now(),
    method: "POST",
    requestUrl: new URL("https://proxy.example.com/v1/chat/completions"),
    headers,
    originalHeaders: new Headers(headers),
    headerLog: JSON.stringify(Object.fromEntries(headers.entries())),
    request: {
      model: "claude-sonnet-4-5",
      log: JSON.stringify(message),
      message,
    },
    userAgent: "OpenAI-Compatible/2026.04",
    context: null,
    clientAbortSignal: null,
    userName: "test-user",
    authState: { success: true, user: null, key: null, apiKey: null },
    provider: null,
    messageContext: null,
    sessionId: null,
    requestSequence: 1,
    originalFormat: "openai",
    providerType: null,
    originalModelName: null,
    originalUrlPathname: null,
    providerChain: [],
    cacheTtlResolved: null,
    context1mApplied: false,
    cachedPriceData: undefined,
    cachedBillingModelSource: undefined,
    forwardedRequestBody: null,
    endpointPolicy: resolveEndpointPolicy("/v1/chat/completions"),
    setCacheTtlResolved: vi.fn(),
    getCacheTtlResolved: vi.fn(() => null),
    getCurrentModel: vi.fn(() => "claude-sonnet-4-5"),
    clientRequestsContext1m: vi.fn(() => false),
    setContext1mApplied: vi.fn(),
    getContext1mApplied: vi.fn(() => false),
    getGroupCostMultiplier: vi.fn(() => 1),
    getEndpointPolicy: vi.fn(() => resolveEndpointPolicy("/v1/chat/completions")),
    isHeaderModified: vi.fn(() => false),
  });

  return session as ProxySession;
}

describe("ProxyForwarder - OpenCode Go conversion", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("sends a stable Anthropic Messages request with Anthropic headers", async () => {
    const provider = createProvider();
    const captured: Array<{ url: string; init: RequestInit }> = [];
    const fetchWithoutAutoDecode = vi.spyOn(ProxyForwarder as never, "fetchWithoutAutoDecode");
    fetchWithoutAutoDecode.mockImplementation(async (url: string, init: RequestInit) => {
      captured.push({ url, init });
      return new Response(
        JSON.stringify({
          id: "msg_test",
          type: "message",
          role: "assistant",
          model: "claude-sonnet-4-5",
          content: [{ type: "text", text: "ok" }],
          stop_reason: "end_turn",
          usage: { input_tokens: 5, output_tokens: 1 },
        }),
        { status: 200, headers: { "content-type": "application/json" } }
      );
    });

    const { doForward } = ProxyForwarder as unknown as {
      doForward: (session: ProxySession, provider: Provider, baseUrl: string) => Promise<Response>;
    };
    const firstSession = createSession("a1b2");
    const secondSession = createSession("c3d4");

    await doForward(firstSession, provider, provider.url);
    await doForward(secondSession, provider, provider.url);

    expect(captured).toHaveLength(2);
    expect(captured[0].url).toBe("https://opencode.example.com/v1/messages");

    const upstreamHeaders = new Headers(captured[0].init.headers);
    expect(upstreamHeaders.get("authorization")).toBe("Bearer upstream-key");
    expect(upstreamHeaders.get("x-api-key")).toBe("upstream-key");
    expect(upstreamHeaders.get("anthropic-version")).toBe("2023-06-01");

    const firstBody = JSON.parse(String(captured[0].init.body));
    const secondBody = JSON.parse(String(captured[1].init.body));
    expect(firstBody).toEqual(secondBody);
    expect(firstBody.system).toEqual([
      {
        type: "text",
        text: "x-anthropic-billing-header: cc_version=2.1.177; cc_entrypoint=cli\nYou are a coding assistant.",
      },
    ]);
    expect(firstBody.messages).toEqual([
      { role: "user", content: [{ type: "text", text: "first" }] },
      { role: "user", content: [{ type: "text", text: "The date changed." }] },
    ]);
    expect(JSON.stringify(firstBody)).not.toContain("cch=");
    expect(firstSession.request.message).toEqual(createMessage("a1b2"));
  });
});
