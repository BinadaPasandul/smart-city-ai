import { useCallback, useState } from "react";

import { ApiError, apiClient } from "@/services/apiClient";
import type { ChatError, ChatMessage, ChatResponse } from "@/types/chat";

/**
 * Chat state backed by the real POST /api/v1/chat endpoint via apiClient.
 *
 * `sendMessage` appends the user's message immediately, sets
 * `isAssistantTyping` for the duration of the actual network request (no
 * artificial delay), then appends exactly one assistant message mapped from
 * either a successful response, a logical backend failure (HTTP 200,
 * `success: false`), or a client-side/network error. See `mapResponseToMessage`
 * and `describeRequestFailure` below for exactly how each case is handled.
 *
 * All state updates here are plain top-level `setX(...)` calls (or pure
 * functional updaters that only read `prev`) — never nested inside another
 * setter's updater function. That nesting pattern is what caused the earlier
 * StrictMode double-message bug; keeping every setter call pure and
 * top-level is what prevents it from coming back.
 */

export interface UseChatResult {
  messages: ChatMessage[];
  draft: string;
  /** Updates the draft; also marks the conversation as started once non-empty. */
  setDraft: (value: string) => void;
  /** Sends the current trimmed draft, if non-empty and not already in flight. */
  sendMessage: () => void;
  /** True for the duration of an in-flight request (drives the typing indicator and composer lockout). */
  isAssistantTyping: boolean;
  /** Whether the dedicated chat layout should replace the landing page. */
  hasStarted: boolean;
  /** Opens the dedicated chat layout with an empty draft (e.g. the floating chat button). */
  startChat: () => void;
}

export function useChat(): UseChatResult {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [draft, setDraftState] = useState("");
  const [hasStarted, setHasStarted] = useState(false);
  const [isAssistantTyping, setIsAssistantTyping] = useState(false);

  const setDraft = useCallback((value: string) => {
    setDraftState(value);
    if (value.trim().length > 0) {
      setHasStarted(true);
    }
  }, []);

  const startChat = useCallback(() => {
    setHasStarted(true);
  }, []);

  const sendMessage = useCallback(() => {
    const content = draft.trim();
    // Guards against both an empty draft and a second submission while one
    // request is already in flight (the composer is also disabled for this,
    // but sendMessage stays safe to call even if something else triggers it).
    if (!content || isAssistantTyping) {
      return;
    }

    setHasStarted(true);
    setDraftState("");
    const userMessage: ChatMessage = {
      id: crypto.randomUUID(),
      role: "user",
      content,
      timestamp: new Date(),
    };
    setMessages((prev) => [...prev, userMessage]);
    setIsAssistantTyping(true);

    void requestAssistantReply(content).then((assistantMessage) => {
      setMessages((prev) => [...prev, assistantMessage]);
      setIsAssistantTyping(false);
    });
  }, [draft, isAssistantTyping]);

  return { messages, draft, setDraft, sendMessage, isAssistantTyping, hasStarted, startChat };
}

/** Call the backend and always resolve to one assistant message — never throws. */
async function requestAssistantReply(message: string): Promise<ChatMessage> {
  try {
    // Context is intentionally empty: see ChatRequest's docstring in types/chat.ts.
    const response = await apiClient.post<ChatResponse>("/chat", { message, context: {} });
    return mapResponseToMessage(response);
  } catch (error) {
    return mapFailureToMessage(describeRequestFailure(error));
  }
}

/** Map a real backend response (success or logical failure) onto one message. */
function mapResponseToMessage(response: ChatResponse): ChatMessage {
  return {
    id: crypto.randomUUID(),
    role: "assistant",
    content: response.success
      ? response.answer
      : response.error?.message ?? "The assistant couldn't complete this request.",
    timestamp: new Date(),
    sources: response.sources,
    metadata: {
      selected_agents: response.metadata.selected_agents,
      execution_status: response.metadata.execution_status,
      successful_agents: response.metadata.successful_agents,
      failed_agents: response.metadata.failed_agents,
      synthesis_method: response.metadata.synthesis_method,
      answer_basis: response.metadata.answer_basis,
    },
    error: response.error ?? undefined,
  };
}

function mapFailureToMessage(error: ChatError): ChatMessage {
  return {
    id: crypto.randomUUID(),
    role: "assistant",
    content: error.message,
    timestamp: new Date(),
    error,
  };
}

/**
 * Translate an HTTP error or a network failure into one safe, user-facing
 * message. Never surfaces a stack trace or raw server exception text.
 */
function describeRequestFailure(error: unknown): ChatError {
  if (error instanceof ApiError) {
    const detail = isRecord(error.body) ? error.body.detail : undefined;

    switch (error.status) {
      case 401:
        return {
          code: "unauthorized",
          message: safeMessage(detail, "You need to sign in to continue."),
        };
      case 422:
        // FastAPI's 422 body is a list of field errors, not a user-safe
        // string — never render that directly.
        return {
          code: "invalid_request",
          message: "Your message couldn't be sent. Please rephrase it and try again.",
        };
      case 429:
        return {
          code: "rate_limit_exceeded",
          message: safeMessage(
            detail,
            "You're sending requests too quickly. Please wait a moment and try again.",
          ),
        };
      case 500:
        return {
          code: "internal_error",
          message: safeMessage(
            isRecord(detail) ? detail.error : undefined,
            "Something went wrong on our end. Please try again.",
          ),
        };
      default:
        return {
          code: "unknown_error",
          message: "Something went wrong. Please try again.",
        };
    }
  }

  // apiClient's fetch() itself threw: offline, DNS failure, CORS, etc.
  return {
    code: "network_error",
    message: "Could not reach the server. Check your connection and try again.",
  };
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

/** Pull a `message` string out of an unknown error-detail shape, or fall back. */
function safeMessage(detail: unknown, fallback: string): string {
  if (isRecord(detail) && typeof detail.message === "string" && detail.message.trim()) {
    return detail.message;
  }
  return fallback;
}
