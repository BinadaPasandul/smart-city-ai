/**
 * Types mirroring the backend's public chat contract exactly
 * (backend/app/api/schemas/chat.py — ChatRequest/ChatResponse/ChatSource/
 * ChatMetadata/ChatError), as confirmed against the running backend.
 *
 * These exist so the eventual chat feature can be built against a typed
 * contract from day one. Nothing here is wired up to the UI yet.
 */

/** The three specialist agents the orchestrator can currently select. */
export type SpecialistAgentName = "mobility" | "environment" | "public_services";

/** Stable error categories the backend can return in `error.code`. */
export type ChatErrorCode =
  | "invalid_request"
  | "agent_not_found"
  | "agent_execution_failed"
  | "timeout"
  | "unsupported_request"
  | "needs_clarification";

export interface ChatRequest {
  /** 1–5000 chars (see backend CHAT_MAX_MESSAGE_LENGTH); trimmed server-side. */
  message: string;
  /**
   * Optional structured context (e.g. `{ latitude, longitude }` or
   * `{ location }`). The backend now reaches specialists correctly via
   * `SpecialistContextAdapter`, but this integration phase intentionally
   * sends `{}` — see useChat.ts — rather than inventing browser geolocation
   * or other context that wasn't asked for.
   */
  context?: Record<string, unknown>;
}

export interface ChatSource {
  name: string;
  source_type: string;
  url: string | null;
  retrieved_at: string | null;
  metadata: Record<string, unknown>;
}

export interface ChatError {
  code: ChatErrorCode | (string & {});
  message: string;
}

export type ExecutionStatus = "complete" | "partial_success" | "failed";

export interface ChatMetadata {
  routing_method: string | null;
  understanding_method: string | null;
  nlp_confidence: number | null;
  gemini_understanding_fallback_used: boolean;
  selected_agents: SpecialistAgentName[];
  execution_status: ExecutionStatus | null;
  successful_agents: SpecialistAgentName[];
  failed_agents: SpecialistAgentName[];
  synthesis_method: string | null;
  synthesis_used_agents: SpecialistAgentName[];
  synthesis_limitations: string[];
  web_search_used: boolean;
  web_search_status: string | null;
  web_search_provider: string | null;
  web_search_result_count: number;
  answer_basis: "specialist_plus_web" | "web_fallback" | "specialist" | null;
}

export interface ChatResponse {
  request_id: string;
  success: boolean;
  answer: string;
  sources: ChatSource[];
  metadata: ChatMetadata;
  error: ChatError | null;
}

/* -------------------------------------------------------------------------
 * Local UI state types
 *
 * These are NOT the backend contract above — they exist purely in React
 * state for the chat UI. A `ChatMessage` is not a `ChatResponse`: it's the
 * result of useChat's mapping layer flattening one real (or failed) backend
 * call into something a message bubble can render, keeping only the pieces
 * worth preserving for a later sources/metadata UI rather than the whole
 * response object.
 * ---------------------------------------------------------------------- */

export type MessageRole = "user" | "assistant";

/**
 * The subset of `ChatMetadata` worth keeping on a message today. Deliberately
 * not the whole `ChatMetadata` shape (see module docstring) — e.g. Gemini
 * retry bookkeeping and web-search internals aren't rendered anywhere yet.
 */
export interface ChatMessageMetadata {
  selected_agents: SpecialistAgentName[];
  execution_status: ExecutionStatus | null;
  successful_agents: SpecialistAgentName[];
  failed_agents: SpecialistAgentName[];
  synthesis_method: string | null;
  answer_basis: ChatMetadata["answer_basis"];
}

export interface ChatMessage {
  id: string;
  role: MessageRole;
  content: string;
  timestamp: Date;
  /**
   * Only present on assistant messages that came from a real backend call
   * (absent for the user's own messages, and for client-side network-error
   * messages that never reached the backend at all).
   */
  sources?: ChatSource[];
  metadata?: ChatMessageMetadata;
  /** Set when the message represents a logical backend failure or a client-side error. */
  error?: ChatError;
}
