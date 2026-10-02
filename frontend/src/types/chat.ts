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
   * Optional structured context. As of the last backend inspection this is
   * NOT reliably delivered to specialist agents end-to-end — prefer putting
   * location/intent in `message` until that is confirmed fixed server-side.
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
 * Local UI state types (Phase 3A)
 *
 * These are NOT the backend contract above — they exist purely in React
 * state for the local-only chat preview. A `ChatMessage` is not a
 * `ChatResponse`; there's no `sources`/`metadata`/`error` here because
 * nothing has actually been retrieved from the backend yet.
 * ---------------------------------------------------------------------- */

export type MessageRole = "user" | "assistant";

export interface ChatMessage {
  id: string;
  role: MessageRole;
  content: string;
  timestamp: Date;
}
