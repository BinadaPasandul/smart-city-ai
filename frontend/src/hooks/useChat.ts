import { useCallback, useState } from "react";

import type { ChatMessage } from "@/types/chat";

/**
 * Local-only chat state for the Phase 3A UI preview.
 *
 * IMPORTANT: this does not call the backend. `sendMessage` appends the
 * user's message, then — after a short simulated delay so the "typing"
 * state is visible rather than instantaneous — appends a static assistant
 * reply explaining the assistant isn't connected yet. Real answers arrive
 * in a later phase that replaces the body of `sendMessage` with an actual
 * POST /api/v1/chat call; nothing here should be mistaken for that.
 */
const PREVIEW_REPLY =
  "I'm not connected to live city data yet — this is a preview of the chat interface. Real answers will arrive once the backend is connected in a later phase.";

const ASSISTANT_REPLY_DELAY_MS = 700;

export interface UseChatResult {
  messages: ChatMessage[];
  draft: string;
  /** Updates the draft; also marks the conversation as started once non-empty. */
  setDraft: (value: string) => void;
  /** Sends the current trimmed draft, if non-empty, and clears it. */
  sendMessage: () => void;
  isAssistantTyping: boolean;
  /** Whether the dedicated chat layout should replace the landing page. */
  hasStarted: boolean;
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

  const sendMessage = useCallback(() => {
    const content = draft.trim();
    if (!content) {
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

    window.setTimeout(() => {
      setMessages((prev) => [
        ...prev,
        {
          id: crypto.randomUUID(),
          role: "assistant",
          content: PREVIEW_REPLY,
          timestamp: new Date(),
        },
      ]);
      setIsAssistantTyping(false);
    }, ASSISTANT_REPLY_DELAY_MS);
  }, [draft]);

  return { messages, draft, setDraft, sendMessage, isAssistantTyping, hasStarted };
}
