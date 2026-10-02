import { useCallback, useState } from "react";

const STORAGE_KEY = "smart-city-ai.auth-token";

/**
 * Minimal foundation for a future bearer token. No login UI or auth flow
 * exists yet — this only gives later work a single, consistent place to
 * read/write the token so the rest of the app never touches storage
 * directly. The backend's /chat endpoint does not require a token today
 * (AUTH_ENABLED=false by default) but is built to require one later.
 */
export function useAuthToken() {
  const [token, setTokenState] = useState<string | null>(() => {
    try {
      return localStorage.getItem(STORAGE_KEY);
    } catch {
      return null;
    }
  });

  const setToken = useCallback((next: string | null) => {
    setTokenState(next);
    try {
      if (next) {
        localStorage.setItem(STORAGE_KEY, next);
      } else {
        localStorage.removeItem(STORAGE_KEY);
      }
    } catch {
      // Storage can be unavailable (private browsing, blocked cookies, etc.);
      // the in-memory state above still works for the current page session.
    }
  }, []);

  return { token, setToken, isAuthenticated: token !== null };
}
