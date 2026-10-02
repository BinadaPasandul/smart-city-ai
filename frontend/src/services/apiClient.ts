/**
 * Thin fetch wrapper foundation for talking to the backend API.
 *
 * Not called anywhere yet — this only establishes the base URL, error
 * handling, and auth-header shape so the chat feature can be built on top
 * of a single, consistent client later instead of ad-hoc fetch calls.
 */

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL;

export class ApiError extends Error {
  readonly status: number;
  readonly body: unknown;

  constructor(message: string, status: number, body: unknown) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.body = body;
  }
}

interface RequestOptions {
  /** Bearer token, once auth is enabled backend-side. Omitted when null. */
  token?: string | null;
  signal?: AbortSignal;
}

async function request<TResponse>(
  path: string,
  init: RequestInit,
  options: RequestOptions = {},
): Promise<TResponse> {
  const headers = new Headers(init.headers);
  headers.set("Content-Type", "application/json");
  if (options.token) {
    headers.set("Authorization", `Bearer ${options.token}`);
  }

  const response = await fetch(`${API_BASE_URL}${path}`, {
    ...init,
    headers,
    signal: options.signal,
    // The backend's CORS policy disables credentialed requests
    // (allow_credentials=False); auth is via the Authorization header only.
    credentials: "omit",
  });

  const contentType = response.headers.get("content-type") ?? "";
  const body = contentType.includes("application/json")
    ? await response.json()
    : await response.text();

  if (!response.ok) {
    throw new ApiError(
      `Request to ${path} failed with status ${response.status}`,
      response.status,
      body,
    );
  }

  return body as TResponse;
}

export const apiClient = {
  get: <TResponse>(path: string, options?: RequestOptions) =>
    request<TResponse>(path, { method: "GET" }, options),

  post: <TResponse, TBody = unknown>(
    path: string,
    body: TBody,
    options?: RequestOptions,
  ) =>
    request<TResponse>(
      path,
      { method: "POST", body: JSON.stringify(body) },
      options,
    ),
};
