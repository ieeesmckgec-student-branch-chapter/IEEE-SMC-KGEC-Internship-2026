// src/lib/scoring-api.ts
export const API_BASE_URL = (import.meta.env.VITE_API_URL ?? "").replace(/\/$/, "") || window.location.origin;

/**
 * POST path for scoring a statement.
 */
export const SCORE_ENDPOINT = "/score";
export const SCORE_FILE_FIELD = "file";

export class ScoringApiError extends Error {
  status?: number;
  constructor(message: string, status?: number) {
    super(message);
    this.name = "ScoringApiError";
    this.status = status;
  }
}

export async function scoreStatement(file: File, opts: { signal?: AbortSignal } = {}) {
  const formData = new FormData();
  formData.append(SCORE_FILE_FIELD, file, file.name);

  const url = `${API_BASE_URL}${SCORE_ENDPOINT}`.replace(/([^:]\/)\/+/g, "$1"); // normalize

  let res: Response;
  try {
    res = await fetch(url, {
      method: "POST",
      body: formData,
      signal: opts.signal,
      credentials: "include", // include cookies if server sets them
    });
  } catch (err: any) {
    // Improve diagnostics for common browser network errors
    const msg = (() => {
      if (err && err.name === "AbortError") return "Request aborted.";
      if (err && err.message) {
        // common messages: "Failed to fetch", "NetworkError when attempting to fetch resource."
        if (err.message.includes("Failed to fetch") || err.message.includes("NetworkError")) {
          return `Network error reaching scoring API at ${url}. Possible causes: API not running, CORS blocked, mixed-content (HTTPS page requesting HTTP), or DNS/network issue. See browser DevTools Network tab for details.`;
        }
        return `Network error: ${err.message}`;
      }
      return `Could not reach the scoring API at ${url}.`;
    })();
    throw new ScoringApiError(msg);
  }

  if (!res.ok) {
    let detail = "";
    try {
      detail = await res.text();
    } catch {
      /* ignore */
    }
    throw new ScoringApiError(
      `Scoring API returned ${res.status} ${res.statusText}.${detail ? ` ${detail.slice(0, 300)}` : ""}`,
      res.status,
    );
  }

  const contentType = (res.headers.get("content-type") || "").toLowerCase();
  const bodyText = await res.text();

  if (contentType.includes("application/json") || contentType.includes("application/ld+json")) {
    try {
      return JSON.parse(bodyText);
    } catch {
      throw new ScoringApiError(
        `Scoring API said it returned JSON but the body wasn't parseable. First 300 chars: ${bodyText.slice(0, 300)}`,
      );
    }
  }

  // Try to extract JSON blob from HTML
  const match = bodyText.match(/\{[\s\S]*\}/);
  if (match) {
    try {
      return JSON.parse(match[0]);
    } catch {
      // fall through
    }
  }

  throw new ScoringApiError(
    `Scoring API returned a non-JSON response (content-type: ${contentType || "unknown"}). First 300 chars: ${bodyText.slice(0, 300)}`,
  );
}