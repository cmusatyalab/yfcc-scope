/**
 * Parse a failed fetch response and return the error message.
 * Tries to extract `error` from JSON body first, falls back to status info.
 */
export async function getErrorMessage(response) {
  try {
    const body = await response.json();
    if (body && typeof body.error === "string" && body.error.trim()) {
      return body.error;
    }
  } catch {
    // Ignore non-JSON error responses.
  }
  return response.statusText || `HTTP ${response.status}`;
}
