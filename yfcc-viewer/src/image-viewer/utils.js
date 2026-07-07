/**
 * Parse a failed fetch response and return the error message.
 * Tries to extract `error` from JSON body first, falls back to status info.
 */
export async function getErrorMessage(res) {
  try {
    const data = await res.json();
    if (data?.error) return data.error;
  } catch {
    /* ignore parse errors */
  }
  return `${res.status} ${res.statusText}`;
}
