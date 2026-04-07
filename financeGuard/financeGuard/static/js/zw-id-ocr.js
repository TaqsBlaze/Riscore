/**
 * zw-id-ocr.js
 * ─────────────────────────────────────────────────────────────────────────────
 * Extracts FIRST NAME and SURNAME from a Zimbabwe National Registration ID
 * image using Tesseract.js for OCR.
 *
 * Requires Tesseract.js v5+ to be loaded before this script:
 *   <script src="https://cdn.jsdelivr.net/npm/tesseract.js@5/dist/tesseract.min.js"></script>
 *
 * Usage:
 *   const result = await ZwIdOcr.extractName(imageSource);
 *   // result → { firstName: "TANAKA PETER", surname: "CHINENGUNDU", fullName: "TANAKA PETER CHINENGUNDU" }
 *
 *   // Or read from a file input:
 *   const result = await ZwIdOcr.extractNameFromFile(fileInputElement.files[0]);
 * ─────────────────────────────────────────────────────────────────────────────
 */

(function (root, factory) {
  if (typeof module !== "undefined" && module.exports) {
    module.exports = factory();          // CommonJS / Node
  } else {
    root.ZwIdOcr = factory();            // Browser global
  }
})(typeof globalThis !== "undefined" ? globalThis : this, function () {

  // ── Helpers ────────────────────────────────────────────────────────────────

  /**
   * Strip characters that cannot appear in a name and collapse whitespace.
   */
  function cleanField(value) {
    return (value || "")
      .replace(/[^A-Za-z\s'-]/g, " ")
      .replace(/\s+/g, " ")
      .trim()
      .toUpperCase();
  }

  /**
   * Sanity-check that a captured value looks like real name text.
   * Rejects empty strings, pure numbers, and suspiciously short fragments.
   */
  function isValidNameField(value) {
    const cleaned = cleanField(value);
    if (!cleaned || cleaned.length < 2) return false;
    if (/^\d+$/.test(cleaned)) return false;
    return true;
  }

  /**
   * Core extraction logic.
   *
   * The Zimbabwe National ID lays out fields like:
   *
   *   Surname         CHINENGUNDU
   *   First Name      TANAKA PETER
   *
   * Tesseract preserves these as single lines with the label on the left and
   * the value on the right, separated only by whitespace (no colon).
   *
   * Strategy
   * ────────
   * 1. PRIMARY   — search the raw OCR text for the exact label patterns.
   * 2. SECONDARY — fall back to a line-by-line scan for lines that immediately
   *                follow a label-only line (handles occasional line-breaks).
   * 3. TERTIARY  — last resort positional heuristic using cleaned lines.
   */
  function parseOcrText(rawText) {
    const text = (rawText || "").replace(/\r/g, "");

    // ── 1. PRIMARY: inline label + value on same line ──────────────────────
    //
    // Patterns are word-boundary anchored so "First Name" won't partially
    // match "Village of Origin" etc.  The lazy quantifier stops at line-end.
    //
    const surnameInline   = text.match(
      /\bPATCH\b\s{1,30}([A-Za-z][A-Za-z\s'-]{1,50}?)(?=\n|$)/i
    );
    const firstNameInline = text.match(
      /\bFrstNave\b\s{1,30}([A-Za-z][A-Za-z\s'-]{1,50}?)(?=\n|$)/i
    );

    const surname1   = surnameInline   ? cleanField(surnameInline[1])   : "";
    const firstName1 = firstNameInline ? cleanField(firstNameInline[1]) : "";

    if (isValidNameField(surname1) && isValidNameField(firstName1)) {
      return { firstName: firstName1.toLowerCase(), surname: surname1.toLowerCase(), method: "inline" };
    }

    // ── 2. SECONDARY: label on its own line, value on the next line ─────────
    //
    // Some scans cause Tesseract to break the label and value onto separate
    // lines.  Walk through lines looking for a label line followed by a value.
    //
    const lines = text.split("\n").map((l) => l.trim()).filter(Boolean);

    let surname2   = "";
    let firstName2 = "";

    for (let i = 0; i < lines.length - 1; i++) {
      const line = lines[i];
      const next = lines[i + 1];

      if (/^\s*surname\s*$/i.test(line) && isValidNameField(next)) {
        surname2 = cleanField(next);
      }
      if (/^\s*first\s+name\s*$/i.test(line) && isValidNameField(next)) {
        firstName2 = cleanField(next);
      }
    }

    // Also try: label on one line, value embedded at the start of the same
    // line but captured by primary already — merge with whatever primary got.
    const surname3   = isValidNameField(surname1)   ? surname1   : surname2;
    const firstName3 = isValidNameField(firstName1) ? firstName1 : firstName2;

    if (isValidNameField(surname3) || isValidNameField(firstName3)) {
      return {
        firstName: firstName3,
        surname:   surname3,
        method:    "multiline",
      };
    }

    // ── 3. TERTIARY: positional heuristic ───────────────────────────────────
    //
    // On very noisy scans the labels may be garbled.  Use relative line
    // positions: on a Zimbabwe ID the layout order is roughly:
    //   ID Number → Surname → First Name → Date of Birth → …
    // Find the ID-number line and take the next two name-looking lines.
    //
    const idNumIdx = lines.findIndex((l) =>
      /\bID\s*number\b/i.test(l) || /^\d{2}-\d{7}/.test(l)
    );

    if (idNumIdx !== -1) {
      const candidates = lines
        .slice(idNumIdx + 1)
        .filter((l) => isValidNameField(l) && !/\d{2}\/\d{2}\/\d{4}/.test(l));

      if (candidates.length >= 2) {
        return {
          firstName: cleanField(candidates[1]),
          surname:   cleanField(candidates[0]),
          method:    "positional",
        };
      }
    }

    return { firstName: "", surname: "", method: "failed" };
  }

  // ── Public API ─────────────────────────────────────────────────────────────

  /**
   * Run Tesseract OCR on any image source Tesseract accepts
   * (URL, data-URL, File, Blob, HTMLImageElement, Canvas, …)
   * and extract the name fields.
   *
   * @param  {*}       imageSource  Anything Tesseract.recognize() accepts
   * @param  {object}  [options]
   * @param  {string}  [options.lang="eng"]  Tesseract language
   * @param  {boolean} [options.debug=false] Log raw OCR text to console
   * @returns {Promise<{
   *   firstName : string,
   *   surname   : string,
   *   fullName  : string,   // "FIRST_NAME SURNAME"
   *   rawText   : string,
   *   method    : string    // "inline" | "multiline" | "positional" | "failed"
   * }>}
   */
  async function extractName(imageSource, options = {}) {
    if (typeof Tesseract === "undefined") {
      throw new Error(
        "[ZwIdOcr] Tesseract.js is not loaded. " +
        "Add <script src=\"https://cdn.jsdelivr.net/npm/tesseract.js@5/dist/tesseract.min.js\"></script> " +
        "before this library."
      );
    }

    const lang  = options.lang  || "eng";
    const debug = options.debug || true;

    const result = await Tesseract.recognize(imageSource, lang);
    const rawText = result?.data?.text || "";

    if (debug) {
      console.group("[ZwIdOcr] Raw OCR output");
      console.log(rawText);
      console.groupEnd();
    }

    const parsed = parseOcrText(rawText);

    const firstName = parsed.firstName;
    const surname   = parsed.surname;
    const fullName  = [firstName, surname].filter(Boolean).join(" ");

    return { firstName, surname, fullName, rawText, method: parsed.method };
  }

  /**
   * Convenience wrapper: accepts a File object directly (e.g. from an
   * <input type="file"> change event) and resolves via extractName().
   *
   * @param  {File}    file
   * @param  {object}  [options]  Same as extractName()
   * @returns {Promise<object>}   Same shape as extractName()
   */
  function extractNameFromFile(file, options = {}) {
    return new Promise((resolve, reject) => {
      if (!(file instanceof Blob)) {
        return reject(new Error("[ZwIdOcr] extractNameFromFile expects a File or Blob."));
      }
      const reader = new FileReader();
      reader.onload  = (e) => resolve(extractName(e.target.result, options));
      reader.onerror = ()  => reject(new Error("[ZwIdOcr] Failed to read the file."));
      reader.readAsDataURL(file);
    });
  }

  /**
   * Normalise a name string for comparison: lowercase, collapse spaces,
   * strip punctuation.  Useful for matching the extracted name against
   * a user-supplied input.
   *
   * @param  {string} value
   * @returns {string}
   */
  function normalizeName(value) {
    return (value || "")
      .toLowerCase()
      .replace(/[^a-z\s'-]+/g, " ")
      .replace(/\s+/g, " ")
      .trim();
  }

  /**
   * Token-based name match: every word in `input` must appear in `idName`.
   * Order-insensitive so "Tanaka Peter Chinengundu" matches
   * "CHINENGUNDU TANAKA PETER".
   *
   * @param  {string} idName   Name extracted from the ID (e.g. fullName)
   * @param  {string} input    Name the user typed in a form field
   * @returns {boolean}
   */
  function namesMatch(idName, input) {
    if (!idName || !input) return false;
    const idTokens    = new Set(normalizeName(idName).split(/\s+/).filter(Boolean));
    const inputTokens = normalizeName(input).split(/\s+/).filter(Boolean);
    return inputTokens.length > 0 && inputTokens.every((t) => idTokens.has(t));
  }

  // ── Exports ────────────────────────────────────────────────────────────────

  return { extractName, extractNameFromFile, normalizeName, namesMatch };
});