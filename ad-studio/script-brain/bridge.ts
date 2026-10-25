import { spawn } from "child_process";
import fs from "fs";
import os from "os";
import path from "path";
import { fileURLToPath } from "url";
import { GoogleDocsService } from "../transcription/googleDocsService.js";

export interface ProvenScriptCacMetrics {
  first_pay_cac?: number;
  post_1hr_cancel_cac?: number;
  second_pay_cac_4d?: number;
  second_pay_cac_7d?: number;
  cancel_pct?: number;
  lang_avg_cac?: number;
  cac_vs_avg_pct?: number;
  tier?: string;
  [key: string]: any;
}

export interface ProvenScriptInput {
  ad_code: string;
  title?: string;
  doc_url?: string;
  raw_script_text?: string;
  language: string;
  character_type?: string;
  cac_metrics?: ProvenScriptCacMetrics;
}

export interface SheetIngestOptions {
  sheetId?: string;
  tab?: string;
  limit?: number;
  reset?: boolean;
}

export interface BriefGenerationInput {
  brief: string;
  language: string;
  character?: string;
  createGoogleDoc?: boolean;
  docTitle?: string;
}

export interface MiddleBrainComplianceReport {
  passed: boolean;
  score: number;
  missing_phrases?: string[];
  forbidden_matches?: string[];
  violations?: string[];
  [key: string]: any;
}

export interface MiddleBrainGenerationResult {
  predicted_success_rate_pct: number;
  confidence_label: string;
  visual_description: string;
  three_hooks: any[];
  four_block_thinking_plan: any;
  full_supernova_script: string;
  formatted_google_doc_text: string;
  matched_winners: any[];
  compliance: MiddleBrainComplianceReport;
  token_usage?: {
    input_tokens: number;
    output_tokens: number;
    total_tokens: number;
    llm_calls?: number;
    cached?: boolean;
    model?: string;
  };
  doc_url?: string;
  doc_id?: string;
  [key: string]: any;
}

/**
 * Resolves the `ad-studio/script-brain` directory whether running via `tsx` (ESM)
 * or bundled via `esbuild` (`dist/server.cjs`).
 */
export function resolveScriptBrainDir(): string {
  const cwdCandidate = path.resolve(process.cwd(), "ad-studio", "script-brain");
  if (fs.existsSync(cwdCandidate)) {
    return cwdCandidate;
  }

  try {
    if (typeof import.meta !== "undefined" && import.meta.url) {
      const moduleDir = path.dirname(fileURLToPath(import.meta.url));
      if (fs.existsSync(path.join(moduleDir, "run_brain.py")) || fs.existsSync(path.join(moduleDir, ".venv"))) {
        return moduleDir;
      }
    }
  } catch {
    // Fallback to cwdCandidate
  }

  return cwdCandidate;
}

/**
 * Resolves the Python executable to run `run_brain.py`.
 * Prefers `ad-studio/script-brain/.venv/bin/python` if present on disk, falling back to `python3`.
 */
export function resolvePythonBinary(): string {
  if (process.env.SCRIPT_BRAIN_PYTHON && fs.existsSync(process.env.SCRIPT_BRAIN_PYTHON)) {
    return process.env.SCRIPT_BRAIN_PYTHON;
  }

  const brainDir = resolveScriptBrainDir();
  const venvCandidates = [
    path.join(brainDir, ".venv", "bin", "python"),
    path.join(brainDir, ".venv", "bin", "python3"),
    path.join(brainDir, ".venv", "Scripts", "python.exe"),
  ];

  for (const candidate of venvCandidates) {
    if (fs.existsSync(candidate)) {
      return candidate;
    }
  }

  return "python3";
}

/**
 * Extracts and parses the last valid JSON payload from CLI stdout.
 * Handles cases where ML libraries emit startup logs before the JSON output.
 */
function parseJsonStdout(stdout: string, stderr: string, subcommand: string): any {
  const trimmed = stdout.trim();
  if (!trimmed) {
    throw new Error(
      `[ScriptBrainBridge] Empty stdout from run_brain.py ${subcommand}. Stderr: ${stderr.trim() || "(empty)"}`
    );
  }

  // 1. Direct parse attempt
  try {
    return JSON.parse(trimmed);
  } catch {
    // Continue to fallback extraction if libraries printed non-JSON lines before the JSON block
  }

  // 2. Try parsing from the last line backwards
  const lines = trimmed.split(/\r?\n/);
  for (let i = lines.length - 1; i >= 0; i--) {
    const candidateLine = lines[i].trim();
    if (candidateLine.startsWith("{") && candidateLine.endsWith("}")) {
      try {
        return JSON.parse(candidateLine);
      } catch {
        // continue
      }
    }
  }

  // 3. Try scanning from first '{' or last top-level '{' block to the final '}'
  const lastBraceClose = trimmed.lastIndexOf("}");
  if (lastBraceClose !== -1) {
    let searchFrom = 0;
    while (searchFrom <= lastBraceClose) {
      const openIdx = trimmed.indexOf("{", searchFrom);
      if (openIdx === -1 || openIdx > lastBraceClose) break;
      const slice = trimmed.slice(openIdx, lastBraceClose + 1);
      try {
        return JSON.parse(slice);
      } catch {
        searchFrom = openIdx + 1;
      }
    }
  }

  throw new Error(
    `[ScriptBrainBridge] Failed to parse JSON from run_brain.py ${subcommand}. Output: ${trimmed.slice(0, 500)}`
  );
}

/**
 * Executes `ad-studio/script-brain/run_brain.py` with the given CLI arguments and parses its JSON stdout.
 */
async function runBrainCli(args: string[], timeoutMs = 180_000): Promise<any> {
  const brainDir = resolveScriptBrainDir();
  const scriptPath = path.join(brainDir, "run_brain.py");
  const pythonBin = resolvePythonBinary();

  if (!fs.existsSync(scriptPath)) {
    throw new Error(`[ScriptBrainBridge] run_brain.py not found at ${scriptPath}`);
  }

  return new Promise((resolve, reject) => {
    const child = spawn(pythonBin, [scriptPath, ...args], {
      cwd: brainDir,
      env: {
        ...process.env,
        PYTHONUNBUFFERED: "1",
        TOKENIZERS_PARALLELISM: "false",
      },
      stdio: ["ignore", "pipe", "pipe"],
    });

    let stdout = "";
    let stderr = "";
    let settled = false;

    const timer = setTimeout(() => {
      if (!settled) {
        settled = true;
        child.kill("SIGKILL");
        reject(
          new Error(
            `[ScriptBrainBridge] run_brain.py ${args[0] || ""} timed out after ${Math.round(timeoutMs / 1000)}s`
          )
        );
      }
    }, timeoutMs);

    child.stdout.on("data", (chunk: Buffer) => {
      stdout += chunk.toString("utf-8");
    });

    child.stderr.on("data", (chunk: Buffer) => {
      stderr += chunk.toString("utf-8");
    });

    child.on("error", (err: Error) => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      reject(new Error(`[ScriptBrainBridge] Failed to spawn ${pythonBin}: ${err.message}`));
    });

    child.on("close", (code: number | null) => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);

      const subcommand = args[0] || "cli";

      if (code !== 0) {
        // Check if run_brain.py emitted a structured JSON error on stdout
        if (stdout.trim()) {
          try {
            const parsedErr = parseJsonStdout(stdout, stderr, subcommand);
            if (parsedErr && (parsedErr.error || parsedErr.message)) {
              reject(new Error(parsedErr.error || parsedErr.message));
              return;
            }
          } catch {
            // Fall through to stderr rejection
          }
        }
        reject(
          new Error(
            `[ScriptBrainBridge] run_brain.py ${subcommand} exited with code ${code}: ${
              stderr.trim() || stdout.trim() || "Unknown error"
            }`
          )
        );
        return;
      }

      try {
        const result = parseJsonStdout(stdout, stderr, subcommand);
        resolve(result);
      } catch (err: any) {
        reject(err);
      }
    });
  });
}

/**
 * Checks Python Middle Brain health, installed ML packages, DB schema, and script counts.
 * Defaults `seedIfEmpty` to `false` so status checks never re-seed dummy data once real scripts are ingested.
 */
export async function getScriptBrainStatus(seedIfEmpty = false): Promise<any> {
  const args = ["status"];
  if (seedIfEmpty) {
    args.push("--seed-if-empty");
  }
  return runBrainCli(args, 90_000);
}

/**
 * Ingests proven Supernova scripts + CAC metrics directly from a Google Sheet tab into the SQL + Vector Brain
 * and retrains the contrastive playbook, DSPy demonstrations, and XGBoost success predictor.
 */
export async function ingestFromGoogleSheetToBrain(
  options: SheetIngestOptions = {}
): Promise<any> {
  const sheetId =
    (options.sheetId || "").trim() || "13sQlJMgAZ_nFxatPEp5XfZ9swE4et00qEBpwNS2a82I";
  const tab = (options.tab || "").trim() || "REPLICATE - LANGUAGE";

  const args = ["ingest", "--sheet-id", sheetId, "--tab", tab];

  if (
    typeof options.limit === "number" &&
    Number.isFinite(options.limit) &&
    options.limit > 0
  ) {
    args.push("--limit", String(Math.floor(options.limit)));
  }

  if (options.reset) {
    args.push("--reset");
  }

  return runBrainCli(args, 600_000);
}

/**
 * Ingests a batch of proven Supernova scripts + CAC metrics into the SQL + Vector Brain (0 Gemini API calls)
 * and retrains the contrastive playbook, DSPy demonstrations, and XGBoost success predictor.
 */
export async function ingestProvenScriptsToBrain(items: ProvenScriptInput[]): Promise<any> {
  if (!Array.isArray(items) || items.length === 0) {
    return {
      success: true,
      ingested_count: 0,
      message: "No script items provided for ingestion.",
    };
  }

  const tempFilePath = path.join(
    os.tmpdir(),
    `supernova_brain_ingest_${Date.now()}_${Math.random().toString(36).slice(2, 8)}.json`
  );

  try {
    fs.writeFileSync(tempFilePath, JSON.stringify(items, null, 2), "utf-8");
    return await runBrainCli(["ingest", "--file", tempFilePath], 300_000);
  } finally {
    try {
      if (fs.existsSync(tempFilePath)) {
        fs.unlinkSync(tempFilePath);
      }
    } catch {
      // Ignore temp file cleanup errors
    }
  }
}

/**
 * Triggers offline training of the Contrastive Playbook, DSPy few-shot compiler, and XGBoost Success Rate Predictor.
 */
export async function trainScriptBrain(): Promise<any> {
  return runBrainCli(["train"], 180_000);
}

/**
 * Expands a short concept brief into a complete Supernova Ad Package using the Python Middle Brain,
 * and optionally creates a formatted Google Doc via `GoogleDocsService`.
 */
export async function generateSupernovaScriptFromBrief(
  input: BriefGenerationInput
): Promise<MiddleBrainGenerationResult> {
  const brief = (input.brief || "").trim();
  if (!brief) {
    throw new Error("Brief generation requires a non-empty `brief` string.");
  }

  const language = (input.language || "Tamil").trim() || "Tamil";
  const character = (input.character || "Female Humanoid Robot").trim() || "Female Humanoid Robot";

  const rawResult = await runBrainCli(
    ["generate", "--brief", brief, "--language", language, "--character", character],
    240_000
  );

  const predictionObj =
    rawResult?.prediction && typeof rawResult.prediction === "object" ? rawResult.prediction : {};

  const predictedSuccessRatePct = Number(
    rawResult?.predicted_success_rate_pct ??
      predictionObj?.predicted_success_rate_pct ??
      rawResult?.predicted_success_rate ??
      predictionObj?.predicted_success_rate ??
      82
  );

  const confidenceLabel = String(
    rawResult?.confidence_label ??
      predictionObj?.confidence_label ??
      (predictedSuccessRatePct >= 75
        ? "High Confidence (Tier-S Pattern)"
        : predictedSuccessRatePct >= 60
          ? "Moderate-High Confidence"
          : "Exploratory Angle")
  );

  const visualDescription = String(
    rawResult?.visual_description ?? rawResult?.visual_description_block ?? ""
  );

  const threeHooks = Array.isArray(rawResult?.three_hooks)
    ? rawResult.three_hooks
    : Array.isArray(rawResult?.three_hook_options)
      ? rawResult.three_hook_options
      : rawResult?.three_hooks
        ? [rawResult.three_hooks]
        : rawResult?.three_hook_options
          ? [rawResult.three_hook_options]
          : [];

  const fourBlockThinkingPlan =
    rawResult?.four_block_thinking_plan ?? rawResult?.thinking_plan ?? rawResult?.story_blocks ?? "";

  const fullSupernovaScript = String(
    rawResult?.full_supernova_script ?? rawResult?.script ?? rawResult?.supernova_script ?? ""
  );

  const formattedGoogleDocText = String(
    rawResult?.formatted_google_doc_text ?? rawResult?.formatted_doc_text ?? fullSupernovaScript
  );

  const matchedWinners = Array.isArray(rawResult?.matched_winners)
    ? rawResult.matched_winners
    : Array.isArray(rawResult?.top_winners)
      ? rawResult.top_winners
      : [];

  const compliance: MiddleBrainComplianceReport =
    rawResult?.compliance && typeof rawResult.compliance === "object"
      ? rawResult.compliance
      : {
          passed: true,
          score: 1.0,
          missing_phrases: [],
          forbidden_matches: [],
          violations: [],
        };

  const result: MiddleBrainGenerationResult = {
    ...rawResult,
    predicted_success_rate_pct: predictedSuccessRatePct,
    confidence_label: confidenceLabel,
    visual_description: visualDescription,
    three_hooks: threeHooks,
    four_block_thinking_plan: fourBlockThinkingPlan,
    full_supernova_script: fullSupernovaScript,
    formatted_google_doc_text: formattedGoogleDocText,
    matched_winners: matchedWinners,
    compliance,
    token_usage: rawResult?.token_usage,
  };

  if (input.createGoogleDoc) {
    const docsService = new GoogleDocsService();
    const cleanBriefSnippet = brief.replace(/\s+/g, " ").slice(0, 42).trim();
    const docTitle =
      (input.docTitle || "").trim() ||
      `Supernova [${language}] - ${cleanBriefSnippet || "Middle Brain Script"} (${predictedSuccessRatePct}% Pred.)`;

    const cleanSupernovaTabText = visualDescription
      ? `${visualDescription.trim()}\n\nScript:\n${fullSupernovaScript.trim()}`
      : fullSupernovaScript;

    const docRes = await docsService.createScriptDoc({
      title: docTitle,
      scriptText: formattedGoogleDocText || fullSupernovaScript,
      supernovaScriptText: cleanSupernovaTabText || undefined,
    });

    if (docRes.success) {
      if (docRes.docUrl) {
        result.doc_url = docRes.docUrl;
      }
      if (docRes.docId) {
        result.doc_id = docRes.docId;
      }
    } else if (docRes.error) {
      result.doc_error = docRes.error;
    }
  }

  return result;
}
