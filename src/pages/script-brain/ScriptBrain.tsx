import React, { useState, useEffect, useCallback } from "react";
import { Link } from "react-router-dom";
import {
  ArrowLeft,
  Sparkles,
  Brain,
  Database,
  TrendingUp,
  CheckCircle2,
  AlertCircle,
  Loader2,
  Copy,
  Check,
  Plus,
  RefreshCw,
  ExternalLink,
  Zap,
  FileText,
  Sliders,
  ChevronDown,
  ChevronUp,
  FileSpreadsheet,
  Search,
  Filter,
} from "lucide-react";

const REPLICATION_SHEET_NAME = "Automation - Replication Queue W43";
const REPLICATION_SHEET_ID = "13sQlJMgAZ_nFxatPEp5XfZ9swE4et00qEBpwNS2a82I";
const REPLICATION_SHEET_URL = `https://docs.google.com/spreadsheets/d/${REPLICATION_SHEET_ID}/edit`;

const REPLICATION_SHEET_TABS = [
  "REPLICATE - LANGUAGE",
  "REPLICATE - CHARACTER",
  "REPLICATE - LANGUAGE - WEAK 2ND PAY",
  "REPLICATE - CHARACTER - WEAK 2ND PAY",
  "NOT PROPOSED - LANGUAGE",
] as const;

const SYNC_LIMIT_OPTIONS = [
  { value: "5", label: "5 scripts" },
  { value: "20", label: "20 scripts" },
  { value: "50", label: "50 scripts" },
  { value: "all", label: "All scripts" },
] as const;

interface BrainPackageStatus {
  installed: boolean;
  version: string | null;
}

interface BrainStatusResponse {
  status: string;
  python_version: string;
  gemini_configured: boolean;
  packages: Record<string, BrainPackageStatus>;
  db_stats: {
    total_count: number;
    total_scripts?: number;
    by_cac_tier: Record<string, number>;
    by_language: Record<string, number>;
    postgres_active: boolean;
    storage_backend: string;
  };
}

interface ProvenRecord {
  ad_code: string;
  title: string;
  doc_url?: string;
  language: string;
  character_type: string;
  cac_tier: "TIER_S" | "TIER_A" | "TIER_B";
  spends?: number | string | null;
  spend?: number | string | null;
  available_tabs?: Array<string | { title?: string; tab_id?: string }>;
  regional_tabs?: Array<string | { title?: string; tab_id?: string }>;
  doc_tabs?: Array<string | { title?: string; tab_id?: string }>;
  tabs?: Array<string | { title?: string; tab_id?: string }>;
  selected_tab_title?: string;
  source_tab?: string;
  cac_metrics: {
    spends?: number | string | null;
    spend?: number | string | null;
    total_spend?: number | string | null;
    ad_spend?: number | string | null;
    first_pay_cac?: number | null;
    post_1hr_cancel_cac?: number | null;
    second_pay_cac_4d?: number | null;
    second_pay_cac_7d?: number | null;
    cancel_pct?: number | null;
    lang_avg_cac?: number | null;
    cac_vs_avg_pct?: number | null;
    [key: string]: any;
  };
  english_script: string;
  native_script?: string;
  story_blocks: {
    block_1_hook?: string;
    block_2_pain_and_old_methods?: string;
    block_3_live_fix_and_usps?: string;
    block_4_desire_and_cta?: string;
  };
  visual_blueprint: {
    split_screen_type?: string;
    hook_visuals?: string;
    body_visuals?: string;
    video_flow?: string;
    characters_description?: string;
    raw_visual_text?: string;
  };
  metadata_tags: {
    persona?: string;
    setting?: string;
    hook_angle?: string;
    hook_word_count?: number;
    avg_words_per_line?: number;
    total_turns?: number;
    ai_teacher_entry_turn?: number;
    available_tabs?: Array<string | { title?: string }>;
    regional_tabs?: Array<string | { title?: string }>;
    doc_tabs?: Array<string | { title?: string }>;
    tabs?: Array<string | { title?: string }>;
    selected_tab_title?: string;
    source_tab?: string;
    spends?: number | string | null;
    [key: string]: any;
  };
  updated_at?: string;
}

interface GenerationResult {
  predicted_success_rate_pct: number;
  confidence_label: string;
  historical_cluster_win_rate_pct?: number;
  pacing_alignment_score_pct?: number;
  compliance_score_pct?: number;
  explanation_bullets?: string[];
  visual_description: string;
  three_hooks: string[];
  four_block_thinking_plan: string;
  full_supernova_script: string;
  formatted_google_doc_text: string;
  matched_winners: Array<{
    ad_code: string;
    title?: string;
    language?: string;
    cac_tier?: string;
    cosine_similarity?: number;
    cac_metrics?: { cac_vs_avg_pct?: number };
    metadata_tags?: { persona?: string; hook_angle?: string };
  }>;
  compliance: {
    passed: boolean;
    score: number;
    missing_phrases: string[];
    forbidden_matches: string[];
  };
  doc_url?: string;
  doc_id?: string;
  token_usage?: {
    input_tokens: number;
    output_tokens: number;
    total_tokens: number;
    llm_calls?: number;
    cached?: boolean;
    model?: string;
  };
}

const SAMPLE_BRIEFS = [
  {
    label: "Tamil — Mother & School Diary Mistake",
    language: "Tamil",
    character: "Female Humanoid Robot",
    brief:
      "A homemaker mother in Chennai tries to read her son's English school circular out loud at the study table, mispronounces a simple sentence, and her 9-year-old son corrects her in front of her husband.",
  },
  {
    label: "Telugu — IT Standup Call Freeze",
    language: "Telugu",
    character: "Female Robot",
    brief:
      "A software engineer in Hyderabad knows all the technical work inside out, but during the morning Zoom standup call with the US client he says broken English ('Yesterday my task is completed only') and his junior colleagues smirk.",
  },
  {
    label: "Hindi — Restaurant Menu Hesitation",
    language: "Hindi",
    character: "Miss Nova",
    brief:
      "A father takes his family to a nice cafe on Sunday, gets confused when the waiter asks a fast question in English, and tells the waiter in broken English to just bring whatever the next table is eating.",
  },
];

function formatInrMetric(val: number | string | null | undefined): string {
  if (val === null || val === undefined || val === "") return "—";
  if (typeof val === "number") {
    if (!Number.isFinite(val)) return "—";
    return `₹${val.toLocaleString("en-IN", { maximumFractionDigits: 1 })}`;
  }
  const trimmed = String(val).trim();
  if (!trimmed || trimmed === "-" || trimmed.toLowerCase() === "na" || trimmed.toLowerCase() === "n/a") {
    return "—";
  }
  const num = Number(trimmed.replace(/[₹$,\s]/g, ""));
  if (Number.isFinite(num) && trimmed.replace(/[₹$,.\-\d\s]/g, "") === "") {
    return `₹${num.toLocaleString("en-IN", { maximumFractionDigits: 1 })}`;
  }
  return trimmed.startsWith("₹") ? trimmed : `₹${trimmed}`;
}

function getScriptSpends(r: ProvenRecord): string {
  const rawSpend =
    r.cac_metrics?.spends ??
    r.cac_metrics?.spend ??
    r.cac_metrics?.total_spend ??
    r.cac_metrics?.ad_spend ??
    r.spends ??
    r.spend ??
    r.metadata_tags?.spends ??
    r.metadata_tags?.spend;
  return formatInrMetric(rawSpend);
}

function getRegionalDocTabs(r: ProvenRecord): string[] {
  const candidates = [
    r.available_tabs,
    r.regional_tabs,
    r.doc_tabs,
    r.tabs,
    r.metadata_tags?.available_tabs,
    r.metadata_tags?.regional_tabs,
    r.metadata_tags?.doc_tabs,
    r.metadata_tags?.tabs,
    r.cac_metrics?.available_tabs,
  ];
  for (const list of candidates) {
    if (Array.isArray(list) && list.length > 0) {
      const normalized = list
        .map((item) => (typeof item === "string" ? item.trim() : String(item?.title || "").trim()))
        .filter(Boolean);
      if (normalized.length > 0) {
        return Array.from(new Set(normalized));
      }
    }
  }
  const singleTab = r.selected_tab_title || r.metadata_tags?.selected_tab_title;
  if (singleTab && String(singleTab).trim()) {
    return [String(singleTab).trim()];
  }
  return [];
}

export default function ScriptBrain() {
  const [activeTab, setActiveTab] = useState<"generate" | "database" | "architecture">("generate");
  const [statusData, setStatusData] = useState<BrainStatusResponse | null>(null);
  const [records, setRecords] = useState<ProvenRecord[]>([]);
  const [playbook, setPlaybook] = useState<any>(null);
  const [storeUpdatedAt, setStoreUpdatedAt] = useState<string | null>(null);
  const [loadingStatus, setLoadingStatus] = useState(true);

  // Google Sheet Replication Queue Sync state
  const [selectedSheetTab, setSelectedSheetTab] = useState<string>(REPLICATION_SHEET_TABS[0]);
  const [syncLimit, setSyncLimit] = useState<string>("20");
  const [replaceExistingStore, setReplaceExistingStore] = useState<boolean>(false);
  const [syncingSheet, setSyncingSheet] = useState<boolean>(false);
  const [syncMessage, setSyncMessage] = useState<{
    type: "success" | "error";
    text: string;
    details?: string;
  } | null>(null);

  // Filter & Search state for Trained Scripts table
  const [searchQuery, setSearchQuery] = useState("");
  const [tierFilter, setTierFilter] = useState<"ALL" | "TIER_S" | "TIER_A" | "TIER_B">("ALL");
  const [langFilter, setLangFilter] = useState<string>("ALL");

  // Generator state
  const [brief, setBrief] = useState(SAMPLE_BRIEFS[0].brief);
  const [language, setLanguage] = useState(SAMPLE_BRIEFS[0].language);
  const [character, setCharacter] = useState(SAMPLE_BRIEFS[0].character);
  const [createGoogleDoc, setCreateGoogleDoc] = useState(false);
  const [docTitle, setDocTitle] = useState("");
  const [generating, setGenerating] = useState(false);
  const [genError, setGenError] = useState<string | null>(null);
  const [result, setResult] = useState<GenerationResult | null>(null);
  const [copiedKey, setCopiedKey] = useState<string | null>(null);

  // Manual Ingestion modal / state
  const [showIngestForm, setShowIngestForm] = useState(false);
  const [newAdCode, setNewAdCode] = useState("");
  const [newTitle, setNewTitle] = useState("");
  const [newLanguage, setNewLanguage] = useState("Tamil");
  const [newCharacter, setNewCharacter] = useState("Female Humanoid Robot");
  const [newDocUrl, setNewDocUrl] = useState("");
  const [newRawScript, setNewRawScript] = useState("");
  const [newSecondPayCac, setNewSecondPayCac] = useState("420");
  const [newLangAvgCac, setNewLangAvgCac] = useState("540");
  const [newCancelPct, setNewCancelPct] = useState("11.5");
  const [ingesting, setIngesting] = useState(false);
  const [training, setTraining] = useState(false);
  const [expandedAdCodes, setExpandedAdCodes] = useState<Record<string, boolean>>({});

  const toggleExpandScript = (adCode: string) => {
    setExpandedAdCodes((prev) => ({
      ...prev,
      [adCode]: !prev[adCode],
    }));
  };

  const loadBrainData = useCallback(async () => {
    setLoadingStatus(true);
    try {
      const [statusRes, scriptsRes] = await Promise.all([
        fetch("/api/script-brain/status"),
        fetch("/api/script-brain/scripts"),
      ]);
      if (statusRes.ok) {
        const s = await statusRes.json();
        setStatusData(s);
      }
      if (scriptsRes.ok) {
        const r = await scriptsRes.json();
        const loadedScripts: ProvenRecord[] = r.scripts || r.records || [];
        setRecords(loadedScripts);
        setPlaybook(r.playbook || null);
        if (r.updated_at) {
          setStoreUpdatedAt(r.updated_at);
        }
      } else {
        // Fallback to /api/script-brain/records if /scripts is unavailable
        const fallbackRes = await fetch("/api/script-brain/records");
        if (fallbackRes.ok) {
          const r = await fallbackRes.json();
          setRecords(r.records || []);
          setPlaybook(r.playbook || null);
        }
      }
    } catch (err) {
      console.error("Failed to fetch Script Brain status:", err);
    } finally {
      setLoadingStatus(false);
    }
  }, []);

  useEffect(() => {
    loadBrainData();
  }, [loadBrainData]);

  const handleSyncFromSheet = async () => {
    if (syncingSheet) return;
    setSyncingSheet(true);
    setSyncMessage(null);
    try {
      const parsedLimit = syncLimit === "all" ? undefined : Number(syncLimit);
      const res = await fetch("/api/script-brain/ingest-sheet", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          sheetId: REPLICATION_SHEET_ID,
          tab: selectedSheetTab,
          limit: parsedLimit,
          reset: replaceExistingStore,
        }),
      });
      const data = await res.json();
      if (!res.ok || data.error) {
        throw new Error(data.error || "Failed to sync scripts from Replication Queue Google Sheet");
      }

      const ingestInfo = data.ingest_result || data.result?.ingest_result || data.result || data;
      const ingestedCount =
        ingestInfo.ingested_count ??
        data.items_submitted ??
        ingestInfo.total_records ??
        0;
      const tierCounts = ingestInfo.tier_counts || {};
      const detailParts: string[] = [];
      if (tierCounts.TIER_S !== undefined || tierCounts.TIER_A !== undefined || tierCounts.TIER_B !== undefined) {
        detailParts.push(
          `Tier S: ${tierCounts.TIER_S ?? 0} · Tier A: ${tierCounts.TIER_A ?? 0} · Tier B: ${tierCounts.TIER_B ?? 0}`
        );
      }
      if (ingestInfo.failed_count) {
        detailParts.push(`${ingestInfo.failed_count} skipped/failed`);
      }

      setSyncMessage({
        type: "success",
        text: `Synced ${ingestedCount} script${ingestedCount === 1 ? "" : "s"} from "${selectedSheetTab}" into Script Brain & retrained models.`,
        details: detailParts.join(" · ") || undefined,
      });

      await loadBrainData();
      setActiveTab("database");
    } catch (err: any) {
      setSyncMessage({
        type: "error",
        text: err.message || "Unexpected error while syncing from Google Sheet",
      });
    } finally {
      setSyncingSheet(false);
    }
  };

  const handleGenerate = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!brief.trim() || generating) return;
    setGenerating(true);
    setGenError(null);
    try {
      const res = await fetch("/api/script-brain/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          brief: brief.trim(),
          language,
          character,
          createGoogleDoc,
          docTitle: docTitle.trim() || undefined,
        }),
      });
      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.error || "Failed to generate Supernova script");
      }
      setResult(data.result);
    } catch (err: any) {
      setGenError(err.message || "Unexpected error during generation");
    } finally {
      setGenerating(false);
    }
  };

  const handleIngestSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!newAdCode.trim() || (!newDocUrl.trim() && !newRawScript.trim())) return;
    setIngesting(true);
    try {
      const cac7d = parseFloat(newSecondPayCac) || 0;
      const avgCac = parseFloat(newLangAvgCac) || 0;
      const cancelPct = parseFloat(newCancelPct) || 0;
      const res = await fetch("/api/script-brain/ingest", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          items: [
            {
              ad_code: newAdCode.trim().toUpperCase(),
              title: newTitle.trim() || newAdCode.trim().toUpperCase(),
              language: newLanguage,
              character_type: newCharacter,
              doc_url: newDocUrl.trim() || undefined,
              raw_script_text: newRawScript.trim() || undefined,
              cac_metrics: {
                second_pay_cac_7d: cac7d,
                lang_avg_cac: avgCac,
                cancel_pct: cancelPct,
              },
            },
          ],
        }),
      });
      const data = await res.json();
      if (!res.ok) {
        alert(data.error || "Failed to ingest script");
      } else {
        setShowIngestForm(false);
        setNewAdCode("");
        setNewTitle("");
        setNewDocUrl("");
        setNewRawScript("");
        await loadBrainData();
      }
    } catch (err: any) {
      alert(`Error: ${err.message}`);
    } finally {
      setIngesting(false);
    }
  };

  const handleRetrain = async () => {
    if (training) return;
    setTraining(true);
    try {
      const res = await fetch("/api/script-brain/train", { method: "POST" });
      if (res.ok) {
        await loadBrainData();
      }
    } finally {
      setTraining(false);
    }
  };

  const copyText = (text: string, key: string) => {
    navigator.clipboard.writeText(text);
    setCopiedKey(key);
    setTimeout(() => setCopiedKey(null), 2000);
  };

  // Filtered scripts for Trained Scripts in Brain table
  const filteredRecords = records.filter((r) => {
    if (tierFilter !== "ALL" && r.cac_tier !== tierFilter) return false;
    if (langFilter !== "ALL" && r.language?.toLowerCase() !== langFilter.toLowerCase()) return false;
    if (searchQuery.trim()) {
      const q = searchQuery.trim().toLowerCase();
      const hay = `${r.ad_code} ${r.title || ""} ${r.language || ""} ${r.character_type || ""} ${
        r.metadata_tags?.persona || ""
      } ${r.metadata_tags?.hook_angle || ""} ${r.english_script || ""}`.toLowerCase();
      if (!hay.includes(q)) return false;
    }
    return true;
  });

  const availableLanguages = Array.from(
    new Set(records.map((r) => r.language).filter(Boolean))
  );

  const renderTrainedScriptsSection = () => (
    <div className="flex flex-col gap-4">
      {/* Header & Filter Bar */}
      <div className="bg-white border border-slate-200 rounded-xl p-5 shadow-2xs flex flex-col gap-4">
        <div className="flex flex-wrap items-center justify-between gap-4">
          <div>
            <div className="flex items-center gap-2">
              <h2 className="text-sm font-bold text-slate-900">
                Trained Scripts in Brain ({records.length})
              </h2>
              <span className="text-[10px] font-mono px-2 py-0.5 rounded-full bg-indigo-50 text-indigo-700 border border-indigo-200">
                GET /api/script-brain/scripts
              </span>
              {storeUpdatedAt && (
                <span className="text-[11px] text-slate-400 font-mono">
                  Updated {new Date(storeUpdatedAt).toLocaleString()}
                </span>
              )}
            </div>
            <p className="text-xs text-slate-500 mt-0.5">
              Showing each ingested script&apos;s Ad Code, Title, Language &amp; Regional Doc Tabs, CAC Tier, Spends, 1st-Pay CAC, Post-1hr Cancel CAC, 2nd-Pay 4d/7d CAC, and expandable Visual Blueprint + 4 Story Blocks + English Translated Script.
            </p>
          </div>
          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={loadBrainData}
              disabled={loadingStatus}
              className="px-3 py-2 rounded-lg text-xs font-semibold bg-slate-100 hover:bg-slate-200 text-slate-700 flex items-center gap-1.5 cursor-pointer"
            >
              <RefreshCw className={`w-3.5 h-3.5 ${loadingStatus ? "animate-spin" : ""}`} />
              <span>Refresh List</span>
            </button>
            <button
              type="button"
              onClick={handleRetrain}
              disabled={training}
              className="px-3.5 py-2 rounded-lg text-xs font-semibold bg-slate-100 hover:bg-slate-200 text-slate-800 flex items-center gap-1.5 cursor-pointer"
            >
              <RefreshCw className={`w-3.5 h-3.5 ${training ? "animate-spin" : ""}`} />
              <span>{training ? "Training DSPy & XGBoost..." : "Re-Train Brain (0 API Calls)"}</span>
            </button>
            <button
              type="button"
              onClick={() => setShowIngestForm(!showIngestForm)}
              className="px-3.5 py-2 rounded-lg text-xs font-semibold bg-slate-900 hover:bg-black text-white flex items-center gap-1.5 cursor-pointer"
            >
              <Plus className="w-3.5 h-3.5" />
              <span>Add Manual Script</span>
            </button>
          </div>
        </div>

        {/* Search & Filters */}
        <div className="flex flex-wrap items-center gap-3 pt-3 border-t border-slate-100">
          <div className="relative flex-1 min-w-[220px]">
            <Search className="w-3.5 h-3.5 text-slate-400 absolute left-3 top-1/2 -translate-y-1/2" />
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Search by ad code (e.g. SN3195), title, persona, hook, or dialogue..."
              className="w-full rounded-lg border border-slate-200 bg-slate-50 pl-8 pr-3 py-1.5 text-xs text-slate-900 focus:bg-white focus:outline-none focus:border-purple-600"
            />
          </div>

          <div className="flex items-center gap-1.5">
            <Filter className="w-3.5 h-3.5 text-slate-400" />
            <span className="text-[11px] font-mono uppercase text-slate-500">CAC Tier:</span>
            {(["ALL", "TIER_S", "TIER_A", "TIER_B"] as const).map((tier) => (
              <button
                key={tier}
                type="button"
                onClick={() => setTierFilter(tier)}
                className={`px-2.5 py-1 rounded-md text-[11px] font-mono font-semibold cursor-pointer border transition-colors ${
                  tierFilter === tier
                    ? "bg-slate-900 text-white border-slate-900"
                    : "bg-slate-50 text-slate-600 border-slate-200 hover:bg-slate-100"
                }`}
              >
                {tier === "ALL" ? "All Tiers" : tier}
              </button>
            ))}
          </div>

          <div className="flex items-center gap-1.5">
            <span className="text-[11px] font-mono uppercase text-slate-500">Language:</span>
            <select
              value={langFilter}
              onChange={(e) => setLangFilter(e.target.value)}
              className="rounded-lg border border-slate-200 bg-slate-50 px-2.5 py-1 text-xs text-slate-800 focus:outline-none focus:border-purple-600"
            >
              <option value="ALL">All Languages</option>
              {availableLanguages.map((lang) => (
                <option key={lang} value={lang}>
                  {lang}
                </option>
              ))}
            </select>
          </div>
        </div>
      </div>

      {/* Optional Manual Single-Script Ingest Form */}
      {showIngestForm && (
        <form
          onSubmit={handleIngestSubmit}
          className="bg-white border border-purple-200 rounded-xl p-5 shadow-sm flex flex-col gap-4"
        >
          <h3 className="text-xs font-bold uppercase font-mono text-purple-800">
            Ingest Single Proven Script + CAC Data (Zero Gemini API Calls)
          </h3>
          <div className="grid grid-cols-1 sm:grid-cols-4 gap-3">
            <div>
              <label className="block text-xs font-semibold text-slate-700 mb-1">Ad Code *</label>
              <input
                type="text"
                value={newAdCode}
                onChange={(e) => setNewAdCode(e.target.value)}
                placeholder="e.g. SN3290"
                className="w-full rounded-lg border border-slate-200 px-3 py-1.5 text-xs"
                required
              />
            </div>
            <div>
              <label className="block text-xs font-semibold text-slate-700 mb-1">Title</label>
              <input
                type="text"
                value={newTitle}
                onChange={(e) => setNewTitle(e.target.value)}
                placeholder="e.g. Tamil Mother PTM Winner"
                className="w-full rounded-lg border border-slate-200 px-3 py-1.5 text-xs"
              />
            </div>
            <div>
              <label className="block text-xs font-semibold text-slate-700 mb-1">Language</label>
              <select
                value={newLanguage}
                onChange={(e) => setNewLanguage(e.target.value)}
                className="w-full rounded-lg border border-slate-200 px-3 py-1.5 text-xs"
              >
                <option value="Tamil">Tamil</option>
                <option value="Telugu">Telugu</option>
                <option value="Malayalam">Malayalam</option>
                <option value="Kannada">Kannada</option>
                <option value="Hindi">Hindi</option>
              </select>
            </div>
            <div>
              <label className="block text-xs font-semibold text-slate-700 mb-1">Character</label>
              <select
                value={newCharacter}
                onChange={(e) => setNewCharacter(e.target.value)}
                className="w-full rounded-lg border border-slate-200 px-3 py-1.5 text-xs"
              >
                <option value="Female Humanoid Robot">Female Humanoid Robot</option>
                <option value="Female Robot">Female Robot</option>
                <option value="Miss Nova">Miss Nova</option>
              </select>
            </div>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
            <div>
              <label className="block text-xs font-semibold text-slate-700 mb-1">
                Ad 2nd-Pay CAC 7d (₹)
              </label>
              <input
                type="number"
                value={newSecondPayCac}
                onChange={(e) => setNewSecondPayCac(e.target.value)}
                className="w-full rounded-lg border border-slate-200 px-3 py-1.5 text-xs"
              />
            </div>
            <div>
              <label className="block text-xs font-semibold text-slate-700 mb-1">
                Language Avg CAC (₹)
              </label>
              <input
                type="number"
                value={newLangAvgCac}
                onChange={(e) => setNewLangAvgCac(e.target.value)}
                className="w-full rounded-lg border border-slate-200 px-3 py-1.5 text-xs"
              />
            </div>
            <div>
              <label className="block text-xs font-semibold text-slate-700 mb-1">
                Cancellation %
              </label>
              <input
                type="number"
                step="0.1"
                value={newCancelPct}
                onChange={(e) => setNewCancelPct(e.target.value)}
                className="w-full rounded-lg border border-slate-200 px-3 py-1.5 text-xs"
              />
            </div>
          </div>

          <div>
            <label className="block text-xs font-semibold text-slate-700 mb-1">
              Google Doc URL (or paste script text below)
            </label>
            <input
              type="url"
              value={newDocUrl}
              onChange={(e) => setNewDocUrl(e.target.value)}
              placeholder="https://docs.google.com/document/d/..."
              className="w-full rounded-lg border border-slate-200 px-3 py-1.5 text-xs mb-2"
            />
            <textarea
              rows={4}
              value={newRawScript}
              onChange={(e) => setNewRawScript(e.target.value)}
              placeholder="Or paste the Google Doc text directly (in English, Tamil, Telugu, Malayalam, Kannada, or Hindi — auto-translated locally with 0 Gemini calls)..."
              className="w-full rounded-lg border border-slate-200 px-3 py-2 text-xs"
            />
          </div>

          <div className="flex justify-end gap-2">
            <button
              type="button"
              onClick={() => setShowIngestForm(false)}
              className="px-3 py-1.5 rounded-lg text-xs text-slate-600 hover:bg-slate-100"
            >
              Cancel
            </button>
            <button
              type="submit"
              disabled={ingesting}
              className="px-4 py-1.5 rounded-lg text-xs font-semibold bg-purple-600 hover:bg-purple-700 text-white cursor-pointer"
            >
              {ingesting ? "Ingesting & Embedding..." : "Ingest & Train Brain"}
            </button>
          </div>
        </form>
      )}

      {/* Trained Scripts Table with Expandable Details */}
      <div className="bg-white border border-slate-200 rounded-xl overflow-hidden shadow-2xs">
        <div className="overflow-x-auto">
          <table className="w-full text-left border-collapse">
            <thead>
              <tr className="bg-slate-50 border-b border-slate-200 text-[11px] font-mono uppercase text-slate-500">
                <th className="py-3 px-4">Ad Code &amp; Title</th>
                <th className="py-3 px-3">Language &amp; Doc Tabs</th>
                <th className="py-3 px-3">CAC Tier</th>
                <th className="py-3 px-3">Spends</th>
                <th className="py-3 px-3">1st-Pay CAC</th>
                <th className="py-3 px-3">Post-1hr Cancel CAC</th>
                <th className="py-3 px-3">2nd-Pay 4d CAC</th>
                <th className="py-3 px-3">2nd-Pay 7d CAC</th>
                <th className="py-3 px-4 text-right">Blueprint &amp; Script</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100 text-xs">
              {filteredRecords.length === 0 ? (
                <tr>
                  <td colSpan={9} className="py-8 px-4 text-center text-slate-500">
                    No trained scripts match the current filter. Use &ldquo;Sync From Replication Sheet&rdquo; above to ingest proven scripts.
                  </td>
                </tr>
              ) : (
                filteredRecords.map((r) => {
                  const isExpanded = Boolean(expandedAdCodes[r.ad_code]);
                  const docTabs = getRegionalDocTabs(r);
                  const spendsFormatted = getScriptSpends(r);
                  const firstPayCac = formatInrMetric(r.cac_metrics?.first_pay_cac);
                  const post1hrCancelCac = formatInrMetric(r.cac_metrics?.post_1hr_cancel_cac);
                  const secondPay4dCac = formatInrMetric(r.cac_metrics?.second_pay_cac_4d);
                  const secondPay7dCac = formatInrMetric(r.cac_metrics?.second_pay_cac_7d);
                  const cacVsAvg = r.cac_metrics?.cac_vs_avg_pct;
                  const cancelPct = r.cac_metrics?.cancel_pct;

                  return (
                    <React.Fragment key={r.ad_code}>
                      <tr
                        className={`hover:bg-slate-50/90 transition-colors ${
                          isExpanded ? "bg-purple-50/30" : ""
                        }`}
                      >
                        {/* Ad Code & Title */}
                        <td className="py-3.5 px-4 max-w-[260px]">
                          <div className="flex items-center gap-2">
                            <span className="font-mono font-bold text-slate-900">{r.ad_code}</span>
                            {r.doc_url && (
                              <a
                                href={r.doc_url}
                                target="_blank"
                                rel="noopener noreferrer"
                                title="Open Source Google Doc"
                                className="text-purple-600 hover:text-purple-800 inline-flex items-center gap-0.5 text-[11px] font-medium"
                              >
                                <ExternalLink className="w-3 h-3" />
                              </a>
                            )}
                          </div>
                          <div className="text-xs text-slate-700 font-medium truncate mt-0.5" title={r.title}>
                            {r.title || r.ad_code}
                          </div>
                          <div className="text-[11px] text-slate-400 mt-0.5 flex flex-wrap items-center gap-1.5">
                            <span>{r.character_type}</span>
                            {r.metadata_tags?.persona && (
                              <>
                                <span>·</span>
                                <span>{r.metadata_tags.persona}</span>
                              </>
                            )}
                          </div>
                        </td>

                        {/* Language & Available Regional Doc Tabs */}
                        <td className="py-3.5 px-3">
                          <div className="font-semibold text-slate-900">{r.language}</div>
                          {docTabs.length > 0 ? (
                            <div className="flex flex-wrap gap-1 mt-1">
                              {docTabs.map((tabName, tIdx) => (
                                <span
                                  key={`${r.ad_code}-tab-${tIdx}`}
                                  className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-slate-100 text-slate-600 border border-slate-200"
                                  title="Regional Google Doc Tab"
                                >
                                  {tabName}
                                </span>
                              ))}
                            </div>
                          ) : (
                            <div className="text-[10px] font-mono text-slate-400 mt-0.5">
                              Tab: {r.language} / Supernova Script
                            </div>
                          )}
                        </td>

                        {/* CAC Tier */}
                        <td className="py-3.5 px-3">
                          <span
                            className={`text-[10px] font-mono font-semibold px-2 py-0.5 rounded border inline-block ${
                              r.cac_tier === "TIER_S"
                                ? "bg-emerald-50 text-emerald-700 border-emerald-200"
                                : r.cac_tier === "TIER_A"
                                ? "bg-blue-50 text-blue-700 border-blue-200"
                                : "bg-amber-50 text-amber-700 border-amber-200"
                            }`}
                          >
                            {r.cac_tier}
                          </span>
                          {cacVsAvg !== undefined && cacVsAvg !== null && (
                            <div
                              className={`text-[10px] font-mono mt-1 ${
                                cacVsAvg < 0 ? "text-emerald-700 font-semibold" : "text-amber-700"
                              }`}
                            >
                              {cacVsAvg > 0 ? `+${cacVsAvg}%` : `${cacVsAvg}%`} vs avg
                            </div>
                          )}
                        </td>

                        {/* Spends */}
                        <td className="py-3.5 px-3 font-mono text-slate-800">{spendsFormatted}</td>

                        {/* 1st-Pay CAC */}
                        <td className="py-3.5 px-3 font-mono text-slate-800">{firstPayCac}</td>

                        {/* Post-1hr Cancel CAC */}
                        <td className="py-3.5 px-3 font-mono text-slate-800">
                          <div>{post1hrCancelCac}</div>
                          {cancelPct !== undefined && cancelPct !== null && (
                            <div className="text-[10px] text-slate-400">{cancelPct}% cancel</div>
                          )}
                        </td>

                        {/* 2nd-Pay 4d CAC */}
                        <td className="py-3.5 px-3 font-mono text-slate-800">{secondPay4dCac}</td>

                        {/* 2nd-Pay 7d CAC */}
                        <td className="py-3.5 px-3 font-mono font-semibold text-slate-900">
                          <div>{secondPay7dCac}</div>
                          {r.cac_metrics?.lang_avg_cac ? (
                            <div className="text-[10px] font-normal text-slate-400">
                              Avg: ₹{r.cac_metrics.lang_avg_cac}
                            </div>
                          ) : null}
                        </td>

                        {/* Expand / Collapse Action */}
                        <td className="py-3.5 px-4 text-right">
                          <button
                            type="button"
                            onClick={() => toggleExpandScript(r.ad_code)}
                            className="inline-flex items-center gap-1 px-2.5 py-1.5 rounded-lg text-xs font-semibold bg-purple-50 hover:bg-purple-100 text-purple-700 border border-purple-200 transition-colors cursor-pointer"
                          >
                            <span>{isExpanded ? "Hide Details" : "Inspect Script"}</span>
                            {isExpanded ? (
                              <ChevronUp className="w-3.5 h-3.5" />
                            ) : (
                              <ChevronDown className="w-3.5 h-3.5" />
                            )}
                          </button>
                        </td>
                      </tr>

                      {/* Expanded Details Row: Visual Blueprint + 4 Story Blocks + English Translated Script */}
                      {isExpanded && (
                        <tr className="bg-slate-50/90 border-b border-slate-200">
                          <td colSpan={9} className="p-5">
                            <div className="grid grid-cols-1 lg:grid-cols-12 gap-4">
                              {/* Column 1: Visual Blueprint (4 cols) */}
                              <div className="lg:col-span-4 bg-white border border-indigo-200/80 rounded-xl p-4 shadow-2xs flex flex-col gap-2.5">
                                <div className="flex items-center justify-between border-b border-slate-100 pb-2">
                                  <h4 className="text-xs font-mono uppercase font-bold text-indigo-700">
                                    Visual Blueprint (SQL Column)
                                  </h4>
                                  <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-indigo-50 text-indigo-700">
                                    Hook: {r.metadata_tags?.hook_word_count ?? "—"}w · Turn #
                                    {r.metadata_tags?.ai_teacher_entry_turn ?? 2}
                                  </span>
                                </div>
                                <div className="text-xs text-slate-700 flex flex-col gap-2 leading-relaxed">
                                  <div>
                                    <span className="font-bold text-slate-900">Split Screen: </span>
                                    {r.visual_blueprint?.split_screen_type || "Standard 2-panel split screen"}
                                  </div>
                                  {r.visual_blueprint?.hook_visuals && (
                                    <div>
                                      <span className="font-bold text-slate-900">Hook Visuals: </span>
                                      {r.visual_blueprint.hook_visuals}
                                    </div>
                                  )}
                                  {r.visual_blueprint?.body_visuals && (
                                    <div>
                                      <span className="font-bold text-slate-900">Body Visuals: </span>
                                      {r.visual_blueprint.body_visuals}
                                    </div>
                                  )}
                                  {r.visual_blueprint?.video_flow && (
                                    <div>
                                      <span className="font-bold text-slate-900">Video Flow: </span>
                                      {r.visual_blueprint.video_flow}
                                    </div>
                                  )}
                                  {r.visual_blueprint?.characters_description && (
                                    <div className="whitespace-pre-wrap bg-indigo-50/40 p-2.5 rounded-lg border border-indigo-100 text-[11px]">
                                      <span className="font-bold text-indigo-950 block mb-0.5">
                                        Characters:
                                      </span>
                                      {r.visual_blueprint.characters_description}
                                    </div>
                                  )}
                                </div>
                              </div>

                              {/* Column 2: 4 Story Blocks (4 cols) */}
                              <div className="lg:col-span-4 bg-white border border-purple-200/80 rounded-xl p-4 shadow-2xs flex flex-col gap-2">
                                <div className="flex items-center justify-between border-b border-slate-100 pb-2">
                                  <h4 className="text-xs font-mono uppercase font-bold text-purple-700">
                                    4 Story Blocks (Vector Embedded)
                                  </h4>
                                  <span className="text-[10px] font-mono text-slate-500">
                                    {r.metadata_tags?.total_turns ?? "—"} turns ·{" "}
                                    {r.metadata_tags?.avg_words_per_line ?? "—"}w/line
                                  </span>
                                </div>
                                <div className="flex flex-col gap-2 text-xs leading-relaxed">
                                  <div className="p-2.5 rounded-lg bg-purple-50/40 border border-purple-100">
                                    <span className="font-bold text-purple-950 block text-[11px] uppercase font-mono mb-0.5">
                                      Block 1 — Hook &amp; AI Teacher Interrupt
                                    </span>
                                    <div className="whitespace-pre-wrap text-slate-800">
                                      {r.story_blocks?.block_1_hook || "—"}
                                    </div>
                                  </div>
                                  <div className="p-2.5 rounded-lg bg-slate-50 border border-slate-200">
                                    <span className="font-bold text-slate-900 block text-[11px] uppercase font-mono mb-0.5">
                                      Block 2 — Pain &amp; Failed Old Methods
                                    </span>
                                    <div className="whitespace-pre-wrap text-slate-700">
                                      {r.story_blocks?.block_2_pain_and_old_methods || "—"}
                                    </div>
                                  </div>
                                  <div className="p-2.5 rounded-lg bg-slate-50 border border-slate-200">
                                    <span className="font-bold text-slate-900 block text-[11px] uppercase font-mono mb-0.5">
                                      Block 3 — Live Sentence Fix &amp; 4 USPs
                                    </span>
                                    <div className="whitespace-pre-wrap text-slate-700">
                                      {r.story_blocks?.block_3_live_fix_and_usps || "—"}
                                    </div>
                                  </div>
                                  <div className="p-2.5 rounded-lg bg-emerald-50/40 border border-emerald-200">
                                    <span className="font-bold text-emerald-950 block text-[11px] uppercase font-mono mb-0.5">
                                      Block 4 — Desire &amp; 30-Day Install CTA
                                    </span>
                                    <div className="whitespace-pre-wrap text-slate-800">
                                      {r.story_blocks?.block_4_desire_and_cta || "—"}
                                    </div>
                                  </div>
                                </div>
                              </div>

                              {/* Column 3: English Translated Script (4 cols) */}
                              <div className="lg:col-span-4 bg-white border border-slate-200 rounded-xl p-4 shadow-2xs flex flex-col gap-2">
                                <div className="flex items-center justify-between border-b border-slate-100 pb-2">
                                  <h4 className="text-xs font-mono uppercase font-bold text-slate-800 flex items-center gap-1.5">
                                    <FileText className="w-3.5 h-3.5 text-emerald-600" />
                                    <span>English Translated Script</span>
                                  </h4>
                                  <button
                                    type="button"
                                    onClick={() =>
                                      copyText(r.english_script || "", `script_${r.ad_code}`)
                                    }
                                    className="text-[11px] text-slate-500 hover:text-slate-900 flex items-center gap-1 cursor-pointer"
                                  >
                                    {copiedKey === `script_${r.ad_code}` ? (
                                      <>
                                        <Check className="w-3 h-3 text-emerald-600" />
                                        <span>Copied</span>
                                      </>
                                    ) : (
                                      <>
                                        <Copy className="w-3 h-3" />
                                        <span>Copy Script</span>
                                      </>
                                    )}
                                  </button>
                                </div>
                                <pre className="text-xs text-slate-800 whitespace-pre-wrap font-sans leading-relaxed bg-slate-50 p-3 rounded-lg border border-slate-200 max-h-[420px] overflow-y-auto">
                                  {r.english_script || "No English translated script available."}
                                </pre>
                              </div>
                            </div>
                          </td>
                        </tr>
                      )}
                    </React.Fragment>
                  );
                })
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );

  return (
    <div className="min-h-screen w-full bg-[#F8FAFC] font-sans text-slate-900 flex flex-col">
      {/* Top Navbar */}
      <header className="w-full bg-white border-b border-slate-200 px-6 py-3.5 sticky top-0 z-30 flex items-center justify-between shadow-2xs">
        <div className="flex items-center gap-4">
          <Link
            to="/"
            className="flex items-center gap-1.5 text-xs font-semibold text-slate-500 hover:text-slate-900 transition-colors py-1.5 px-3 rounded-lg hover:bg-slate-100"
          >
            <ArrowLeft className="w-4 h-4" />
            <span>Suite Home</span>
          </Link>
          <div className="h-4 w-px bg-slate-200" />
          <div className="flex items-center gap-2.5">
            <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-purple-600 to-indigo-700 flex items-center justify-center text-white shadow-xs">
              <Brain className="w-4 h-4" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h1 className="text-sm font-bold text-slate-900">Supernova Script Brain</h1>
                <span className="text-[10px] font-mono font-semibold px-2 py-0.5 rounded-full bg-purple-50 text-purple-700 border border-purple-200">
                  Python DSPy + ChromaDB + XGBoost
                </span>
              </div>
              <p className="text-[11px] text-slate-500">
                Middle Brain trained on proven CAC-winning scripts (0 Gemini calls for vector &amp; CAC training)
              </p>
            </div>
          </div>
        </div>

        {/* Navigation Tabs */}
        <div className="flex items-center gap-2">
          <div className="flex items-center bg-slate-100 p-1 rounded-lg border border-slate-200">
            <button
              type="button"
              onClick={() => setActiveTab("generate")}
              className={`px-3 py-1.5 rounded-md text-xs font-semibold transition-all cursor-pointer flex items-center gap-1.5 ${
                activeTab === "generate"
                  ? "bg-white text-slate-900 shadow-2xs"
                  : "text-slate-600 hover:text-slate-900"
              }`}
            >
              <Sparkles className="w-3.5 h-3.5 text-purple-600" />
              <span>Brief → Script Generator</span>
            </button>
            <button
              type="button"
              onClick={() => setActiveTab("database")}
              className={`px-3 py-1.5 rounded-md text-xs font-semibold transition-all cursor-pointer flex items-center gap-1.5 ${
                activeTab === "database"
                  ? "bg-white text-slate-900 shadow-2xs"
                  : "text-slate-600 hover:text-slate-900"
              }`}
            >
              <Database className="w-3.5 h-3.5 text-indigo-600" />
              <span>Trained Scripts in Brain ({records.length})</span>
            </button>
            <button
              type="button"
              onClick={() => setActiveTab("architecture")}
              className={`px-3 py-1.5 rounded-md text-xs font-semibold transition-all cursor-pointer flex items-center gap-1.5 ${
                activeTab === "architecture"
                  ? "bg-white text-slate-900 shadow-2xs"
                  : "text-slate-600 hover:text-slate-900"
              }`}
            >
              <Sliders className="w-3.5 h-3.5 text-emerald-600" />
              <span>Python Brain &amp; Playbook</span>
            </button>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="w-full max-w-7xl mx-auto px-6 py-6 flex-1 flex flex-col gap-6">
        {/* Top Metrics Strip */}
        <div className="grid grid-cols-1 sm:grid-cols-4 gap-4">
          <div className="bg-white border border-slate-200 rounded-xl p-4 shadow-2xs">
            <div className="text-[11px] font-mono uppercase text-slate-500 mb-1">Proven Scripts Indexed</div>
            <div className="text-2xl font-bold text-slate-900">
              {statusData?.db_stats?.total_count ?? statusData?.db_stats?.total_scripts ?? records.length}
            </div>
            <div className="text-[11px] text-slate-500 mt-1">
              Tier S: {statusData?.db_stats?.by_cac_tier?.TIER_S ?? records.filter((r) => r.cac_tier === "TIER_S").length} · Tier A:{" "}
              {statusData?.db_stats?.by_cac_tier?.TIER_A ?? records.filter((r) => r.cac_tier === "TIER_A").length} · Tier B:{" "}
              {statusData?.db_stats?.by_cac_tier?.TIER_B ?? records.filter((r) => r.cac_tier === "TIER_B").length}
            </div>
          </div>

          <div className="bg-white border border-slate-200 rounded-xl p-4 shadow-2xs">
            <div className="text-[11px] font-mono uppercase text-slate-500 mb-1">Vector &amp; Embedding Engine</div>
            <div className="text-sm font-bold text-slate-900 flex items-center gap-1.5 mt-1">
              <CheckCircle2 className="w-4 h-4 text-emerald-600" />
              <span>ChromaDB {statusData?.packages?.chromadb?.version || "1.5.9"}</span>
            </div>
            <div className="text-[11px] text-slate-500 mt-1.5">
              Local CPU embeddings (0 Gemini API calls)
            </div>
          </div>

          <div className="bg-white border border-slate-200 rounded-xl p-4 shadow-2xs">
            <div className="text-[11px] font-mono uppercase text-slate-500 mb-1">Thinking Compiler</div>
            <div className="text-sm font-bold text-slate-900 flex items-center gap-1.5 mt-1">
              <Zap className="w-4 h-4 text-purple-600" />
              <span>Stanford DSPy {statusData?.packages?.dspy?.version || "3.1.3"}</span>
            </div>
            <div className="text-[11px] text-slate-500 mt-1.5">
              TextGrad {statusData?.packages?.textgrad?.version || "0.1.8"} contrastive rules active
            </div>
          </div>

          <div className="bg-white border border-slate-200 rounded-xl p-4 shadow-2xs">
            <div className="text-[11px] font-mono uppercase text-slate-500 mb-1">CAC Success Predictor</div>
            <div className="text-sm font-bold text-slate-900 flex items-center gap-1.5 mt-1">
              <TrendingUp className="w-4 h-4 text-indigo-600" />
              <span>XGBoost {statusData?.packages?.xgboost?.version || "3.2.0"} + Sklearn</span>
            </div>
            <div className="text-[11px] text-slate-500 mt-1.5">
              12-feature historical CAC probability scorer
            </div>
          </div>
        </div>

        {/* Connected Replication Queue Google Sheet Sync Card */}
        <section className="bg-white border border-emerald-200/90 rounded-xl p-5 shadow-2xs flex flex-col gap-3.5">
          <div className="flex flex-wrap items-center justify-between gap-4">
            <div className="flex items-start gap-3">
              <div className="w-9 h-9 rounded-lg bg-emerald-50 border border-emerald-200 flex items-center justify-center text-emerald-700 shrink-0 mt-0.5">
                <FileSpreadsheet className="w-5 h-5" />
              </div>
              <div>
                <div className="flex flex-wrap items-center gap-2">
                  <span className="text-xs font-mono uppercase font-bold text-emerald-800">
                    Connected Replication Queue Google Sheet
                  </span>
                  <span className="text-xs font-bold text-slate-900">
                    {REPLICATION_SHEET_NAME}
                  </span>
                  <code className="text-[10px] font-mono px-2 py-0.5 rounded bg-slate-100 text-slate-600 border border-slate-200">
                    ID: {REPLICATION_SHEET_ID}
                  </code>
                  <a
                    href={REPLICATION_SHEET_URL}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-[11px] font-semibold text-emerald-700 hover:text-emerald-900 inline-flex items-center gap-1"
                  >
                    <span>Open Sheet</span>
                    <ExternalLink className="w-3 h-3" />
                  </a>
                </div>
                <p className="text-xs text-slate-500 mt-0.5">
                  Reads proven scripts, Google Doc Smart Chips (including regional tabs), Spends, and 1st-Pay / Post-1hr Cancel / 2nd-Pay 4d &amp; 7d CAC directly into the Brain with 0 Gemini calls.
                </p>
              </div>
            </div>

            {/* Sync Controls */}
            <div className="flex flex-wrap items-center gap-3">
              <div>
                <label htmlFor="sheet-tab-select" className="block text-[10px] font-mono uppercase text-slate-500 mb-1">
                  Replication Sheet Tab
                </label>
                <select
                  id="sheet-tab-select"
                  value={selectedSheetTab}
                  onChange={(e) => setSelectedSheetTab(e.target.value)}
                  disabled={syncingSheet}
                  className="rounded-lg border border-slate-200 bg-slate-50 px-3 py-1.5 text-xs font-medium text-slate-900 focus:bg-white focus:outline-none focus:border-emerald-600"
                >
                  {REPLICATION_SHEET_TABS.map((tabName) => (
                    <option key={tabName} value={tabName}>
                      {tabName}
                    </option>
                  ))}
                </select>
              </div>

              <div>
                <label htmlFor="sheet-limit-select" className="block text-[10px] font-mono uppercase text-slate-500 mb-1">
                  Ingest Limit
                </label>
                <select
                  id="sheet-limit-select"
                  value={syncLimit}
                  onChange={(e) => setSyncLimit(e.target.value)}
                  disabled={syncingSheet}
                  className="rounded-lg border border-slate-200 bg-slate-50 px-3 py-1.5 text-xs font-medium text-slate-900 focus:bg-white focus:outline-none focus:border-emerald-600"
                >
                  {SYNC_LIMIT_OPTIONS.map((opt) => (
                    <option key={opt.value} value={opt.value}>
                      {opt.label}
                    </option>
                  ))}
                </select>
              </div>

              <label className="flex items-center gap-2 text-xs text-slate-700 cursor-pointer pt-4 select-none">
                <input
                  type="checkbox"
                  checked={replaceExistingStore}
                  onChange={(e) => setReplaceExistingStore(e.target.checked)}
                  disabled={syncingSheet}
                  className="rounded border-slate-300 text-emerald-600 focus:ring-emerald-500"
                />
                <span className="font-medium">Replace existing store</span>
              </label>

              <div className="pt-3.5">
                <button
                  type="button"
                  onClick={handleSyncFromSheet}
                  disabled={syncingSheet}
                  className="px-4 py-2 rounded-lg text-xs font-semibold bg-emerald-600 hover:bg-emerald-700 disabled:opacity-50 text-white flex items-center gap-1.5 cursor-pointer shadow-2xs transition-colors"
                >
                  {syncingSheet ? (
                    <>
                      <Loader2 className="w-3.5 h-3.5 animate-spin" />
                      <span>Syncing &amp; Training...</span>
                    </>
                  ) : (
                    <>
                      <RefreshCw className="w-3.5 h-3.5" />
                      <span>Sync From Replication Sheet</span>
                    </>
                  )}
                </button>
              </div>
            </div>
          </div>

          {syncMessage && (
            <div
              className={`p-3 rounded-lg border text-xs flex items-center justify-between gap-3 ${
                syncMessage.type === "success"
                  ? "bg-emerald-50/80 border-emerald-200 text-emerald-900"
                  : "bg-red-50 border-red-200 text-red-700"
              }`}
            >
              <div className="flex items-center gap-2">
                {syncMessage.type === "success" ? (
                  <CheckCircle2 className="w-4 h-4 text-emerald-600 shrink-0" />
                ) : (
                  <AlertCircle className="w-4 h-4 text-red-600 shrink-0" />
                )}
                <span className="font-semibold">{syncMessage.text}</span>
                {syncMessage.details && (
                  <span className="font-mono text-[11px] opacity-80">({syncMessage.details})</span>
                )}
              </div>
              <button
                type="button"
                onClick={() => setSyncMessage(null)}
                className="text-[11px] underline opacity-75 hover:opacity-100 cursor-pointer"
              >
                Dismiss
              </button>
            </div>
          )}
        </section>

        {/* TAB 1: GENERATOR */}
        {activeTab === "generate" && (
          <div className="flex flex-col gap-8">
            <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
              {/* Left Input Column */}
              <form
                onSubmit={handleGenerate}
                className="lg:col-span-5 bg-white border border-slate-200 rounded-xl p-6 shadow-2xs flex flex-col gap-4"
              >
                <div>
                  <h2 className="text-sm font-bold text-slate-900">1. Enter Concept Brief Idea</h2>
                  <p className="text-xs text-slate-500 mt-0.5">
                    The Middle Brain matches your brief against proven CAC winners, builds the 4 Story Blocks + Visual Description, and predicts the Success Rate %.
                  </p>
                </div>

                {/* Quick Sample Presets */}
                <div>
                  <label className="block text-[11px] font-mono uppercase text-slate-500 mb-1.5">
                    Try a Sample Concept Brief
                  </label>
                  <div className="flex flex-wrap gap-1.5">
                    {SAMPLE_BRIEFS.map((sample) => (
                      <button
                        key={sample.label}
                        type="button"
                        onClick={() => {
                          setBrief(sample.brief);
                          setLanguage(sample.language);
                          setCharacter(sample.character);
                        }}
                        className="text-[11px] px-2.5 py-1 rounded-md bg-slate-100 hover:bg-purple-50 text-slate-700 hover:text-purple-700 border border-slate-200 hover:border-purple-200 transition-colors cursor-pointer"
                      >
                        {sample.label}
                      </button>
                    ))}
                  </div>
                </div>

                <div>
                  <label htmlFor="brief-textarea" className="block text-xs font-semibold text-slate-700 mb-1.5">
                    Concept Brief (1–4 sentences in English)
                  </label>
                  <textarea
                    id="brief-textarea"
                    rows={5}
                    value={brief}
                    onChange={(e) => setBrief(e.target.value)}
                    placeholder="Describe who the character is, where they are, and what English hesitation or mistake happens..."
                    className="w-full rounded-lg border border-slate-200 bg-slate-50 px-3.5 py-2.5 text-xs text-slate-900 focus:bg-white focus:outline-none focus:border-purple-600 leading-relaxed"
                    required
                  />
                </div>

                <div className="grid grid-cols-2 gap-3">
                  <div>
                    <label htmlFor="lang-select" className="block text-xs font-semibold text-slate-700 mb-1">
                      Target Region / Language
                    </label>
                    <select
                      id="lang-select"
                      value={language}
                      onChange={(e) => setLanguage(e.target.value)}
                      className="w-full rounded-lg border border-slate-200 bg-slate-50 px-3 py-2 text-xs text-slate-900 focus:outline-none focus:border-purple-600"
                    >
                      <option value="Tamil">Tamil (South)</option>
                      <option value="Telugu">Telugu (South)</option>
                      <option value="Malayalam">Malayalam (South)</option>
                      <option value="Kannada">Kannada (South)</option>
                      <option value="Hindi">Hindi (North)</option>
                    </select>
                  </div>

                  <div>
                    <label htmlFor="char-select" className="block text-xs font-semibold text-slate-700 mb-1">
                      AI Teacher Character
                    </label>
                    <select
                      id="char-select"
                      value={character}
                      onChange={(e) => setCharacter(e.target.value)}
                      className="w-full rounded-lg border border-slate-200 bg-slate-50 px-3 py-2 text-xs text-slate-900 focus:outline-none focus:border-purple-600"
                    >
                      <option value="Female Humanoid Robot">Female Humanoid Robot</option>
                      <option value="Female Robot">Female Robot</option>
                      <option value="Miss Nova">Miss Nova</option>
                    </select>
                  </div>
                </div>

                {/* Optional Google Doc Creation */}
                <div className="pt-2 border-t border-slate-100 flex flex-col gap-2">
                  <label className="flex items-center gap-2 text-xs text-slate-700 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={createGoogleDoc}
                      onChange={(e) => setCreateGoogleDoc(e.target.checked)}
                      className="rounded border-slate-300 text-purple-600 focus:ring-purple-500"
                    />
                    <span className="font-medium">Also create a Google Doc in Shared Drive</span>
                  </label>
                  {createGoogleDoc && (
                    <input
                      type="text"
                      value={docTitle}
                      onChange={(e) => setDocTitle(e.target.value)}
                      placeholder="Optional Google Doc title (e.g. SN_TAMIL_PTM_01)"
                      className="w-full rounded-lg border border-slate-200 bg-slate-50 px-3 py-1.5 text-xs text-slate-900"
                    />
                  )}
                </div>

                {genError && (
                  <div className="p-3 rounded-lg bg-red-50 border border-red-200 text-xs text-red-700 flex items-start gap-2">
                    <AlertCircle className="w-4 h-4 shrink-0 mt-0.5" />
                    <span>{genError}</span>
                  </div>
                )}

                <button
                  type="submit"
                  disabled={generating || !brief.trim()}
                  className="w-full py-2.5 px-4 rounded-lg text-xs font-semibold text-white bg-gradient-to-r from-[#090D16] via-[#1E1B4B] to-[#4C1D95] hover:from-black hover:to-[#3B0764] disabled:opacity-50 flex items-center justify-center gap-2 cursor-pointer shadow-xs transition-all"
                >
                  {generating ? (
                    <>
                      <Loader2 className="w-4 h-4 animate-spin" />
                      <span>Running Python Middle Brain (ChromaDB → DSPy → XGBoost)...</span>
                    </>
                  ) : (
                    <>
                      <Sparkles className="w-4 h-4" />
                      <span>Generate Strongest Supernova Script + Success Score</span>
                    </>
                  )}
                </button>
              </form>

              {/* Right Output Column */}
              <div className="lg:col-span-7 flex flex-col gap-5">
                {!result && !generating && (
                  <div className="bg-white border border-dashed border-slate-300 rounded-xl p-12 text-center">
                    <Brain className="w-10 h-10 text-slate-400 mx-auto mb-3" />
                    <h3 className="text-sm font-bold text-slate-800 mb-1">
                      Ready to Think &amp; Write Your Supernova Script
                    </h3>
                    <p className="text-xs text-slate-500 max-w-md mx-auto">
                      Enter a short Concept Brief on the left and click Generate. The Python Middle Brain will pull the top 3 CAC-winning scripts, build a matching Visual Description + 4-Block Blueprint, write 3 Hook Options + the Full Script, and calculate the Approximate Success Rate %.
                    </p>
                  </div>
                )}

                {generating && (
                  <div className="bg-white border border-slate-200 rounded-xl p-10 text-center shadow-2xs">
                    <Loader2 className="w-8 h-8 text-purple-600 animate-spin mx-auto mb-3" />
                    <h3 className="text-sm font-bold text-slate-900">
                      Python Middle Brain Is Thinking...
                    </h3>
                    <p className="text-xs text-slate-500 mt-1">
                      1. Matching brief vector against proven CAC winners in ChromaDB · 2. Expanding 1-line brief into 4 Story Blocks &amp; Visual Setup with DSPy · 3. Scoring CAC Success Probability with XGBoost
                    </p>
                  </div>
                )}

                {result && (
                  <>
                    {/* Success Rate & Matched Winners Banner */}
                    <div className="bg-white border border-slate-200 rounded-xl p-5 shadow-2xs">
                      <div className="flex flex-wrap items-center justify-between gap-4 pb-4 border-b border-slate-100">
                        <div className="flex items-center gap-3.5">
                          <div className="w-14 h-14 rounded-xl bg-gradient-to-br from-emerald-500 to-teal-700 text-white flex flex-col items-center justify-center shadow-xs">
                            <span className="text-lg font-extrabold leading-none">
                              {result.predicted_success_rate_pct}%
                            </span>
                            <span className="text-[9px] font-mono uppercase mt-0.5">Success</span>
                          </div>
                          <div>
                            <div className="flex flex-wrap items-center gap-2">
                              <h3 className="text-sm font-bold text-slate-900">
                                Approximate Success Rate: {result.predicted_success_rate_pct}%
                              </h3>
                              <span className="text-[10px] font-mono font-semibold px-2 py-0.5 rounded-full bg-emerald-50 text-emerald-700 border border-emerald-200">
                                {result.confidence_label}
                              </span>
                              {result.token_usage && (
                                <span className="text-[10px] font-mono font-semibold px-2 py-0.5 rounded-full bg-purple-50 text-purple-700 border border-purple-200">
                                  Gemini Tokens: In {result.token_usage.input_tokens.toLocaleString()} · Out {result.token_usage.output_tokens.toLocaleString()} · Total {result.token_usage.total_tokens.toLocaleString()}
                                </span>
                              )}
                            </div>
                            <p className="text-xs text-slate-500 mt-0.5">
                              Historical Cluster Win Rate: {result.historical_cluster_win_rate_pct ?? 75}% · Pacing Match:{" "}
                              {result.pacing_alignment_score_pct ?? 90}% · Supernova Compliance:{" "}
                              {result.compliance_score_pct ?? 100}%
                            </p>
                          </div>
                        </div>

                        <div className="flex items-center gap-2">
                          {result.doc_url && (
                            <a
                              href={result.doc_url}
                              target="_blank"
                              rel="noopener noreferrer"
                              className="px-3 py-1.5 rounded-lg text-xs font-semibold bg-emerald-600 hover:bg-emerald-700 text-white flex items-center gap-1.5"
                            >
                              <span>Open Google Doc</span>
                              <ExternalLink className="w-3.5 h-3.5" />
                            </a>
                          )}
                          <button
                            type="button"
                            onClick={() => copyText(result.formatted_google_doc_text, "full_doc")}
                            className="px-3 py-1.5 rounded-lg text-xs font-semibold bg-slate-900 hover:bg-black text-white flex items-center gap-1.5 cursor-pointer"
                          >
                            {copiedKey === "full_doc" ? (
                              <>
                                <Check className="w-3.5 h-3.5 text-emerald-400" />
                                <span>Copied Complete Doc!</span>
                              </>
                            ) : (
                              <>
                                <Copy className="w-3.5 h-3.5" />
                                <span>Copy Complete Doc</span>
                              </>
                            )}
                          </button>
                        </div>
                      </div>

                      {/* Matched CAC Winners */}
                      {result.matched_winners && result.matched_winners.length > 0 && (
                        <div className="mt-3.5">
                          <div className="text-[11px] font-mono uppercase text-slate-500 mb-2">
                            Matched Proven CAC Winners from Brain:
                          </div>
                          <div className="grid grid-cols-1 sm:grid-cols-3 gap-2.5">
                            {result.matched_winners.map((w) => (
                              <div
                                key={w.ad_code}
                                className="p-2.5 rounded-lg bg-slate-50 border border-slate-200 text-xs"
                              >
                                <div className="flex items-center justify-between font-bold text-slate-900">
                                  <span>{w.ad_code}</span>
                                  <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-emerald-50 text-emerald-700 border border-emerald-200">
                                    {w.cac_tier || "TIER_S"}{" "}
                                    {w.cac_metrics?.cac_vs_avg_pct
                                      ? `(${w.cac_metrics.cac_vs_avg_pct > 0 ? "+" : ""}${w.cac_metrics.cac_vs_avg_pct}% CAC)`
                                      : ""}
                                  </span>
                                </div>
                                <div className="text-[11px] text-slate-600 mt-1 truncate">
                                  {w.language} · {w.metadata_tags?.persona || "Proven Winner"}
                                </div>
                              </div>
                            ))}
                          </div>
                        </div>
                      )}
                    </div>

                    {/* 3 Hook + Bridge Options */}
                    <div className="bg-white border border-slate-200 rounded-xl p-5 shadow-2xs">
                      <h3 className="text-xs font-mono uppercase font-bold text-purple-700 mb-3">
                        3 Recommended Hook + Bridge Options (Lines 1–2 → Converge on Line 3)
                      </h3>
                      <div className="flex flex-col gap-2.5">
                        {result.three_hooks.map((hookText, idx) => (
                          <div
                            key={idx}
                            className="p-3 rounded-lg bg-purple-50/40 border border-purple-200/80 text-xs text-slate-800 whitespace-pre-wrap leading-relaxed"
                          >
                            <span className="font-bold text-purple-900">
                              Option {String.fromCharCode(65 + idx)}:
                            </span>{" "}
                            {typeof hookText === "string" ? hookText : JSON.stringify(hookText)}
                          </div>
                        ))}
                      </div>
                    </div>

                    {/* Visual Description Block */}
                    <div className="bg-white border border-slate-200 rounded-xl p-5 shadow-2xs">
                      <div className="flex items-center justify-between mb-2">
                        <h3 className="text-xs font-mono uppercase font-bold text-indigo-700">
                          Generated Visual Description (Modeled on Winner Visual Blueprints)
                        </h3>
                        <button
                          type="button"
                          onClick={() => copyText(result.visual_description, "visual")}
                          className="text-[11px] text-slate-500 hover:text-slate-900 flex items-center gap-1 cursor-pointer"
                        >
                          {copiedKey === "visual" ? <Check className="w-3 h-3" /> : <Copy className="w-3 h-3" />}
                          <span>Copy</span>
                        </button>
                      </div>
                      <pre className="text-xs text-slate-800 whitespace-pre-wrap font-sans leading-relaxed bg-slate-50 p-3.5 rounded-lg border border-slate-200">
                        {result.visual_description}
                      </pre>
                    </div>

                    {/* Full Supernova Script */}
                    <div className="bg-white border border-slate-200 rounded-xl p-5 shadow-2xs">
                      <div className="flex items-center justify-between mb-2">
                        <h3 className="text-xs font-mono uppercase font-bold text-slate-900">
                          Complete Supernova Script
                        </h3>
                        <button
                          type="button"
                          onClick={() => copyText(result.full_supernova_script, "script")}
                          className="text-[11px] text-slate-500 hover:text-slate-900 flex items-center gap-1 cursor-pointer"
                        >
                          {copiedKey === "script" ? <Check className="w-3 h-3" /> : <Copy className="w-3 h-3" />}
                          <span>Copy Dialogue</span>
                        </button>
                      </div>
                      <pre className="text-xs text-slate-900 whitespace-pre-wrap font-sans leading-relaxed bg-slate-50 p-4 rounded-lg border border-slate-200">
                        {result.full_supernova_script}
                      </pre>
                    </div>

                    {/* 4-Block Strategy Plan */}
                    <div className="bg-white border border-slate-200 rounded-xl p-5 shadow-2xs">
                      <h3 className="text-xs font-mono uppercase font-bold text-slate-600 mb-2">
                        Middle Brain 4-Block Thinking Plan
                      </h3>
                      <pre className="text-xs text-slate-700 whitespace-pre-wrap font-sans leading-relaxed bg-slate-50 p-3.5 rounded-lg border border-slate-200">
                        {typeof result.four_block_thinking_plan === "string"
                          ? result.four_block_thinking_plan
                          : JSON.stringify(result.four_block_thinking_plan, null, 2)}
                      </pre>
                    </div>
                  </>
                )}
              </div>
            </div>

            {/* Trained Scripts in Brain Section also visible below Generator */}
            {renderTrainedScriptsSection()}
          </div>
        )}

        {/* TAB 2: TRAINED SCRIPTS IN BRAIN */}
        {activeTab === "database" && renderTrainedScriptsSection()}

        {/* TAB 3: PYTHON BRAIN & CONTRASTIVE PLAYBOOK */}
        {activeTab === "architecture" && (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
            <div className="bg-white border border-slate-200 rounded-xl p-5 shadow-2xs">
              <h3 className="text-xs font-mono uppercase font-bold text-slate-800 mb-3">
                Contrastive Winner-vs-Loser Rules (Extracted from CAC Data)
              </h3>
              <div className="flex flex-col gap-2 text-xs text-slate-700">
                {(playbook?.why_winners_won || [
                  "Tier-S winners open with an active spoken English mistake or embarrassing quote in <= 12 words.",
                  "The AI Teacher interrupts on Turn 2 in 100% of Tier-S winners.",
                  "All 4 Supernova USPs appear naturally in the first half after an on-the-spot sentence correction.",
                ]).map((rule: string, idx: number) => (
                  <div key={idx} className="p-2.5 rounded-lg bg-emerald-50/50 border border-emerald-200">
                    ✅ {rule}
                  </div>
                ))}
              </div>
            </div>

            <div className="bg-white border border-slate-200 rounded-xl p-5 shadow-2xs">
              <h3 className="text-xs font-mono uppercase font-bold text-slate-800 mb-3">
                High-CAC Loser Patterns to Avoid (Tier-B Anti-Patterns)
              </h3>
              <div className="flex flex-col gap-2 text-xs text-slate-700">
                {(playbook?.why_losers_failed || [
                  "Avoid slow introductory greetings or generic monologues ('Hello friends, do you want to speak English?').",
                  "Avoid delaying the AI Teacher's entrance past Turn 3.",
                  "Never use banned phrases like 'learn English' or '10 minutes'.",
                ]).map((rule: string, idx: number) => (
                  <div key={idx} className="p-2.5 rounded-lg bg-amber-50/60 border border-amber-200">
                    ⚠️ {rule}
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}
