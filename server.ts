import 'dotenv/config';
import express from 'express';
import cors from 'cors';
import fs from 'fs';
import path from 'path';
import { createServer as createViteServer } from 'vite';
import {
  getScriptBrainStatus,
  ingestProvenScriptsToBrain,
  ingestFromGoogleSheetToBrain,
  trainScriptBrain,
  generateSupernovaScriptFromBrief,
} from './ad-studio/script-brain/bridge.ts';

const app = express();
const PORT = Number(process.env.PORT || 4000);

app.use(cors());
app.use(express.json({ limit: '50mb' }));

app.get('/api/script-brain/status', async (req, res) => {
  try {
    const storePath = path.join(process.cwd(), 'ad-studio', 'script-brain', 'data', 'brain_store.json');
    const playbookPath = path.join(process.cwd(), 'ad-studio', 'script-brain', 'data', 'contrastive_playbook.json');
    const dspyPath = path.join(process.cwd(), 'ad-studio', 'script-brain', 'data', 'compiled_dspy_brain.json');
    const xgbPath = path.join(process.cwd(), 'ad-studio', 'script-brain', 'data', 'xgb_success_model.pkl');

    let records: any[] = [];
    if (fs.existsSync(storePath)) {
      const raw = JSON.parse(fs.readFileSync(storePath, 'utf-8'));
      records = Array.isArray(raw) ? raw : Object.values(raw.records || raw.scripts || raw || {});
    }

    const byTier: Record<string, number> = { TIER_S: 0, TIER_A: 0, TIER_B: 0 };
    const byLanguage: Record<string, number> = {};
    for (const r of records) {
      if (!r || typeof r !== 'object') continue;
      const t = String(r.cac_tier || 'TIER_A');
      byTier[t] = (byTier[t] || 0) + 1;
      const l = String(r.language || 'Unknown');
      byLanguage[l] = (byLanguage[l] || 0) + 1;
    }

    res.json({
      status: 'ok',
      total_records: records.length,
      db_stats: { total_scripts: records.length, by_tier: byTier, by_language: byLanguage },
      artifacts: {
        brain_store_json: { exists: fs.existsSync(storePath) },
        contrastive_playbook_json: { exists: fs.existsSync(playbookPath) },
        compiled_dspy_brain_json: { exists: fs.existsSync(dspyPath) },
        xgb_success_model_pkl: { exists: fs.existsSync(xgbPath) },
      },
    });
  } catch (error: any) {
    res.status(500).json({ error: error.message || 'Failed to read Script Brain status' });
  }
});

app.get('/api/script-brain/scripts', async (_req, res) => {
  try {
    const storePath = path.join(process.cwd(), 'ad-studio', 'script-brain', 'data', 'brain_store.json');
    if (!fs.existsSync(storePath)) {
      return res.json({ scripts: [], total: 0 });
    }
    const raw = JSON.parse(fs.readFileSync(storePath, 'utf-8'));
    const scripts = Array.isArray(raw) ? raw : Object.values(raw.records || raw.scripts || raw || {});
    res.json({ scripts, total: scripts.length });
  } catch (error: any) {
    res.status(500).json({ error: error.message || 'Failed to load trained scripts' });
  }
});

app.post('/api/script-brain/ingest', async (req, res) => {
  try {
    const items = Array.isArray(req.body?.items) ? req.body.items : [];
    const result = await ingestProvenScriptsToBrain(items);
    res.json(result);
  } catch (error: any) {
    res.status(500).json({ error: error.message || 'Failed to ingest scripts' });
  }
});

app.post('/api/script-brain/ingest-sheet', async (req, res) => {
  try {
    const { sheetId, tab, limit, reset } = req.body || {};
    const result = await ingestFromGoogleSheetToBrain({ sheetId, tab, limit, reset });
    res.json(result);
  } catch (error: any) {
    res.status(500).json({ error: error.message || 'Failed to ingest from Google Sheet' });
  }
});

app.post('/api/script-brain/train', async (_req, res) => {
  try {
    const result = await trainScriptBrain();
    res.json(result);
  } catch (error: any) {
    res.status(500).json({ error: error.message || 'Failed to train Script Brain' });
  }
});

app.post('/api/script-brain/generate', async (req, res) => {
  try {
    const { brief, language, character, createGoogleDoc, docTitle } = req.body || {};
    if (!brief || !String(brief).trim()) {
      return res.status(400).json({ error: 'Concept brief is required' });
    }
    const result = await generateSupernovaScriptFromBrief({
      brief: String(brief).trim(),
      language: String(language || 'Tamil').trim(),
      character: character ? String(character).trim() : 'Female Humanoid Robot',
      createGoogleDoc: Boolean(createGoogleDoc),
      docTitle: docTitle ? String(docTitle).trim() : undefined,
    });
    res.json(result);
  } catch (error: any) {
    res.status(500).json({ error: error.message || 'Failed to generate script' });
  }
});

async function start() {
  const vite = await createViteServer({
    server: { middlewareMode: true },
    appType: 'spa',
  });
  app.use(vite.middlewares);
  app.listen(PORT, () => {
    console.log(`Supernova Script Writer-AI running on http://localhost:${PORT}`);
  });
}

start();
