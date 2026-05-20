import { NextResponse } from 'next/server';
import fs from 'fs';
import { getBumbleLogFilePath } from '@/lib/bumbleLog';

export const dynamic = 'force-dynamic';

const DYNAMIC_THRESHOLD = {
  fallbackThreshold: 54,
  targetRightRate: 0.25,
  window: 200,
  minHistory: 50,
  minThreshold: 48,
  maxThreshold: 62,
};

type ParsedScoreRow = {
  score: number | null;
  action: string;
  method: string;
  faceWeight: string;
  regressorPath: string;
  multimodalRegressorPath: string;
  threshold: number | null;
  dynamicEnabled: boolean | null;
  dynamicMode: string;
  dynamicWindow: number | null;
  dynamicTargetRightRate: number | null;
  dynamicPercentile: number | null;
  dynamicMinHistory: number | null;
  dynamicMinThreshold: number | null;
  dynamicMaxThreshold: number | null;
  ts: number;
};

type LatestScore = {
  score: number | null;
  face_biased: number | null;
  multimodal: number | null;
  ridge: number | null;
  knn: number | null;
  action: string;
  method: string;
  screenshot: string | null;
  divergence: number | null;
};

function parseTimestamp(ts: string) {
  if (!ts || ts.length < 15) return 0;
  const year = parseInt(ts.substring(0, 4));
  const month = parseInt(ts.substring(4, 6)) - 1;
  const day = parseInt(ts.substring(6, 8));
  const hour = parseInt(ts.substring(9, 11));
  const minute = parseInt(ts.substring(11, 13));
  const second = parseInt(ts.substring(13, 15));
  return new Date(year, month, day, hour, minute, second).getTime();
}

function normalizeConfigValue(value: string) {
  return value.replace(/\//g, '\\').toLowerCase();
}

function rowMatchesConfig(row: ParsedScoreRow, currentConfig: ParsedScoreRow | null) {
  if (!currentConfig) return false;
  const fields: Array<keyof Pick<ParsedScoreRow, 'method' | 'faceWeight' | 'regressorPath' | 'multimodalRegressorPath'>> = [
    'method',
    'faceWeight',
    'regressorPath',
    'multimodalRegressorPath',
  ];

  return fields.every((field) => {
    const actual = row[field];
    const expected = currentConfig[field];
    if (!actual) return true;
    return normalizeConfigValue(actual) === normalizeConfigValue(expected);
  });
}

function quantile(values: number[], probability: number) {
  const ordered = [...values].sort((a, b) => a - b);
  if (ordered.length === 0) return null;
  if (probability <= 0) return ordered[0];
  if (probability >= 1) return ordered[ordered.length - 1];
  const position = (ordered.length - 1) * probability;
  const lower = Math.floor(position);
  const upper = Math.ceil(position);
  if (lower === upper) return ordered[lower];
  return ordered[lower] * (upper - position) + ordered[upper] * (position - lower);
}

function clamp(value: number, minimum: number, maximum: number) {
  return Math.max(minimum, Math.min(maximum, value));
}

function parseOptionalNumber(value: string | undefined) {
  if (!value) return null;
  const parsed = parseFloat(value);
  return isNaN(parsed) ? null : parsed;
}

function parseOptionalBoolean(value: string | undefined) {
  if (!value) return null;
  const normalized = value.toLowerCase();
  if (normalized === 'true') return true;
  if (normalized === 'false') return false;
  return null;
}

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);
  const offsetStr = searchParams.get('offset');
  const offset = offsetStr ? parseInt(offsetStr, 10) : 0;

  const filepath = getBumbleLogFilePath('scores.csv');
  try {
    const fileContent = fs.readFileSync(filepath, 'utf-8');
    const lines = fileContent.split('\n');
    
    if (lines.length === 0) {
      return NextResponse.json({ error: 'File is empty' }, { status: 400 });
    }

    const headers = lines[0].trim().split(',');
    const tsIdx = headers.indexOf('timestamp');
    const methodIdx = headers.indexOf('method');
    const actionIdx = headers.indexOf('action');
    const scoreIdx = headers.indexOf('score');
    const screenshotIdx = headers.indexOf('screenshot');
    const fbIdx = headers.indexOf('face_biased');
    const mmIdx = headers.indexOf('multimodal');
    const rgIdx = headers.indexOf('ridge');
    const knIdx = headers.indexOf('knn');
    const regressorPathIdx = headers.indexOf('regressor_path');
    const multimodalRegressorPathIdx = headers.indexOf('multimodal_regressor_path');
    const thresholdIdx = headers.indexOf('threshold');
    const dynamicEnabledIdx = headers.indexOf('dynamic_enabled');
    const dynamicModeIdx = headers.indexOf('dynamic_mode');
    const dynamicWindowIdx = headers.indexOf('dynamic_window');
    const dynamicTargetRightRateIdx = headers.indexOf('dynamic_target_right_rate');
    const dynamicPercentileIdx = headers.indexOf('dynamic_percentile');
    const dynamicMinHistoryIdx = headers.indexOf('dynamic_min_history');
    const dynamicMinThresholdIdx = headers.indexOf('dynamic_min_threshold');
    const dynamicMaxThresholdIdx = headers.indexOf('dynamic_max_threshold');
    const faceWeightIdx = headers.indexOf('face_weight');

    if ([fbIdx, mmIdx, rgIdx, knIdx, actionIdx, scoreIdx, methodIdx, tsIdx].includes(-1)) {
      return NextResponse.json({ error: 'Missing columns' }, { status: 400 });
    }

    let fbTotal = 0, fbCount = 0;
    let mmTotal = 0, mmCount = 0;
    let rgTotal = 0, rgCount = 0;
    let knTotal = 0, knCount = 0;
    let scoreTotal = 0, scoreCount = 0;
    
    let leftCount = 0, rightCount = 0;
    let leftScoreTotal = 0, rightScoreTotal = 0;
    
    const methodDistribution: Record<string, number> = {};
    const scoreDistribution: Record<string, number> = {
      '1-20': 0, '20-40': 0, '40-60': 0, '60-80': 0, '80-100': 0
    };
    
    let latestScore: LatestScore | null = null;
    
    let firstTs = 0;
    let lastTs = 0;
    
    let totalValidRows = 0;
    const parsedRows: ParsedScoreRow[] = [];

    for (let i = 1; i < lines.length; i++) {
      const line = lines[i].trim();
      if (!line) continue;
      
      const row = line.split(',');
      if (row.length <= Math.max(fbIdx, mmIdx, rgIdx, knIdx, actionIdx, scoreIdx)) continue;
      
      totalValidRows++;

      const tsVal = parseTimestamp(row[tsIdx]);
      const method = row[methodIdx];
      const overallScore = parseFloat(row[scoreIdx]);
      const action = row[actionIdx]?.toLowerCase();
      const fb = parseFloat(row[fbIdx]);
      const mm = parseFloat(row[mmIdx]);
      const rg = parseFloat(row[rgIdx]);
      const kn = parseFloat(row[knIdx]);
      const parsedRow = {
        score: isNaN(overallScore) ? null : overallScore,
        action,
        method,
        faceWeight: faceWeightIdx !== -1 ? row[faceWeightIdx] || '' : '',
        regressorPath: regressorPathIdx !== -1 ? row[regressorPathIdx] || '' : '',
        multimodalRegressorPath: multimodalRegressorPathIdx !== -1 ? row[multimodalRegressorPathIdx] || '' : '',
        threshold: thresholdIdx !== -1 ? parseOptionalNumber(row[thresholdIdx]) : null,
        dynamicEnabled: dynamicEnabledIdx !== -1 ? parseOptionalBoolean(row[dynamicEnabledIdx]) : null,
        dynamicMode: dynamicModeIdx !== -1 ? row[dynamicModeIdx] || '' : '',
        dynamicWindow: dynamicWindowIdx !== -1 ? parseOptionalNumber(row[dynamicWindowIdx]) : null,
        dynamicTargetRightRate: dynamicTargetRightRateIdx !== -1 ? parseOptionalNumber(row[dynamicTargetRightRateIdx]) : null,
        dynamicPercentile: dynamicPercentileIdx !== -1 ? parseOptionalNumber(row[dynamicPercentileIdx]) : null,
        dynamicMinHistory: dynamicMinHistoryIdx !== -1 ? parseOptionalNumber(row[dynamicMinHistoryIdx]) : null,
        dynamicMinThreshold: dynamicMinThresholdIdx !== -1 ? parseOptionalNumber(row[dynamicMinThresholdIdx]) : null,
        dynamicMaxThreshold: dynamicMaxThresholdIdx !== -1 ? parseOptionalNumber(row[dynamicMaxThresholdIdx]) : null,
        ts: tsVal,
      };
      parsedRows.push(parsedRow);
      
      const validScores = [fb, mm, rg, kn].filter(v => !isNaN(v));
      let divergence = null;
      if (validScores.length > 1) {
        const mean = validScores.reduce((a, b) => a + b, 0) / validScores.length;
        const variance = validScores.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / validScores.length;
        divergence = Math.sqrt(variance);
      }
      
      latestScore = {
        score: isNaN(overallScore) ? null : overallScore,
        face_biased: isNaN(fb) ? null : fb,
        multimodal: isNaN(mm) ? null : mm,
        ridge: isNaN(rg) ? null : rg,
        knn: isNaN(kn) ? null : kn,
        action: action,
        method: method,
        screenshot: screenshotIdx !== -1 ? row[screenshotIdx] : null,
        divergence
      };

      if (totalValidRows <= offset) continue;

      if (tsVal > 0) {
        if (firstTs === 0) firstTs = tsVal;
        lastTs = tsVal;
      }

      if (method) {
        methodDistribution[method] = (methodDistribution[method] || 0) + 1;
      }

      if (!isNaN(overallScore)) {
        scoreTotal += overallScore;
        scoreCount++;
        
        if (overallScore <= 20) scoreDistribution['1-20']++;
        else if (overallScore <= 40) scoreDistribution['20-40']++;
        else if (overallScore <= 60) scoreDistribution['40-60']++;
        else if (overallScore <= 80) scoreDistribution['60-80']++;
        else scoreDistribution['80-100']++;
      }

      if (action === 'left') {
        leftCount++;
        if (!isNaN(overallScore)) leftScoreTotal += overallScore;
      } else if (action === 'right') {
        rightCount++;
        if (!isNaN(overallScore)) rightScoreTotal += overallScore;
      }

      if (!isNaN(fb)) { fbTotal += fb; fbCount++; }
      if (!isNaN(mm)) { mmTotal += mm; mmCount++; }
      if (!isNaN(rg)) { rgTotal += rg; rgCount++; }
      if (!isNaN(kn)) { knTotal += kn; knCount++; }
    }

    const activeConfig: ParsedScoreRow | null = parsedRows.length > 0 ? parsedRows[parsedRows.length - 1] : null;
    const cliWindow = activeConfig?.dynamicWindow ?? DYNAMIC_THRESHOLD.window;
    const cliPercentile = activeConfig?.dynamicPercentile;
    const cliTargetRightRate = activeConfig?.dynamicTargetRightRate
      ?? (cliPercentile === null || cliPercentile === undefined
        ? DYNAMIC_THRESHOLD.targetRightRate
        : 1 - ((cliPercentile > 1 ? cliPercentile / 100 : cliPercentile)));
    const configuredMinHistory = activeConfig?.dynamicMinHistory ?? DYNAMIC_THRESHOLD.minHistory;
    const cliMinHistory = Math.min(configuredMinHistory, cliWindow);
    const cliMinThreshold = activeConfig?.dynamicMinThreshold ?? DYNAMIC_THRESHOLD.minThreshold;
    const cliMaxThreshold = activeConfig?.dynamicMaxThreshold ?? DYNAMIC_THRESHOLD.maxThreshold;

    const recentScores = parsedRows
      .filter((row) => row.score !== null && rowMatchesConfig(row, activeConfig))
      .slice(-cliWindow);
    const recentScoreValues = recentScores.map((row) => row.score as number);
    const rawThreshold = recentScores.length >= cliMinHistory
      ? quantile(recentScoreValues, 1 - cliTargetRightRate)
      : null;
    const computedThreshold = rawThreshold === null
      ? DYNAMIC_THRESHOLD.fallbackThreshold
      : clamp(rawThreshold, cliMinThreshold, cliMaxThreshold);
    const effectiveThreshold = activeConfig?.threshold ?? computedThreshold;
    const projectedRightCount = recentScoreValues.filter((score) => score >= effectiveThreshold).length;
    const actualRightCount = recentScores.filter((row) => row.action === 'right').length;
    const actualLeftCount = recentScores.filter((row) => row.action === 'left').length;
    const actualSwipeCount = actualLeftCount + actualRightCount;

    const totalSwipes = leftCount + rightCount;
    const leftPercent = totalSwipes > 0 ? (leftCount / totalSwipes) * 100 : 0;
    const rightPercent = totalSwipes > 0 ? (rightCount / totalSwipes) * 100 : 0;

    let velocityPerMinute = 0;
    if (lastTs > firstTs && scoreCount > 1) {
      const minutes = (lastTs - firstTs) / (1000 * 60);
      if (minutes > 0) {
        velocityPerMinute = scoreCount / minutes;
      }
    }

    const trend: { time: number; threshold: number; score: number | null; action: string | null }[] = [];
    const matchingScores: number[] = [];
    
    // We only need the past 24 hours of data max for the UI (1 day).
    const oneDayAgo = Date.now() - 24 * 60 * 60 * 1000;
    
    for (const row of parsedRows) {
      if (!rowMatchesConfig(row, activeConfig)) continue;
      
      if (row.score !== null) {
        matchingScores.push(row.score);
      }
      
      // Only keep points within the 1-day UI window
      if (row.ts < oneDayAgo) continue;

      let currentThreshold = row.threshold;
      
      const w = row.dynamicWindow ?? cliWindow;
      const minHist = row.dynamicMinHistory ?? cliMinHistory;
      const tgtRate = row.dynamicTargetRightRate ?? cliTargetRightRate;
      
      if (currentThreshold === null) {
         if (matchingScores.length >= minHist) {
            const recent = matchingScores.slice(-w);
            const raw = quantile(recent, 1 - tgtRate);
            if (raw !== null) {
               currentThreshold = clamp(raw, row.dynamicMinThreshold ?? cliMinThreshold, row.dynamicMaxThreshold ?? cliMaxThreshold);
            } else {
               currentThreshold = DYNAMIC_THRESHOLD.fallbackThreshold;
            }
         } else {
            currentThreshold = DYNAMIC_THRESHOLD.fallbackThreshold;
         }
      }
      
      trend.push({
         time: row.ts,
         threshold: currentThreshold,
         score: row.score,
         action: row.action
      });
    }

    return NextResponse.json({
      averages: {
        score: scoreCount > 0 ? (scoreTotal / scoreCount) : 0,
        face_biased: fbCount > 0 ? (fbTotal / fbCount) : 0,
        multimodal: mmCount > 0 ? (mmTotal / mmCount) : 0,
        ridge: rgCount > 0 ? (rgTotal / rgCount) : 0,
        knn: knCount > 0 ? (knTotal / knCount) : 0,
      },
      actionAverages: {
        left: leftCount > 0 ? (leftScoreTotal / leftCount) : 0,
        right: rightCount > 0 ? (rightScoreTotal / rightCount) : 0,
      },
      swipes: {
        left: leftCount,
        right: rightCount,
        leftPercent,
        rightPercent,
      },
      methodDistribution,
      scoreDistribution,
      velocity: velocityPerMinute,
      latest: latestScore,
      dynamicThreshold: {
        enabled: true,
        active: rawThreshold !== null,
        threshold: effectiveThreshold,
        rawThreshold,
        fallbackThreshold: DYNAMIC_THRESHOLD.fallbackThreshold,
        source: activeConfig?.threshold == null ? 'computed' : 'cli_log',
        mode: activeConfig?.dynamicMode || null,
        targetRightRate: cliTargetRightRate,
        percentile: (1 - cliTargetRightRate) * 100,
        window: cliWindow,
        minHistory: cliMinHistory,
        minThreshold: cliMinThreshold,
        maxThreshold: cliMaxThreshold,
        historyCount: recentScores.length,
        projectedRightCount,
        projectedRightPercent: recentScores.length > 0 ? (projectedRightCount / recentScores.length) * 100 : 0,
        actualLeftCount,
        actualRightCount,
        actualRightPercent: actualSwipeCount > 0 ? (actualRightCount / actualSwipeCount) * 100 : 0,
      },
      trend,
      records: scoreCount,
      totalRecords: totalValidRows
    });
  } catch (error: unknown) {
    const message = error instanceof Error ? error.message : 'Unknown error';
    return NextResponse.json({ error: message }, { status: 500 });
  }
}
