import { NextResponse } from 'next/server';
import fs from 'fs';
import { getBumbleLogFilePath } from '@/lib/bumbleLog';

export const dynamic = 'force-dynamic';

function parseOptionalNumber(value: string | undefined) {
  if (!value) return null;
  const parsed = parseFloat(value);
  return isNaN(parsed) ? null : parsed;
}

function displayFinalScore(rawScore: number | null, finalScore: number | null, decisionMode: string, preferenceProbability: number | null) {
  if (finalScore !== null) return finalScore;
  if (decisionMode === 'preference' && preferenceProbability !== null) {
    return preferenceProbability * 100;
  }
  return rawScore;
}

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);
  const limit = parseInt(searchParams.get('limit') || '50', 10);
  const offset = parseInt(searchParams.get('offset') || '0', 10);
  const setupNameFilter = searchParams.get('setupName') || '';
  const actionFilter = searchParams.get('action') || '';

  const filepath = getBumbleLogFilePath('scores.csv');
  try {
    const fileContent = fs.readFileSync(filepath, 'utf-8');
    const lines = fileContent.split('\n');
    
    if (lines.length === 0) {
      return NextResponse.json({ error: 'File is empty' }, { status: 400 });
    }

    const headers = lines[0].trim().split(',');
    const screenshotIdx = headers.indexOf('screenshot');
    const scoreIdx = headers.indexOf('score');
    const finalScoreIdx = headers.indexOf('final_score');
    const actionIdx = headers.indexOf('action');
    const tsIdx = headers.indexOf('timestamp');
    const fbIdx = headers.indexOf('face_biased');
    const mmIdx = headers.indexOf('multimodal');
    const rgIdx = headers.indexOf('ridge');
    const knIdx = headers.indexOf('knn');
    const decisionModeIdx = headers.indexOf('decision_mode');
    const preferenceProbabilityIdx = headers.indexOf('preference_probability');
    const setupNameIdx = headers.indexOf('setup_name');

    // Extended parameters
    const methodIdx = headers.indexOf('method');
    const storePathIdx = headers.indexOf('store_path');
    const regressorPathIdx = headers.indexOf('regressor_path');
    const multimodalRegressorPathIdx = headers.indexOf('multimodal_regressor_path');
    const thresholdIdx = headers.indexOf('threshold');
    const preferenceModelPathIdx = headers.indexOf('preference_model_path');
    const preferenceThresholdIdx = headers.indexOf('preference_threshold');
    const providerIdx = headers.indexOf('provider');
    const delayIdx = headers.indexOf('delay');
    const kIdx = headers.indexOf('k');
    const faceWeightIdx = headers.indexOf('face_weight');
    const mode247Idx = headers.indexOf('mode_247');

    // Dynamic standard threshold columns
    const dynamicEnabledIdx = headers.indexOf('dynamic_enabled');
    const dynamicModeIdx = headers.indexOf('dynamic_mode');
    const dynamicWindowIdx = headers.indexOf('dynamic_window');
    const dynamicTargetRightRateIdx = headers.indexOf('dynamic_target_right_rate');
    const dynamicPercentileIdx = headers.indexOf('dynamic_percentile');
    const dynamicMinHistoryIdx = headers.indexOf('dynamic_min_history');
    const dynamicMinThresholdIdx = headers.indexOf('dynamic_min_threshold');
    const dynamicMaxThresholdIdx = headers.indexOf('dynamic_max_threshold');

    // Dynamic preference threshold columns
    const dynamicPreferenceEnabledIdx = headers.indexOf('dynamic_preference_enabled');
    const dynamicPreferenceModeIdx = headers.indexOf('dynamic_preference_mode');
    const dynamicPreferenceWindowIdx = headers.indexOf('dynamic_preference_window');
    const dynamicPreferenceTargetRightRateIdx = headers.indexOf('dynamic_preference_target_right_rate');
    const dynamicPreferencePercentileIdx = headers.indexOf('dynamic_preference_percentile');
    const dynamicPreferenceMinHistoryIdx = headers.indexOf('dynamic_preference_min_history');
    const dynamicPreferenceMinThresholdIdx = headers.indexOf('dynamic_preference_min_threshold');
    const dynamicPreferenceMaxThresholdIdx = headers.indexOf('dynamic_preference_max_threshold');

    if (screenshotIdx === -1) {
      return NextResponse.json({ error: 'Missing screenshot column' }, { status: 400 });
    }

    const history = [];
    const uniqueSetups = new Set<string>();
    
    for (let i = lines.length - 1; i > 0; i--) {
      const line = lines[i].trim();
      if (!line) continue;
      
      const row = line.split(',');
      if (row.length <= Math.max(screenshotIdx, scoreIdx, actionIdx)) continue;
      
      const screenshot = row[screenshotIdx];
      if (!screenshot) continue;

      const fb = fbIdx !== -1 ? parseFloat(row[fbIdx]) : NaN;
      const mm = mmIdx !== -1 ? parseFloat(row[mmIdx]) : NaN;
      const rg = rgIdx !== -1 ? parseFloat(row[rgIdx]) : NaN;
      const kn = knIdx !== -1 ? parseFloat(row[knIdx]) : NaN;
      const rawScore = scoreIdx !== -1 ? parseOptionalNumber(row[scoreIdx]) : null;
      const finalScore = finalScoreIdx !== -1 ? parseOptionalNumber(row[finalScoreIdx]) : null;
      const decisionMode = decisionModeIdx !== -1 ? row[decisionModeIdx] || '' : '';
      const preferenceProbability = preferenceProbabilityIdx !== -1 ? parseOptionalNumber(row[preferenceProbabilityIdx]) : null;
      
      let setupName = setupNameIdx !== -1 ? row[setupNameIdx] || '' : '';
      if (!setupName) {
        setupName = 'Primitive';
      }
      uniqueSetups.add(setupName);

      // Filtering logic
      if (setupNameFilter) {
        if (setupName.toLowerCase() !== setupNameFilter.toLowerCase()) {
          continue;
        }
      }

      const actionVal = actionIdx !== -1 ? row[actionIdx]?.toLowerCase() || 'unknown' : 'unknown';
      if (actionFilter) {
        if (actionVal !== actionFilter.toLowerCase()) {
          continue;
        }
      }

      const method = methodIdx !== -1 ? row[methodIdx] || '' : '';
      const store_path = storePathIdx !== -1 ? row[storePathIdx] || '' : '';
      const regressor_path = regressorPathIdx !== -1 ? row[regressorPathIdx] || '' : '';
      const multimodal_regressor_path = multimodalRegressorPathIdx !== -1 ? row[multimodalRegressorPathIdx] || '' : '';
      const threshold = thresholdIdx !== -1 ? parseOptionalNumber(row[thresholdIdx]) : null;
      const preference_model_path = preferenceModelPathIdx !== -1 ? row[preferenceModelPathIdx] || '' : '';
      const preference_threshold = preferenceThresholdIdx !== -1 ? parseOptionalNumber(row[preferenceThresholdIdx]) : null;
      const provider = providerIdx !== -1 ? row[providerIdx] || '' : '';
      const delay = delayIdx !== -1 ? parseOptionalNumber(row[delayIdx]) : null;
      const k = kIdx !== -1 ? parseOptionalNumber(row[kIdx]) : null;
      const face_weight = faceWeightIdx !== -1 ? parseOptionalNumber(row[faceWeightIdx]) : null;
      const mode_247 = mode247Idx !== -1 ? row[mode247Idx]?.toLowerCase() === 'true' : false;

      // Dynamic standard threshold columns
      const dynamic_enabled = dynamicEnabledIdx !== -1 ? row[dynamicEnabledIdx]?.toLowerCase() === 'true' : false;
      const dynamic_mode = dynamicModeIdx !== -1 ? row[dynamicModeIdx] || '' : '';
      const dynamic_window = dynamicWindowIdx !== -1 ? parseOptionalNumber(row[dynamicWindowIdx]) : null;
      const dynamic_target_right_rate = dynamicTargetRightRateIdx !== -1 ? parseOptionalNumber(row[dynamicTargetRightRateIdx]) : null;
      const dynamic_percentile = dynamicPercentileIdx !== -1 ? parseOptionalNumber(row[dynamicPercentileIdx]) : null;
      const dynamic_min_history = dynamicMinHistoryIdx !== -1 ? parseOptionalNumber(row[dynamicMinHistoryIdx]) : null;
      const dynamic_min_threshold = dynamicMinThresholdIdx !== -1 ? parseOptionalNumber(row[dynamicMinThresholdIdx]) : null;
      const dynamic_max_threshold = dynamicMaxThresholdIdx !== -1 ? parseOptionalNumber(row[dynamicMaxThresholdIdx]) : null;

      // Dynamic preference threshold columns
      const dynamic_preference_enabled = dynamicPreferenceEnabledIdx !== -1 ? row[dynamicPreferenceEnabledIdx]?.toLowerCase() === 'true' : false;
      const dynamic_preference_mode = dynamicPreferenceModeIdx !== -1 ? row[dynamicPreferenceModeIdx] || '' : '';
      const dynamic_preference_window = dynamicPreferenceWindowIdx !== -1 ? parseOptionalNumber(row[dynamicPreferenceWindowIdx]) : null;
      const dynamic_preference_target_right_rate = dynamicPreferenceTargetRightRateIdx !== -1 ? parseOptionalNumber(row[dynamicPreferenceTargetRightRateIdx]) : null;
      const dynamic_preference_percentile = dynamicPreferencePercentileIdx !== -1 ? parseOptionalNumber(row[dynamicPreferencePercentileIdx]) : null;
      const dynamic_preference_min_history = dynamicPreferenceMinHistoryIdx !== -1 ? parseOptionalNumber(row[dynamicPreferenceMinHistoryIdx]) : null;
      const dynamic_preference_min_threshold = dynamicPreferenceMinThresholdIdx !== -1 ? parseOptionalNumber(row[dynamicPreferenceMinThresholdIdx]) : null;
      const dynamic_preference_max_threshold = dynamicPreferenceMaxThresholdIdx !== -1 ? parseOptionalNumber(row[dynamicPreferenceMaxThresholdIdx]) : null;

      history.push({
        id: i,
        timestamp: tsIdx !== -1 ? row[tsIdx] : '',
        screenshot: screenshot,
        score: displayFinalScore(rawScore, finalScore, decisionMode, preferenceProbability),
        action: actionVal,
        face_biased: isNaN(fb) ? null : fb,
        multimodal: isNaN(mm) ? null : mm,
        ridge: isNaN(rg) ? null : rg,
        knn: isNaN(kn) ? null : kn,
        setup_name: setupName,
        decision_mode: decisionMode,
        preference_probability: preferenceProbability,

        // Extended telemetry fields
        method,
        store_path,
        regressor_path,
        multimodal_regressor_path,
        threshold,
        preference_model_path,
        preference_threshold,
        provider,
        delay,
        k,
        face_weight,
        mode_247,

        // Dynamic standard threshold columns
        dynamic_enabled,
        dynamic_mode,
        dynamic_window,
        dynamic_target_right_rate,
        dynamic_percentile,
        dynamic_min_history,
        dynamic_min_threshold,
        dynamic_max_threshold,

        // Dynamic preference threshold columns
        dynamic_preference_enabled,
        dynamic_preference_mode,
        dynamic_preference_window,
        dynamic_preference_target_right_rate,
        dynamic_preference_percentile,
        dynamic_preference_min_history,
        dynamic_preference_min_threshold,
        dynamic_preference_max_threshold
      });
    }

    // Sort logic based on query parameter
    const sortBy = searchParams.get('sortBy') || 'latest';
    if (sortBy === 'oldest') {
      history.reverse();
    } else if (sortBy === 'attractive_desc') {
      history.sort((a, b) => {
        if (a.face_biased === null || a.face_biased === undefined) return 1;
        if (b.face_biased === null || b.face_biased === undefined) return -1;
        return b.face_biased - a.face_biased;
      });
    } else if (sortBy === 'attractive_asc') {
      history.sort((a, b) => {
        if (a.face_biased === null || a.face_biased === undefined) return 1;
        if (b.face_biased === null || b.face_biased === undefined) return -1;
        return a.face_biased - b.face_biased;
      });
    }

    const totalCount = history.length;
    const paginated = history.slice(offset, offset + limit);

    return NextResponse.json({ 
      data: paginated,
      total: totalCount,
      limit,
      offset,
      uniqueSetups: Array.from(uniqueSetups).sort()
    });
  } catch (error: unknown) {
    const message = error instanceof Error ? error.message : 'Unknown error';
    return NextResponse.json({ error: message }, { status: 500 });
  }
}
