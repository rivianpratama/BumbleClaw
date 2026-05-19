import { NextResponse } from 'next/server';
import fs from 'fs';

export const dynamic = 'force-dynamic';

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

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);
  const offsetStr = searchParams.get('offset');
  const offset = offsetStr ? parseInt(offsetStr, 10) : 0;

  const filepath = 'D:\\BumbleLog\\scores.csv';
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
    
    let methodDistribution: Record<string, number> = {};
    let scoreDistribution: Record<string, number> = {
      '1-20': 0, '20-40': 0, '40-60': 0, '60-80': 0, '80-100': 0
    };
    
    let latestScore: Record<string, any> | null = null;
    
    let firstTs = 0;
    let lastTs = 0;
    
    let totalValidRows = 0;

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
      records: scoreCount,
      totalRecords: totalValidRows
    });
  } catch (error: any) {
    return NextResponse.json({ error: error.message }, { status: 500 });
  }
}
