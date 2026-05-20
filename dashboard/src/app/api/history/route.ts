import { NextResponse } from 'next/server';
import fs from 'fs';
import { getBumbleLogFilePath } from '@/lib/bumbleLog';

export const dynamic = 'force-dynamic';

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);
  const limit = parseInt(searchParams.get('limit') || '50', 10);
  const offset = parseInt(searchParams.get('offset') || '0', 10);

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
    const actionIdx = headers.indexOf('action');
    const tsIdx = headers.indexOf('timestamp');

    if (screenshotIdx === -1) {
      return NextResponse.json({ error: 'Missing screenshot column' }, { status: 400 });
    }

    const history = [];
    
    for (let i = lines.length - 1; i > 0; i--) {
      const line = lines[i].trim();
      if (!line) continue;
      
      const row = line.split(',');
      if (row.length <= Math.max(screenshotIdx, scoreIdx, actionIdx)) continue;
      
      const screenshot = row[screenshotIdx];
      if (!screenshot) continue;

      history.push({
        id: i,
        timestamp: tsIdx !== -1 ? row[tsIdx] : '',
        screenshot: screenshot,
        score: parseFloat(row[scoreIdx]),
        action: row[actionIdx]?.toLowerCase() || 'unknown'
      });
    }

    const totalCount = history.length;
    const paginated = history.slice(offset, offset + limit);

    return NextResponse.json({ 
      data: paginated,
      total: totalCount,
      limit,
      offset
    });
  } catch (error: unknown) {
    const message = error instanceof Error ? error.message : 'Unknown error';
    return NextResponse.json({ error: message }, { status: 500 });
  }
}
