import { NextResponse } from 'next/server';
import { constants, createReadStream } from 'node:fs';
import { access } from 'node:fs/promises';
import { Readable } from 'node:stream';
import path from 'path';
import { getBumbleLogFilePath } from '@/lib/bumbleLog';

export const dynamic = 'force-dynamic';

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);
  const file = searchParams.get('file');

  if (!file) {
    return new NextResponse('File missing', { status: 400 });
  }

  const safeFile = path.basename(file);
  const filePath = getBumbleLogFilePath(safeFile);

  try {
    try {
      await access(filePath, constants.R_OK);
    } catch {
      return new NextResponse('File not found', { status: 404 });
    }

    const ext = path.extname(safeFile).toLowerCase();
    
    let contentType = 'image/jpeg';
    if (ext === '.webp') contentType = 'image/webp';
    else if (ext === '.png') contentType = 'image/png';
    else if (ext === '.gif') contentType = 'image/gif';

    const stream = Readable.toWeb(createReadStream(filePath)) as ReadableStream;

    return new NextResponse(stream, {
      headers: {
        'Content-Type': contentType,
        'Cache-Control': 'no-store, max-age=0',
      },
    });
  } catch (err: unknown) {
    const message = err instanceof Error ? err.message : 'Unknown error';
    return new NextResponse(message, { status: 500 });
  }
}
