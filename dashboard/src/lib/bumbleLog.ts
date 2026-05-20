import path from 'node:path';

const DEFAULT_LOG_DRIVE = ['D', ':'].join('');
const DEFAULT_LOG_FOLDER = 'BumbleLog';

export function getBumbleLogDir() {
  return process.env.BUMBLE_LOG_DIR || path.join(`${DEFAULT_LOG_DRIVE}${path.sep}`, DEFAULT_LOG_FOLDER);
}

export function getBumbleLogFilePath(file: string) {
  return path.join(getBumbleLogDir(), path.basename(file));
}
