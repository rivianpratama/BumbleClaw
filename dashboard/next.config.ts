import type { NextConfig } from "next";
import path from "path";

const nextConfig: NextConfig = {
  allowedDevOrigins: ['192.168.100.199', '192.168.100.199:7863'],
  turbopack: {
    root: path.resolve(__dirname),
  },
  outputFileTracingExcludes: {
    '/api/image': ['D:/BumbleLog/**/*'],
    '/api/scores': ['D:/BumbleLog/**/*'],
    '/api/history': ['D:/BumbleLog/**/*'],
  },
  experimental: {
    webpackMemoryOptimizations: true,
  },
};

export default nextConfig;
