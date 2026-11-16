import type { NextConfig } from "next";

// S2E_DIST_DIR lets the browser tests build into their own folder, so a test build that points
// at the development fake server can never be served by `npm start`.
const nextConfig: NextConfig = {
  distDir: process.env.S2E_DIST_DIR ?? ".next",
  devIndicators: false,
};

export default nextConfig;
