// Generates synthetic media for the browser tests with ffmpeg test patterns. No recording and no
// dataset clip is used. Output goes to tests/media/, which is git-ignored.
//   camera.mjpeg  fed to Chromium as the fake webcam (--use-file-for-fake-video-capture)
//   clip.webm     the file chosen in the video-input panel
//   not-a-video.txt   a text file, for the rejected-upload test
import { execFileSync } from "node:child_process";
import { existsSync, mkdirSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const webRoot = resolve(dirname(fileURLToPath(import.meta.url)), "..");
export const mediaDir = resolve(webRoot, "tests", "media");
export const cameraFile = resolve(mediaDir, "camera.mjpeg");
export const clipFile = resolve(mediaDir, "clip.webm");
export const badFile = resolve(mediaDir, "not-a-video.txt");
export const CLIP_SECONDS = 6;

function ffmpeg(args) {
  execFileSync("ffmpeg", ["-hide_banner", "-loglevel", "error", "-y", ...args], { stdio: "inherit" });
}

export function makeTestMedia() {
  mkdirSync(mediaDir, { recursive: true });
  if (!existsSync(cameraFile)) {
    ffmpeg(["-f", "lavfi", "-i", "testsrc2=size=640x480:rate=30", "-t", "10",
      "-c:v", "mjpeg", "-q:v", "4", "-pix_fmt", "yuvj420p", "-f", "mjpeg", cameraFile]);
  }
  if (!existsSync(clipFile)) {
    ffmpeg(["-f", "lavfi", "-i", "testsrc2=size=640x480:rate=30", "-t", String(CLIP_SECONDS),
      "-c:v", "libvpx", "-b:v", "1M", "-g", "30", "-an", clipFile]);
  }
  if (!existsSync(badFile)) writeFileSync(badFile, "this is not a video\n");
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  makeTestMedia();
  console.log(`make-test-media: files are in ${mediaDir}`);
}
