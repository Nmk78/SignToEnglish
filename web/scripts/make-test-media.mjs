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

// Real signing for the tests against the real service: local copies derived from the public-domain
// video "COVID-19 Prevention Tips in ASL" (CDC, via Wikimedia Commons; docs/DATA_LICENSES.md),
// seconds 62.5 to 78.5, where the signer fills the frame. They are made only if that video is
// present, stay in the git-ignored tests/media/ folder, and are never committed.
export const signingSource = resolve(
  webRoot, "..", "tests", "fixtures", "clips", "commons_source", "COVID-19_Prevention_Tips_in_ASL.webm",
);
export const signingCameraFile = resolve(mediaDir, "asl_camera.mjpeg");
export const signingClipFile = resolve(mediaDir, "asl_clip.webm");
export const signingShortFile = resolve(mediaDir, "asl_short.webm");
export const SIGNING_START_S = 62.5;
export const SIGNING_CLIP_SECONDS = 16;
export const SIGNING_SHORT_SECONDS = 9;

function makeSigningMedia() {
  if (!existsSync(signingSource)) return;
  const input = ["-ss", String(SIGNING_START_S), "-i", signingSource];
  const picture = "fps=30,scale=640:360";
  if (!existsSync(signingCameraFile)) {
    ffmpeg([...input, "-t", String(SIGNING_CLIP_SECONDS), "-vf", picture, "-an",
      "-c:v", "mjpeg", "-q:v", "4", "-pix_fmt", "yuvj420p", "-f", "mjpeg", signingCameraFile]);
  }
  if (!existsSync(signingClipFile)) {
    ffmpeg([...input, "-t", String(SIGNING_CLIP_SECONDS), "-vf", picture, "-an",
      "-c:v", "libvpx", "-b:v", "1500k", "-g", "30", signingClipFile]);
  }
  if (!existsSync(signingShortFile)) {
    ffmpeg([...input, "-t", String(SIGNING_SHORT_SECONDS), "-vf", picture, "-an",
      "-c:v", "libvpx", "-b:v", "1500k", "-g", "30", signingShortFile]);
  }
}

function ffmpeg(args) {
  execFileSync("ffmpeg", ["-hide_banner", "-loglevel", "error", "-y", ...args], { stdio: "inherit" });
}

export function makeTestMedia() {
  mkdirSync(mediaDir, { recursive: true });
  if (!existsSync(cameraFile)) {
    // A red band on the left edge and a blue band on the right edge of the picture as the camera
    // delivers it, so a test can tell whether the frames sent to the service were mirrored.
    const bands =
      "drawbox=x=0:y=0:w=160:h=480:color=red:t=fill,drawbox=x=480:y=0:w=160:h=480:color=blue:t=fill";
    ffmpeg(["-f", "lavfi", "-i", `testsrc2=size=640x480:rate=30,${bands}`, "-t", "10",
      "-c:v", "mjpeg", "-q:v", "4", "-pix_fmt", "yuvj420p", "-f", "mjpeg", cameraFile]);
  }
  if (!existsSync(clipFile)) {
    ffmpeg(["-f", "lavfi", "-i", "testsrc2=size=640x480:rate=30", "-t", String(CLIP_SECONDS),
      "-c:v", "libvpx", "-b:v", "1M", "-g", "30", "-an", clipFile]);
  }
  if (!existsSync(badFile)) writeFileSync(badFile, "this is not a video\n");
  makeSigningMedia();
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  makeTestMedia();
  console.log(`make-test-media: files are in ${mediaDir}`);
}
