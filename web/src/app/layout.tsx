import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "SignToEnglish | live ASL to English captions",
  description:
    "Project-show demo: continuous ASL signing from a camera or a video file, translated to English sentences by a model running on this computer. One domain, one signer, experimental.",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
