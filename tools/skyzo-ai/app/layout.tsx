import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Skyzo-AI - Multi-Agent LLM System",
  description: "5 agents working in parallel, coordinated by a leader",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className="antialiased">
        {children}
      </body>
    </html>
  );
}
