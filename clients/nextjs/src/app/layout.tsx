import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Creative Coding Assistant",
  description: "Multi-panel creative workflow interface",
  icons: {
    icon: "/icon.svg"
  }
};

export default function RootLayout({
  children
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
