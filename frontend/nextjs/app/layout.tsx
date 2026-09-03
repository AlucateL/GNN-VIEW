import type { Metadata } from "next";
import { Nunito } from "next/font/google";
import "./globals.css";
import { buildCssVars } from '../src/lib/colors';

const nunito = Nunito({
  variable: "--font-nunito",
  subsets: ["latin"],
  weight: ["400", "500", "600", "700", "800"],
  display: "swap",
});

export const metadata: Metadata = {
  title: "AI Earth Model Explorer",
  description: "Interactive GNN graph visualizer for AI weather model exploration",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <head><style dangerouslySetInnerHTML={{ __html: buildCssVars() }} /></head>
      <body className={`${nunito.variable} antialiased`}>
        {children}
      </body>
    </html>
  );
}
