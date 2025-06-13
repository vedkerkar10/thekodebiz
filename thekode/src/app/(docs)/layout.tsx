import type { Metadata } from "next";
import { PlainNavBar } from "@/components/navbar";

export const metadata: Metadata = {
  title: "The Kode | Docs",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <div className="   ">
      <PlainNavBar />

      {children}
    </div>
  );
}
