import type { Metadata } from "next";
import { Inter } from "next/font/google";
import { LoaderDisplay } from "@/components/loader";
import NavBar from "@/components/navbar";
import { Toaster } from "@/components/ui/sonner";
import ClientSessionProvider from "@/components/ClientSessionProvider";


export const metadata: Metadata = {
  title: "The Kode",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
			<main className="h-screen overflow-hidden">
				<ClientSessionProvider>
					<NavBar />
					<LoaderDisplay />

					{children}

					<Toaster richColors />
				</ClientSessionProvider>
			</main>
		);
}
