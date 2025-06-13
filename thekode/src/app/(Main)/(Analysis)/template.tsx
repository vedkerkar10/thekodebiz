'use client'

import { useFileStore } from "@/components/Zustand";
import { useEffect } from "react";

export default function Template({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {

const { setServer, server } = useFileStore();

useEffect(() => {
    localStorage.setItem("server", server);
}, [server]);

useEffect(() => {
  const storedServer = localStorage.getItem("server");
  if (storedServer) {
    setServer(storedServer);
  }
}, [setServer]);
  return <> {children}</>;
}
