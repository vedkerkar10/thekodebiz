"use client";
import { useFileStore } from "@/components/Zustand";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { uuidv4 } from "@/lib/utils";
import { HardDriveDownload, UnplugIcon } from "lucide-react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { useState } from "react";

export default function Page() {
  const {setLoading,loading}=useFileStore()
  const router = useRouter();
  const handleCreateNewSession = () => {
    setLoading(true);
    router.push(`/Home?SessionID=${uuidv4()}`);
    setLoading(false);
  };
  return (
    <main className="flex justify-center  items-start  w-full h-screen">
      {/* <div
        style={{
          pointerEvents: loading ? "auto" : "none",
          opacity: loading ? "1" : "0",
          backdropFilter: loading ? "blur(2px)" : "blur(0px)",
        }}
        className=" transition-all duration-300 absolute w-full h-full bg-neutral-800/50  0 grid place-items-center z-10"
      >
        <div
          style={{
            scale: loading ? 1 : 0.3,
          }}
          className=" transition-all duration-300 h-40 w-32 bg-white/50 backdrop-blur-lg -translate-y-20 rounded-md flex items-center justify-center border-black/10 border"
        >
          <UnplugIcon className="size-12" />
        </div>
      </div> */}
      <div className="p-2 bg-slate-50/10 backdrop-blur-xl rounded-md border border-white/10 mt-32 ">
        <Tabs defaultValue="create" className="w-full">
          <TabsList className="bg-white/10 text-black/50 ">
            <TabsTrigger value="create">Create A New Session</TabsTrigger>
            <TabsTrigger value="join">Join An Existing Session</TabsTrigger>
          </TabsList>
          <TabsContent value="create">
            <Button
              onClick={handleCreateNewSession}
              className="w-full flex gap-2"
              // className="flex gap-2 mt-2 w-full bg-neutral-900 text-white/80 font-medium items-center justify-center p-2 rounded-md"
            >
              Create A New Session <UnplugIcon className="size-4" />
            </Button>
          </TabsContent>
          <TabsContent value="join">
            <Label>Session ID</Label>
            <Input className="bg-white/10 border-white/10" />
            <Button className="flex gap-2 mt-2 w-full">
              Connect to Session <UnplugIcon className="size-4" />
            </Button>
          </TabsContent>
        </Tabs>
      </div>
    </main>
  );
}
