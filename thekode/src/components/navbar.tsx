"use client";
import {
  Menubar,
  MenubarContent,
  MenubarItem,
  MenubarMenu,
  MenubarSeparator,
  MenubarShortcut,
  MenubarTrigger,
} from "@/components/ui/menubar";
import Link from "next/link";
import { signOut, useSession } from "next-auth/react";
import { useFileStore } from "@/components/Zustand";
import { Button } from "./ui/button";
import { LogOut } from "lucide-react";
import { Settings } from "./settings";
import { toast } from "sonner";
import { JsonView } from "./jsonView";

export default function NavBar() {
  const session = useSession();
  const { server, stage } = useFileStore();
  return (
    <div className="p-4 border-b border-white/10 flex justify-between bg-white/10 backdrop-blur-lg">
      <div className="flex items-center justify-center gap-2">
        <Link href={"/"}>
          <img
            src={"/Logo.png"}
            className=" object-fit"
            alt="The Kode Logo"
            height={60}
            width={180}
          />
        </Link>
        <Settings />

        <button
          type="button"
          className="text-xs flex gap-1 bg-white/10 border-white/20 border p-1 hover:bg-white/50 rounded-lg items-center justify-center text-black"
          onClick={async () => {
            try {
              const response = await fetch(`${server}/reload_algos`);
              const data = await response.json();

              if (response.ok && data.status === 200) {
                toast.success("Reloaded Successfully");
              } else {
                toast.error("There was an error reloading");
              }
            } catch (error) {
              toast.error(
                `There was an error reloading ${(error as Error).message}`
              );
            }
          }}
        >
          Reload <span className="font-mono text-black/50">algos.json</span>
          {/* biome-ignore lint/a11y/noSvgWithoutTitle: <explanation> */}
          <svg
            className="size-3"
            xmlns="http://www.w3.org/2000/svg"
            width="32"
            height="32"
            viewBox="0 0 24 24"
          >
            <path
              fill="currentColor"
              d="M12.079 2.25c-4.794 0-8.734 3.663-9.118 8.333H2a.75.75 0 0 0-.528 1.283l1.68 1.666a.75.75 0 0 0 1.056 0l1.68-1.666a.75.75 0 0 0-.528-1.283h-.893c.38-3.831 3.638-6.833 7.612-6.833a7.66 7.66 0 0 1 6.537 3.643a.75.75 0 1 0 1.277-.786A9.16 9.16 0 0 0 12.08 2.25m8.761 8.217a.75.75 0 0 0-1.054 0L18.1 12.133a.75.75 0 0 0 .527 1.284h.899c-.382 3.83-3.651 6.833-7.644 6.833a7.7 7.7 0 0 1-6.565-3.644a.75.75 0 1 0-1.277.788a9.2 9.2 0 0 0 7.842 4.356c4.808 0 8.765-3.66 9.15-8.333H22a.75.75 0 0 0 .527-1.284z"
            />
          </svg>
        </button>
      </div>
      {/* <div>stage : {stage}</div> */}
      <div className="flex gap-4">
        {/* <p className="text-indigo-100/80 text-sm items-center text-center flex ">
          {session?.data?.user?.name
            ? `Signed In As ${session?.data?.user?.name}`
            : "Not Signed In"}
        </p>
        {session?.data?.user?.name && (
          <Button onClick={() => signOut()}>Log Out </Button>
        )} */}
      </div>

      {/* <Image
         src={"/profile.avif"}
         className=" object-fit size-9 rounded-full"
         alt=""
         height={100}
         width={100}
      /> */}
      <JsonView />
    </div>
  );
}

export function PlainNavBar() {
  return (
    <div className="p-4 border-b border-neutral-200 flex justify-between bg-white ">
      <Link href={"/"}>
        <img
          src={"/Logo.png"}
          className=" object-fit"
          alt="The Kode Logo"
          height={60}
          width={180}
        />
      </Link>
      <div className="flex gap-4"></div>

      {/* <Image
         src={"/profile.avif"}
         className=" object-fit size-9 rounded-full"
         alt=""
         height={100}
         width={100}
      /> */}
    </div>
  );
}
