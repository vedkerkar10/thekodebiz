import { CopyIcon, Mail, Share2 } from "lucide-react";
import { Button } from "./ui/button";
import { Label } from "./ui/label";
import { useFileStore } from "@/components/Zustand";
import {
  Dialog,
  DialogClose,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "./ui/dialog";
import { Input } from "./ui/input";
import { usePathname, useRouter } from "next/navigation";
import { useEffect, useState } from "react";
import Link from "next/link";
import { LogosWhatsappIcon, MaterialSymbolsMail } from "./icons";
import { toast } from "sonner";

export default function SessionDisplay() {
  const { sessionID } = useFileStore();
  // const router=usePathname()
  const [fullURL, setFullURL] = useState("");
  useEffect(() => {
    setFullURL(typeof window !== "undefined" ? window.location.href : "");
  }, []);
  return (
    <div className="w-full bg-white/10 backdrop-blur-md  rounded-md p-2 border border-white/10">
      <Label>Session ID :</Label>
      <p className="text-sm font-mono">{sessionID}</p>

      <Dialog>
        <DialogTrigger asChild>
          <Button className="w-full mt-1 flex gap-2">
            Share <Share2 className="size-4" />
          </Button>
        </DialogTrigger>
        <DialogContent className="sm:max-w-md bg-white/10 backdrop-blur-xl border-white/10">
          <DialogHeader>
            <DialogTitle className="text-white">Share link</DialogTitle>
            <DialogDescription>
              Anyone who has access will be able to view this.
            </DialogDescription>
          </DialogHeader>
          <div className="flex items-center space-x-2">
            <div className="grid flex-1 gap-2">
              <Label htmlFor="link" className="sr-only">
                Link
              </Label>
              <Input
                className="bg-white/10 border-white/10"
                id="link"
                defaultValue={fullURL}
                readOnly
              />
            </div>

            <Button
              type="button"
              onClick={() => {
                toast("Session Link Copied to clipboard");
                navigator.clipboard.writeText(`${fullURL}`);
              }}
              size="sm"
              className=""
            >
              <span className="sr-only">Copy</span>
              <CopyIcon className="h-4 w-4" />
            </Button>
          </div>
          <div className="flex gap-2">
            <Link
              className="text-2xl bg-white/10 border-white/10 border   size-12 flex items-center  justify-center rounded-md"
              href={`https://wa.me/?text=Hey! Check out my work at The Kode here: ${fullURL} or use this ID: ${sessionID}`}
            >
              <LogosWhatsappIcon />
            </Link>
            <Link
              className="text-2xl bg-white/10 border-white/10 border   size-12 flex items-center  justify-center rounded-md"
              href={`mailto:?subject=${encodeURIComponent(
                "Check out my work at The Kode"
              )}&body=${encodeURIComponent(
                `Hey! Check out my work at The Kode here: ${fullURL} or use this ID: ${sessionID}`
              )}`}
            >
              <MaterialSymbolsMail className="text-white" />
            </Link>
          </div>
        </DialogContent>
      </Dialog>
    </div>
  );
}
