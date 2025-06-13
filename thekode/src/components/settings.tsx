"use client";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import { Input } from "./ui/input";
import { useFileStore } from "./Zustand";
import { useEffect, useState } from "react";
import { Button } from "./ui/button";
import { IconParkOutlineSettingConfig } from "./icons";
import { toast } from "sonner";

export function Settings() {
  const [serverURLTemp, setServerURLTemp] = useState("");
  const [active, setActive] = useState(false);
  const { server, setServer } = useFileStore();
  useEffect(() => {
    setServerURLTemp(server);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);
  return (
    <div className=" ">
      <Dialog >
        <DialogTrigger
          onClick={() => setActive(true)}
          className="text-xs flex gap-1 bg-white/10 border-white/20 border p-1 hover:bg-white/50 rounded-lg items-center justify-center"
        >
          Configure <IconParkOutlineSettingConfig />
        </DialogTrigger>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Configure</DialogTitle>
            <DialogDescription>
              Edit the host and port based on the hosted backend, the default url is <code >http://localhost:5000/</code>
            </DialogDescription>
          </DialogHeader>
          <div>
            <p>Server URL</p>
            <Input
              onChange={(e) => setServer(e.target.value)}
              value={server}
              placeholder="Server URL"
            />
          </div>
          {/* <div className="mt-2">
            <p>Reload <span className="font-mono text-xs p-2 bg-neutral-300/10  items-center justify-center rounded-md ">algos.json</span></p>
            <Button
              className="w-full"
              onClick={(e) => {
                fetch(`${server}/reload_algos`).then(r => r.json())
                  .then(d => {
                    if (d.status === 200) {

                      toast.success('Reloaded Successfully')
                    } else {
                      toast.error('There was an error reloading')

                    }
                  })
              }}

            >Reload</Button>
          </div> */}
        
        </DialogContent>
      </Dialog>
    </div>
  );
}
