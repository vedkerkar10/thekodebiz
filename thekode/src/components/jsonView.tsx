"use client";

import { useState } from "react";
import { createPortal } from "react-dom";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { coldarkCold } from "react-syntax-highlighter/dist/esm/styles/prism";
import { Button } from "./ui/button";
import { FileJson } from "lucide-react";
import { create } from "zustand";

interface AlgoStore {
    algosJSON: unknown;
    setAlgosJSON: (json: unknown) => void;
}

export const useAlgoJSONStore = create<AlgoStore>((set) => ({
    algosJSON: {},
    setAlgosJSON: (json: unknown) => set({ algosJSON: json }),
}));

export function JsonView() {
    const { algosJSON } = useAlgoJSONStore();
    const formattedJSON = JSON.stringify(algosJSON, null, 2);
    const [isVisible, setIsVisible] = useState(false);

    const toggleVisibility = () => setIsVisible((prev) => !prev);

    // Get the portal root directly without state.
    const portalRoot =
        typeof window !== "undefined" && document.getElementById("json-debug-portal-root");
    if (!portalRoot) return null;

    const portalContent = (
        <>
            {isVisible && (
                <div className="fixed right-4 top-4 w-96 h-96 z-[1000] overflow-y-scroll scrollbar-thin text-xs px-2 bg-sky-50 border border-white/10 rounded-md">
                    <div className="flex justify-end absolute top-0 right-0">
                        <Button
                            onClick={toggleVisibility}
                            size="icon"
                            variant="destructive"
                            className="text-xs p-1 h-min w-min rounded-full -translate-x-2 translate-y-2"
                        >
                            ✖
                        </Button>
                    </div>
                    <SyntaxHighlighter
                        customStyle={{ background: "transparent" }}
                        language="json"
                        style={coldarkCold}
                    >
                        {formattedJSON}
                    </SyntaxHighlighter>
                </div>
            )}
        </>
    );

    return (
        <>
            <Button className="text-xs p-2 h-min w-min" variant="ghost" size="sm" onClick={toggleVisibility}>
                <FileJson size={16} />
            </Button>
            {createPortal(portalContent, portalRoot)}
        </>
    );
}
