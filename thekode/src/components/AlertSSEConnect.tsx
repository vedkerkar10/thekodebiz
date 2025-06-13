"use client";

import React, { useEffect, useState } from "react";
import { useFileStore } from "./Zustand";
import { toast } from "sonner";

import {
  AlertDialog,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";

interface AlertMessage {
  message: string;
  type: "DIALOG" | "TOAST" | "RefetchOutlierAnalysisData";
  id: string;
}

export function AlertSSEConnect() {
  const { server, toggleRefetchOutlierAnalysisData } = useFileStore();
  const [alerts, setAlerts] = useState<AlertMessage[]>([]);
  const MAX_RETRIES = 3;
  const RETRY_DELAY = 2000; // 2 seconds

  useEffect(() => {
    let retryCount = 0;
    let eventSource: EventSource;

    const connectSSE = () => {
      eventSource = new EventSource(`${server}/events`);

      eventSource.onopen = () => {
        console.log("Connected to SSE");
        retryCount = 0; // Reset retry count on successful connection
      };

      eventSource.onerror = () => {
        eventSource.close();

        if (retryCount < MAX_RETRIES) {
          retryCount++;
          console.log(`Retrying connection... Attempt ${retryCount}`);
          setTimeout(connectSSE, RETRY_DELAY);
        } else {
          toast.error(
            "Failed to connect to popup socket after multiple attempts"
          );
        }
      };

      eventSource.onmessage = (event) => {
        const data: Omit<AlertMessage, "id"> = JSON.parse(event.data);
        if (data.type === "DIALOG") {
          setAlerts((prev) => [
            ...prev,
            { ...data, id: Date.now().toString() },
          ]);
        }
        if (data.type === "TOAST") {
          toast(data.message);
        }
        if (data.type === "RefetchOutlierAnalysisData") {
          toggleRefetchOutlierAnalysisData();
        }
      };
    };

    connectSSE();

    return () => {
      eventSource?.close();
    };
  }, [server, toggleRefetchOutlierAnalysisData]);

  const closeAlert = (id: string) => {
    setAlerts(prev => prev.filter(alert => alert.id !== id));
  };

  return (
    <>
      {alerts.map((alert) => (
        <AlertDialog key={alert.id} open={true} onOpenChange={() => closeAlert(alert.id)}>
          <AlertDialogContent>
            <AlertDialogHeader>
              <AlertDialogTitle>Update</AlertDialogTitle>
              <AlertDialogDescription>
                <pre className="text-wrap font-sans">{alert.message}</pre>
              </AlertDialogDescription>
            </AlertDialogHeader>
            <AlertDialogFooter>
              <AlertDialogCancel onClick={() => closeAlert(alert.id)}>
                Close
              </AlertDialogCancel>
            </AlertDialogFooter>
          </AlertDialogContent>
        </AlertDialog>
      ))}
    </>
  );
}
