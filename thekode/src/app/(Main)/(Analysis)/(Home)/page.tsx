"use client";

import SideOptions from "@/components/sideOptions";
import { useFileStore } from "@/components/Zustand";
import { useEffect } from "react";

import SideFlap from "@/components/sideFlap";

import { Tables } from "@/components/Tables";

import { useSearchParams } from "next/navigation";
import { Button } from "@/components/ui/button";
import { Activity, Clock } from "lucide-react";
import { formatSmallTime } from "@/lib/time";
import { HyperTuningSidebar } from "@/components/hyperTuningSidebar";
import DataAnalysisSideFlap from "@/components/ChartingDataAnalysisMode";
import OutlierAnalysisSideFlap from "@/components/ChartingOutlierAnalaysisMode";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";

import { Chart } from "@/components/DAChart";
import { OutlierChart } from "@/components/outlier_analysis";
import { AlertSSEConnect } from "@/components/AlertSSEConnect";
import { JsonView } from "@/components/jsonView";
export default function Page() {
  const searchParams = useSearchParams();
  const {
    collapsed,
    dataAnalysisStage,
    OutlierAnalyisisStage,
    setSessionID,
    stage,
    hyperTuningEstimates,
    dataAnalysisData,
    outlierAnalysisData,
  } = useFileStore();
  useEffect(() => {
    const sessionID = searchParams.get("SessionID");
    if (sessionID !== null) {
      setSessionID(sessionID);
    }
  }, [searchParams, setSessionID]);

  return (
    <main className=" h-full overflow-hidden px-4 pt-4 ">
      <div className="" id="json-debug-portal-root" />
      <AlertSSEConnect />
      <div className="fixed top-[5.6rem] left-72 z-50">
        <SideFlap />
      </div>

      <div className="fixed top-[5.6rem] left-72 z-50">
        <div className="" id="portal-root" />
      </div>

      <div className="fixed top-[5.6rem] left-72 z-50">
        <DataAnalysisSideFlap />
      </div>

      <div className="fixed top-[5.6rem] left-72 z-50">
        <OutlierAnalysisSideFlap />
      </div>

      <div className="fixed z-30 ">
        <SideOptions />
      </div>

      <div className="relative  z-10">
        {hyperTuningEstimates !== null && <HyperTuningSidebar />}
      </div>
      {dataAnalysisStage >= 1 && stage > 2 ? (
        <div
          style={{
            marginLeft: "calc(32.5rem + 8px)",
            width: "calc(100% - 32.5rem - 8px)",
            // top: "5.6rem",
            // dataAnalysisStage < 1
            // 	? "20px"
            // 	: stage === 4
            // 		? "calc(32.5rem + 8px)"
            // 		: "calc(16rem + 8px)",

            paddingLeft: "",
          }}
          className=" relative  w-full h-full"
        >
          <Chart data={dataAnalysisData} />
        </div>
      ) : OutlierAnalyisisStage >= 1 && stage > 2 ? (
        <div
          style={{
            marginLeft: "calc(16rem + 8px)",
            width: "calc(100% - 16rem - 8px)",
            // top: "5.6rem",
            // dataAnalysisStage < 1
            // 	? "20px"
            // 	: stage === 4
            // 		? "calc(32.5rem + 8px)"
            // 		: "calc(16rem + 8px)",

            paddingLeft: "",
          }}
          className=" relative  w-full h-full"
        >
          <OutlierChart data={outlierAnalysisData} />
        </div>
      ) : (
        <div
          style={{
            marginLeft: collapsed
              ? "20px"
              : stage === 4
                ? "calc(32.5rem + 8px)"
                : "calc(16rem + 8px)",

            paddingLeft: "",
          }}
          className=" transition-all duration-500   "
        >
          <Tables />
        </div>
      )}
    </main>
  );
}
