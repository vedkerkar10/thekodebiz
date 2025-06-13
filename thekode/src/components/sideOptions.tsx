"use client";

import { useFileStore } from "@/components/Zustand";
import { FileSelector } from "./fileSelector";
import { AlgorithmSelector } from "./algorithmSelector";
import { AlgorithmOptions } from "./algorithmOptions";
import { ChevronsRight, Dot } from "lucide-react";
import SessionDisplay from "./SessionDisplay";

import PredictionOptions from "./predictionOptions";
import EndOptions from "./EndOptions";
import HyperParameterOptions from "./hyperParameterOptions";
import DynamicSideBar, { SidebarProvider } from "./DynamicSideBar";
import axios from "axios";
import { useEffect, useState } from "react";
import { Button } from "./ui/button";
import { cn } from "@/lib/utils";

type Parameter =
  | {
      id: string;
      type: "number";
      range: { min: number; max: number };
      default?: string;
      step?: number;
    }
  | {
      id: string;
      type: "string";
      range?: string[] | string;
      default?: number;
    };

type FunctionConfig = {
  name: string;
  parameters: Parameter[];
};

type PreprocessStep = {
  dropdown: boolean;
  feature_names: string[];
  functions: FunctionConfig[];
};

export type PreprocessConfig = {
  preprocess_config: {
    [key: string]: PreprocessStep;
  };
};

export type DynamicSidebarConfigType = {
  name: string;
  sidebarName: string;
  buttonText: string;
  modes: string[];
};

function SideOptions() {
  const {
    collapsed,
    setCollapsed,
    selectedFeatures,
    algorithm,
    loading,
    server,
    handleToggleDataAnalysisMode,
    handleToggleOutlierAnalysisMode,
    dataAnalysisStage,
    stage,
    OutlierAnalyisisStage,
  } = useFileStore();
  const [PreprocessConfig, setPreprocessConfig] =
    useState<PreprocessConfig | null>(null);

  const [DynamicSidebarConfig, setDynamicSidebarConfig] = useState<
    DynamicSidebarConfigType[] | null
  >(null);

  useEffect(() => {
    if (!algorithm || !selectedFeatures) return;
    const fd = new FormData();

    fd.append("algorithm", algorithm);
    fd.append("features", JSON.stringify(selectedFeatures));
    axios.post(`${server}/PreProcessConfig`, fd).then((res) => {
      console.log(JSON.stringify(res.data));
      setPreprocessConfig(res.data);
      //   setAlgosJSON(res.data.output_format);
    });
  }, [algorithm, selectedFeatures, server]);

  useEffect(() => {
    if (!algorithm || !selectedFeatures) return;
    const fd = new FormData();

    fd.append("algorithm", algorithm);
    fd.append("features", JSON.stringify(selectedFeatures));
    axios.post(`${server}/DynamicSidebarConfig`, fd).then((res) => {
      setDynamicSidebarConfig(res.data.data);
    });
  }, [algorithm, selectedFeatures, server]);

  return (
    <SidebarProvider>
      <div
        className=" scrollbar-thumb-neutral-200/50 scrollbar-thumb-rounded-full  scrollbar-track-transparent scrollbar-thin  w-64 relative transition-all duration-500  max-h-[calc(100vh-70px-16px)] overflow-y-scroll pb-4 pr-0.5"
        style={{
          translate: collapsed ? "-240px 0px" : "0px 0px",
        }}
      >
        <button
          type="button"
          onClick={() => setCollapsed(!collapsed)}
          className="p-2 absolute top-0 right-0 z-50 cursor-pointer group"
        >
          <ChevronsRight
            style={{
              rotate: collapsed ? "180deg" : "0deg",
            }}
            className=" h-5 group-hover:rotate-180  transition-all group-hover:text-indigo-400"
          />
        </button>
        {/* <SessionDisplay /> */}
        <FileSelector />

        <AlgorithmSelector />
        <AlgorithmOptions />

        {stage >= 3 && (
          <div className="bg-white/10 p-2 rounded-lg border border-white/10 my-2 space-y-3">
            <Button
              disabled={loading}
              onClick={handleToggleDataAnalysisMode}
              className={cn("mt-2 mb-1 w-full transition-all duration-300")}
            >
              Data Analysis {dataAnalysisStage ? <Dot /> : <></>}
            </Button>
            <Button
              disabled={loading}
              onClick={handleToggleOutlierAnalysisMode}
              className={cn("mt-2 mb-1 w-full transition-all duration-300")}
            >
              Outlier Analysis {OutlierAnalyisisStage ? <Dot /> : <></>}
            </Button>

            {DynamicSidebarConfig?.map((conf) => (
              <DynamicSideBar
                key={conf.name}
                config={PreprocessConfig}
                name={conf.name}
                buttonText={conf.buttonText}
                sidebarTitle={conf.sidebarName}
                modes={conf.modes}
              />
            ))}
          </div>
        )}

        <EndOptions />
        <PredictionOptions />

        <HyperParameterOptions />
        <div className="h-12" />
      </div>
    </SidebarProvider>
  );
}

export default SideOptions;
