"use client";

import React, {
  createContext,
  useContext,
  useEffect,
  useState,
  useCallback,
  useMemo,
  type ReactNode,
  useDebugValue,
} from "react";
import ReactDOM from "react-dom";
import { Button } from "./ui/button";
import { Label } from "./ui/label";
import { Input } from "./ui/input";
import { Slider } from "./ui/slider";
import { MultiSelect } from "./multiselect";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "./ui/select";
import { cn } from "@/lib/utils";
import type { PreprocessConfig } from "./sideOptions";
import { useAlgoJSONStore } from "./jsonView";
import axios from "axios";
import { useFileStore } from "./Zustand";
import { Bug, Dot } from "lucide-react";
import { create } from "zustand";
import { useTableTabsProps } from "./Tables";
import { parsePythonDataFrame } from "@/lib/xlsx_preprocess";

interface DynamicSidebarStore {
  sidebarPropsReset: boolean;
  setResetSidebarProps: (value: boolean) => void;
}

export const useDynamicSidebarStore = create<DynamicSidebarStore>(
  (set, get) => ({
    sidebarPropsReset: false,
    setResetSidebarProps: (value) => set({ sidebarPropsReset: value }),
  }),
);

// BOOLEAN
// STRING
// HANDLE CHANGE
// Flap should go away when clicked run preprocess
// proces corr should work after perofimng preprocessing and sampling
//
interface DynamicSidebarProps {
  modes: string[];
  name: string;
  buttonText: string;
  sidebarTitle: string;
  config: PreprocessConfig | null;
}

interface SidebarContextType {
  openSidebar: string | null;
  setOpenSidebar: (id: string | null) => void;
}

const SidebarContext = createContext<SidebarContextType | undefined>(undefined);

export function SidebarProvider({ children }: { children: ReactNode }) {
  const [openSidebar, setOpenSidebar] = useState<string | null>(null);
  return (
    <SidebarContext.Provider value={{ openSidebar, setOpenSidebar }}>
      {children}
    </SidebarContext.Provider>
  );
}

function useSidebar() {
  const context = useContext(SidebarContext);
  if (!context) {
    throw new Error("useSidebar must be used within a SidebarProvider");
  }
  return context;
}

function NumericParameterInput({
  parameter,
  onChange,
}: {
  parameter: PreprocessConfig["preprocess_config"][""]["functions"][0]["parameters"][0];
  onChange: (value: number) => void;
}) {
  const { sidebarPropsReset, setResetSidebarProps } = useDynamicSidebarStore();

  const [value, setValue] = useState<number>(
    typeof parameter.default === "number"
      ? parameter.default
      : Number(parameter.default) || 0,
  );

  // const updateValue = (newValue: number) => {
  //   setValue(newValue);
  //   onChange(newValue);
  // };

  const handleChange = useCallback(
    (newValue: number) => {
      setValue(newValue);
      onChange(newValue);
    },
    [setValue, onChange],
  );
  useEffect(() => {
    if (sidebarPropsReset) {
      handleChange(
        typeof parameter.default === "number"
          ? parameter.default
          : Number(parameter.default) || 0,
      );
      setResetSidebarProps(false);
    }
  }, [
    sidebarPropsReset,
    handleChange,
    setResetSidebarProps,
    parameter.default,
  ]);

  if (parameter.type !== "number") return null;

  return (
    <div className="flex flex-col gap-2 py-1">
      <div className="flex justify-between items-center">
        <Label className="text-sm font-medium text-gray-700">
          {parameter.id}
        </Label>
        <Input
          type="number"
          className="bg-transparent text-end border-none focus:ring-0 rounded-md text-sm"
          value={value}
          onChange={(e) => handleChange(Number(e.target.value))}
        />
      </div>
      <div className="relative">
        <Slider
          minStepsBetweenThumbs={10000}
          max={parameter.range.max}
          className="w-full"
          value={[value]}
          onValueChange={(vals) => handleChange(vals[0])}
          step={parameter.step ? parameter.step : 0.1}
        />
      </div>
    </div>
  );
}

function StringParameterInput({
  parameter,
  onChange,
}: {
  parameter: PreprocessConfig["preprocess_config"][""]["functions"][0]["parameters"][0];
  onChange: (value: string) => void;
}) {
  const { sidebarPropsReset, setResetSidebarProps } = useDynamicSidebarStore();

  const [value, setValue] = useState<string>(
    typeof parameter.default === "string"
      ? parameter.default
      : String(parameter.default),
  );
  const handleChange = useCallback(
    (newValue: string) => {
      setValue(newValue);
      onChange(newValue);
    },
    [setValue, onChange],
  );
  useEffect(() => {
    if (sidebarPropsReset) {
      handleChange("");
      setResetSidebarProps(false);
    }
  }, [sidebarPropsReset, handleChange, setResetSidebarProps]);

  if (parameter.type !== "string") return null;
  return (
    <div className="flex flex-col gap-2 py-1">
      <div className="flex justify-between items-center">
        <Label className="text-sm font-medium text-gray-700">
          {parameter.id}
        </Label>
        <Select value={value} onValueChange={handleChange}>
          <SelectTrigger className="w-full max-w-32">
            <SelectValue placeholder="Select value" />
          </SelectTrigger>
          <SelectContent>
            {Array.isArray(parameter.range) ? (
              parameter.range.map((option: string) => (
                <SelectItem key={option} value={option}>
                  {option}
                </SelectItem>
              ))
            ) : parameter.range ? (
              <SelectItem key={parameter.range} value={parameter.range}>
                {parameter.range}
              </SelectItem>
            ) : null}
          </SelectContent>
        </Select>
      </div>
    </div>
  );
}

function FeatureAlgorithm({
  feature,
  category,
  config,
  onAlgorithmChange,
}: {
  feature: string;
  category: string;
  config?: PreprocessConfig;
  onAlgorithmChange: (
    feature: string,
    algoConfig: { function_name: string; parameters: Record<string, any> },
  ) => void;
}) {
  const { sidebarPropsReset, setResetSidebarProps } = useDynamicSidebarStore();

  const [selectedAlgorithm, setSelectedAlgorithm] = useState<string>("");
  const [algorithmParams, setAlgorithmParams] = useState<Record<string, any>>(
    {},
  );

  const algorithms = useMemo(
    () => config?.preprocess_config[category]?.functions || [],
    [config, category],
  );

  // Pre-fill parameters with defaults when algorithm selection changes.
  useEffect(() => {
    if (selectedAlgorithm) {
      const algo = algorithms.find((a) => a.name === selectedAlgorithm);
      if (algo) {
        const defaultParams = algo.parameters.reduce(
          (acc, param) => {
            acc[param.id] = param.default !== undefined ? param.default : 0;
            return acc;
          },
          {} as Record<string, any>,
        );
        setAlgorithmParams(defaultParams);
      }
    }
  }, [selectedAlgorithm, algorithms]);

  // Report changes upward when algorithm or parameters change.
  useEffect(() => {
    if (selectedAlgorithm) {
      onAlgorithmChange(feature, {
        function_name: selectedAlgorithm,
        parameters: algorithmParams,
      });
    }
  }, [selectedAlgorithm, algorithmParams, feature, onAlgorithmChange]);

  useEffect(() => {
    if (sidebarPropsReset) {
      setSelectedAlgorithm("");
      setResetSidebarProps(false);
    }
  }, [sidebarPropsReset, setSelectedAlgorithm, setResetSidebarProps]);
  return (
    <div>
      <div className="flex items-center justify-between">
        <Label>{feature}</Label>
        <Select value={selectedAlgorithm} onValueChange={setSelectedAlgorithm}>
          <SelectTrigger className="w-1/2 h-8 mt-1 border-white/10">
            <SelectValue placeholder="Select Algorithm" />
          </SelectTrigger>
          <SelectContent className="border-white/10 backdrop-blur-md">
            {algorithms.map((algo) => (
              <SelectItem key={algo.name} value={algo.name}>
                {algo.name}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>
      {selectedAlgorithm &&
        algorithms
          .find((algo) => algo.name === selectedAlgorithm)
          ?.parameters.map((param) => (
            <div key={param.id} className="mb-2">
              {param.type === "number" ? (
                <NumericParameterInput
                  parameter={param}
                  onChange={(value) =>
                    setAlgorithmParams((prev) => ({
                      ...prev,
                      [param.id]: value,
                    }))
                  }
                />
              ) : param.type === "string" ? (
                <StringParameterInput
                  parameter={param}
                  onChange={(value) =>
                    setAlgorithmParams((prev) => ({
                      ...prev,
                      [param.id]: value,
                    }))
                  }
                />
              ) : (
                <></>
              )}
            </div>
          ))}
    </div>
  );
}

function FeatureBlock({
  category,
  config,
  onBlockConfigChange,
}: {
  category: string;
  config: PreprocessConfig;
  onBlockConfigChange: (category: string, blockConfig: any) => void;
}) {
  const [selectedFeatures, setSelectedFeatures] = useState<string[]>([]);
  const [featuresConfig, setFeaturesConfig] = useState<Record<string, any>>({});

  const handleAlgorithmChange = useCallback(
    (
      feature: string,
      algoConfig: { function_name: string; parameters: Record<string, any> },
    ) => {
      setFeaturesConfig((prev) => {
        const newConfig = { ...prev, [feature]: algoConfig };
        // Update block config whenever algorithm changes
        const blockConfig = selectedFeatures
          .map((feat) => {
            const cfg = newConfig[feat];
            if (!cfg) return null;
            // return category === "feature_selection"
            //   ? [cfg.function_name, cfg.parameters]
            //   : { feature_names: [feat], ...cfg };\
            return { feature_names: [feat], ...cfg };
          })
          .filter(Boolean);
        onBlockConfigChange(category, blockConfig);
        return newConfig;
      });
    },
    [category, selectedFeatures, onBlockConfigChange],
  );

  useEffect(() => {
    if (!config.preprocess_config[category].dropdown) {
      // if (config.preprocess_config[category].feature_names.length <) {
      setSelectedFeatures([
        config.preprocess_config[category].feature_names[0],
      ]);
      // }
    }
  }, [category, config]);
  // Update block config when selected features change
  useEffect(() => {
    const blockConfig = selectedFeatures
      .map((feature) => {
        const algoConfig = featuresConfig[feature];
        if (!algoConfig) return null;
        // return category === "feature_selection"
        //   ? [algoConfig.function_name, algoConfig.parameters]
        //   : { feature_names: [feature], ...algoConfig };
        return { feature_names: [feature], ...algoConfig };
      })
      .filter(Boolean);
    onBlockConfigChange(category, blockConfig);
  }, [selectedFeatures, category, onBlockConfigChange, featuresConfig]);

  const featureOptions = config.preprocess_config[category].feature_names.map(
    (feat) => ({
      label: feat,
      value: feat,
    }),
  );

  return (
    <div>
      <Label className="capitalize text-sm mb-1">
        {category.replace("_", " ")}
      </Label>
      {config.preprocess_config[category].dropdown ? (
        <MultiSelect
          options={featureOptions}
          selected={selectedFeatures}
          onChange={setSelectedFeatures}
          placeholder="Select Features"
        />
      ) : (
        <div className="text-xs flex gap-1">
          {featureOptions.map((x) => (
            <p className="text-sky-700/80" key={x.value}>
              {x.label}
            </p>
          ))}
        </div>
      )}

      {selectedFeatures.map((feature) => (
        <FeatureAlgorithm
          key={feature}
          feature={feature}
          category={category}
          config={config}
          onAlgorithmChange={handleAlgorithmChange}
        />
      ))}
    </div>
  );
}

function SidebarPortal({
  open,
  children,
}: {
  open: boolean;
  children: ReactNode;
}) {
  const portalRoot =
    typeof window !== "undefined" && document.getElementById("portal-root");
  if (!portalRoot) return null;
  return ReactDOM.createPortal(
    <div
      className={cn(
        open
          ? "bg-red-100 opacity-100 left-0 blur-0"
          : "bg-gray-100 opacity-0 pointer-events-none -left-24 blur",
        "my-2 border bg-white/10 backdrop-blur-xl border-white/10 absolute overflow-y-scroll -top-2 transition-all max-h-[calc(100vh-12rem)] scrollbar-thin duration-300 -mx-2 rounded-md  w-72 p-2",
      )}
    >
      {children}
    </div>,
    portalRoot,
  );
}

export default function DynamicSidebar({
  name,
  buttonText,
  modes,
  config,
  sidebarTitle,
}: DynamicSidebarProps) {
  const { openSidebar, setOpenSidebar } = useSidebar();
  const { setAlgosJSON, algosJSON } = useAlgoJSONStore();

  const [globalConfig, setGlobalConfig] = useState<Record<string, any>>({});
  const {
    server,
    algorithm,
    target,
    selectedFeatures,
    setLoading,
    loading,
    setDataAnalysisStage,
    dataAnalysisStage,
  } = useFileStore();

  const { setResetSidebarProps } = useDynamicSidebarStore();
  const { addTable } = useTableTabsProps();
  const toggleSidebar = () => {
    console.log("toggled");
    setResetSidebarProps(true);
    setAlgosJSON({});
    setDataAnalysisStage(0);
    setOpenSidebar(openSidebar === name ? null : name);
  };
  useEffect(() => {
    if (dataAnalysisStage !== 0) {
      setOpenSidebar(null);
    }
  }, [dataAnalysisStage, setOpenSidebar]);
  const handleBlockConfigChange = useCallback(
    (category: string, blockConfig: any) => {
      setGlobalConfig((prev) => {
        const newConfig = { [category]: blockConfig };
        // Filter empty arrays here instead of in a separate effect
        const filteredConfig = Object.fromEntries(
          Object.entries(newConfig).filter(
            ([, value]) => Array.isArray(value) && value.length > 0,
          ),
        );
        // Update algosJSON store directly here
        setAlgosJSON(filteredConfig);
        return newConfig;
      });
    },
    [setAlgosJSON],
  );

  // Initialize globalConfig with algosJSON when it changes externally
  useEffect(() => {
    if (algosJSON && Object.keys(globalConfig).length === 0) {
      setGlobalConfig(algosJSON);
    }
  }, [algosJSON, globalConfig]);

  if (selectedFeatures.length === 0) return null;
  if (!config) return null;

  const isOpen = openSidebar === name;
  function handlePreprocess(event: React.MouseEvent<HTMLButtonElement>): void {
    event.preventDefault();
    if (!server || !algorithm || !algosJSON || !target || !selectedFeatures) {
      return;
    }
    const formData = new FormData();
    setLoading(true);
    formData.append("algorithm", algorithm);
    formData.append("pre_process", JSON.stringify(algosJSON));
    formData.append("target", target);
    formData.append("features", JSON.stringify(selectedFeatures));

    axios
      .post(`${server}/PreProcess`, formData)
      .then((d) => {
        if (d.data.has_table) {
          addTable({
            name: d.data.table_name,
            sheet: parsePythonDataFrame(d.data.table as unknown as any) as any,
          });
        }
        setOpenSidebar(null);
        setLoading(false);
      })
      .catch((error) => {
        console.log(error);
        setLoading(false);
      });
  }

  return (
    <div>
      <Button className="w-full" onClick={toggleSidebar}>
        {buttonText} {openSidebar === name ? <Dot /> : <></>}
      </Button>
      <SidebarPortal open={isOpen}>
        <div className="flex flex-col gap-2 p-1">
          {modes.map((mode) => (
            <FeatureBlock
              key={mode}
              category={mode}
              config={config}
              onBlockConfigChange={handleBlockConfigChange}
            />
          ))}
        </div>
        <Button
          disabled={loading}
          className="w-full mt-4"
          onClick={handlePreprocess}
        >
          Go
        </Button>
      </SidebarPortal>
    </div>
  );
}
