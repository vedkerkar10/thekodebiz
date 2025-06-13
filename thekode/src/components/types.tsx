import { ChartDataPoint } from "./DAChart";

export interface row {
  [key: string]: any;
}

export interface header_type {
  field: string;
  type:
    | "string"
    | "number"
    | "bigint"
    | "boolean"
    | "symbol"
    | "undefined"
    | "object"
    | "function";
  headerName: string;
  width: number;
}

// Define types for file, loading, and algorithm state
interface FileState {
  trainingFile: File | null;
  setTrainingFile: (newFile: File | null) => void;
  predictionFile: File | null;
  setPredictionFile: (newFile: File | null) => void;

  hyperParameterFile: File | null;
  setHyperParameterFile: (newFile: File | null) => void;
}

interface LoadingState {
  loading: boolean;
  setLoading: (isLoading: boolean) => void;
  loadingPercent: boolean;
  setLoadingPercent: (amount: boolean) => void;
}

interface AlgorithmState {
  algorithm: "Classification" | "Regression" | "Time Series" | null;
  setAlgorithm: (newAlgorithm: AlgorithmState["algorithm"]) => void;
}
interface ProcessedState {
  processed: boolean;
  setProcessed: (processed: boolean) => void;
}

interface StageState {
  stage: number;
  setStage: (stage: number) => void;
}
interface targetState {
  targets: string[];
  setTargets: (targets: string[]) => void;
  aggregates: string[] | [];
  setAggregates: (aggregates: string[] | []) => void;
}

interface AlgorithmOptionStates {
  target: string | null;
  setTarget: (newTarget: string | null) => void;
  selectedFeatures: string[];
  setSelectedFeatures: (newFeatures: string[]) => void;
  aggregator: string[] | null;
  setAggregator: (newAggregator: string[] | null) => void;
  frequency: string | null;
  setFrequency: (newFrequency: string | null) => void;
  granularity: string | null;
  setGranularity: (newGranularity: string | null) => void;
  percentageAccuracy: number | null;
  setPercentageAccuracy: (percentageAccuracy: number | null) => void;
}

interface SideOptionsState {
  collapsed: boolean;
  setCollapsed: (isCollapsed: boolean) => void;
}

// correlation1Sheet: null,
//   setCorrelation1Sheet: (sheet) => set({ correlation1Sheet: sheet }),
//   correlation2Sheet: null,
//   setCorrelation2Sheet: (sheet) => set({ correlation2Sheet: sheet }),
//   analysisSheet: null,
//   setAnalysisSheet: (sheet) => set({ analysisSheet: sheet }),
//   predictionSheet: null,
//   setPredictionSheet: (sheet) => set({ predictionSheet: sheet }),
interface SheetsDataState {
  currentSheet:
    | "Original"
    | "Correlation1"
    | "Correlation2"
    | "Analysis"
    | "Prediction"
    | string;
  setCurrentSheet: (newSheet: SheetsDataState["currentSheet"]) => void;
  originalSheet: {
    headers: header_type[];
    data: { [key: string]: any; id: number }[];
  } | null;
  setOriginalSheet: (originalSheet: SheetsDataState["originalSheet"]) => void;
  correlation1Sheet: {
    headers: header_type[];
    data: { [key: string]: any; id: number }[];
  } | null;
  setCorrelation1Sheet: (
    correlation1Sheet: SheetsDataState["correlation1Sheet"]
  ) => void;
  correlation2Sheet: {
    headers: header_type[];
    data: { [key: string]: any; id: number }[];
  } | null;
  setCorrelation2Sheet: (
    correlation2Sheet: SheetsDataState["correlation2Sheet"]
  ) => void;
  analysisSheet: {
    headers: header_type[];
    data: { [key: string]: any; id: number }[];
  } | null;
  setAnalysisSheet: (analysisSheet: SheetsDataState["analysisSheet"]) => void;

  predictionSheet: {
    headers: header_type[];
    data: { [key: string]: any; id: number }[];
  } | null;
  setPredictionSheet: (
    predictionSheet: SheetsDataState["predictionSheet"]
  ) => void;

  hyperSheet: {
    headers: header_type[];
    data: { [key: string]: any; id: number }[];
  } | null;
  setHyperSheet: (predictionSheet: SheetsDataState["predictionSheet"]) => void;

  bestParamsSheet:
    | {
        headers: header_type[];
        data: { [key: string]: any; id: number }[];
      }
    | null
    | any;
  setBestParamsSheet: (
    bestParamsSheet: SheetsDataState["bestParamsSheet"]
  ) => void;
  confusionMatrixSheet: {
    headers: header_type[];
    data: { [key: string]: any; id: number }[];
  } | null;
  setConfusionMatrixSheet: (
    confusionMatrixSheet: SheetsDataState["confusionMatrixSheet"]
  ) => void;
  classificationReportSheet: {
    headers: header_type[];
    data: { [key: string]: any; id: number }[];
  } | null;
  setClassificationReportSheet: (
    classificationReportSheet: SheetsDataState["classificationReportSheet"]
  ) => void;

  modelAnalysisSheets: {
    headers: header_type[];
    data: { [key: string]: any; id: number }[];
  } | null;
  setModelAnalysisSheets: (
    analysisSheets: SheetsDataState["modelAnalysisSheets"]
  ) => void;
}

interface SessionState {
  sessionID: string | null;
  setSessionID: (id: string) => void;
}

interface UniqueValuesState {
  uniqueValues: {
    [name: string]: Array<string | number>;
  };
  setUniqueValues: (newUniqueValues: UniqueValuesState["uniqueValues"]) => void;
  // selectedUniqueValues: Array<{
  //   value: number;
  //   name: string;
  //   values: Array<string | number>;
  // }>;
  selectedUniqueValues: {
    [key: string]: Array<string | number>;
  };
  setSelectedUniqueValues: (key: string, value: number | string) => void; // Function to set selected unique values
  removeSelectedUniqueValues: (key: string, value: number | string) => void; // Function to set selected unique values

  // addToSelectedUniqueValues: (newUniqueValues: {
  //   name: string;
  //   value: string | number;
  // }) => void;
}
// Combine all state types
export type CombinedState = FileState &
  LoadingState &
  AlgorithmState &
  ProcessedState &
  StageState &
  targetState &
  AlgorithmOptionStates &
  SideOptionsState &
  SheetsDataState &
  SessionState &
  UniqueValuesState & {
    analysisGraphs: null | any;
    setAnalysisGraphs: (graphs: any) => void;
    predictionGraph: null | any;
    setPredictionGraph: (graphs: any) => void;
    epochs: number;
    setEpochs: (epoch: number) => void;
    server: string;
    setServer: (URL: string) => void;
  } & {
    includedColsDfDisplay: string[];
    setIncludedColsDfDisplay: (cols: string[]) => void;
  } & {
    endMode: "predict" | "train&predict" | "tune_hyperparameters" | null;
    setEndMode: (
      mode: "predict" | "train&predict" | "tune_hyperparameters"
    ) => void;
  } & {
    algorithmPos: {
      headers: header_type[];
      data: { [key: string]: any; id: number }[];
    } | null;
    setAlgorithmPos: (algorithmPos: {
      headers: header_type[];
      data: { [key: string]: any; id: number }[];
    }) => void;
  } & {
    setAnalysisMessage: (message: string) => void;
    analysisMessage: null | string;
  } & {
    hyperTuningEstimates: null | {
      no_of_fits: number;
      estimated_time: number;
    };
    sethyperTuningEstimates: (message: {
      no_of_fits: number;
      estimated_time: number;
    }) => void;

    resetHyperTuningEstimates: () => void;
  } & {
    chosenHyperTuningAlgo: null | string;
    setChosenHyperTuningAlgo: (algo: string) => void;
  } & {
    dataAnalysisStage: number;
    OutlierAnalyisisStage: number;
    dataAnalysisXAxis: string | null;
    dataAnalysisYAxis: string | null;
    dataAnalysisData: ChartDataPoint[] | null;
    outlierAnalysisData: ChartDataPoint[][] | null;
    setDataAnalysisStage: (mode: number) => void;
    setDataAnalysisData: (data: ChartDataPoint[] | null) => void;
    setOutlierAnalysisData: (data: ChartDataPoint[][] | null) => void;

    refetchOutlierAnalysisData: boolean;
    toggleRefetchOutlierAnalysisData: () => void;
    setDataAnalysisXAxis: (data: string | null) => void;
    setDataAnalysisYAxis: (data: string | null) => void;
    handleStartDataAnalysisMode: () => void;
    handleEndDataAnalysisMode: () => void;
    handleInitDataAnalysis: () => void;
    handleToggleDataAnalysisMode: () => void;
    handleToggleOutlierAnalysisMode: () => void;
  };
