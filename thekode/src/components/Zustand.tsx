import { create } from "zustand";
import type { CombinedState } from "./types";
import { produce } from "immer";

// Create a Zustand store for managing global state
export const useFileStore = create<CombinedState>((set, get) => ({
  includedColsDfDisplay: [],
  setIncludedColsDfDisplay: (cols) => set({ includedColsDfDisplay: cols }),
  server: "http://localhost:5000/",
  setServer: (URL) => set({ server: URL }),

  epochs: 20,
  setEpochs: (epoch) => set({ epochs: epoch }),
  analysisGraphs: null,
  setAnalysisGraphs: (graphs) => set({ analysisGraphs: graphs }),
  predictionGraph: null,
  setPredictionGraph: (graphs) => set({ predictionGraph: graphs }),
  trainingFile: null,
  setTrainingFile: (newFile) => set({ trainingFile: newFile }),
  hyperParameterFile: null,
  setHyperParameterFile: (newFile) => set({ hyperParameterFile: newFile }),
  predictionFile: null,
  setPredictionFile: (newFile) => set({ predictionFile: newFile }),
  loading: false,
  setLoading: (isLoading) => set({ loading: isLoading }),
  loadingPercent: false,
  setLoadingPercent: (amount) => set({ loadingPercent: amount }),
  algorithm: null,
  setAlgorithm: (newAlgorithm) => set({ algorithm: newAlgorithm }),
  processed: false,
  setProcessed: (isProcessed) => set({ processed: isProcessed }),
  stage: 1,
  setStage: (newStage) => set({ stage: newStage }),
  targets: ["n/a"],
  setTargets: (newTargets) => set({ targets: newTargets }),
  aggregates: [],
  setAggregates: (newAggregates) => set({ aggregates: newAggregates }),
  target: null,
  setTarget: (newTarget) => set({ target: newTarget }),
  selectedFeatures: [],
  setSelectedFeatures: (newFeatures) => set({ selectedFeatures: newFeatures }),
  aggregator: null,
  setAggregator: (newAggregator) => set({ aggregator: newAggregator }),
  frequency: null,
  setFrequency: (newFrequency) => set({ frequency: newFrequency }),
  granularity: null,
  setGranularity: (newGanularity) => set({ granularity: newGanularity }),
  percentageAccuracy: null,
  setPercentageAccuracy: (newPercent) =>
    set({ percentageAccuracy: newPercent }),
  collapsed: false,
  setCollapsed: (isCollapsed) => set({ collapsed: isCollapsed }),
  currentSheet: "Original",
  setCurrentSheet: (newSheet) => set({ currentSheet: newSheet }),
  originalSheet: null,
  setOriginalSheet: (sheet) => set({ originalSheet: sheet }),
  correlation1Sheet: null,
  setCorrelation1Sheet: (sheet) => set({ correlation1Sheet: sheet }),
  correlation2Sheet: null,
  setCorrelation2Sheet: (sheet) => set({ correlation2Sheet: sheet }),
  analysisSheet: null,
  setAnalysisSheet: (sheet) => set({ analysisSheet: sheet }),

  hyperSheet: null,
  setHyperSheet: (sheet) => set({ hyperSheet: sheet }),
  bestParamsSheet: null,
  setBestParamsSheet: (sheet) => set({ bestParamsSheet: sheet }),
  confusionMatrixSheet: null,
  setConfusionMatrixSheet: (sheet) => set({ confusionMatrixSheet: sheet }),
  classificationReportSheet: null,
  setClassificationReportSheet: (sheet) =>
    set({ classificationReportSheet: sheet }),

  setAnalysisMessage: (message) => set({ analysisMessage: message }),
  analysisMessage: null,
  modelAnalysisSheets: null,
  setModelAnalysisSheets: (sheets) => set({ modelAnalysisSheets: sheets }),

  predictionSheet: null,
  setPredictionSheet: (sheet) => set({ predictionSheet: sheet }),

  endMode: null,
  setEndMode: (mode) => set({ endMode: mode }),
  hyperTuningEstimates: null,
  sethyperTuningEstimates: (estimate) =>
    set({ hyperTuningEstimates: estimate }),
  resetHyperTuningEstimates: () => set({ hyperTuningEstimates: null }),
  chosenHyperTuningAlgo: null,
  setChosenHyperTuningAlgo: (algo) => set({ chosenHyperTuningAlgo: algo }),

  algorithmPos: null,
  setAlgorithmPos: (algoPos) => set({ algorithmPos: algoPos }),

  sessionID: null,
  setSessionID: (id) => set({ sessionID: id }),
  uniqueValues: {},
  setUniqueValues: (newUniqueValues) =>
    set({
      uniqueValues: newUniqueValues,
      // selectedUniqueValues: newUniqueValues,
    }),
  selectedUniqueValues: {},
  dataAnalysisData: null,
  outlierAnalysisData: null,
  setDataAnalysisData: (data) => set({ dataAnalysisData: data }),
  setOutlierAnalysisData: (data) => set({ outlierAnalysisData: data }),
  refetchOutlierAnalysisData: true,
  toggleRefetchOutlierAnalysisData: () => {
    if (get().refetchOutlierAnalysisData)
      return set({
        refetchOutlierAnalysisData: false,
      });

    return set({
      refetchOutlierAnalysisData: true,
    });
  },
  dataAnalysisStage: 0,
  OutlierAnalyisisStage: 0,

  handleInitDataAnalysis: () => {
    set({
      dataAnalysisStage: 1,
    });
  },
  setDataAnalysisStage: (mode: number) => {
    return set({
      dataAnalysisStage: mode,
    });
  },
  handleToggleDataAnalysisMode: () => {
    if (get().dataAnalysisStage >= 1)
      return set({
        dataAnalysisStage: 0,

        OutlierAnalyisisStage: 0,
      });
    if (get().dataAnalysisStage === 0)
      return set({
        dataAnalysisStage: 1,

        OutlierAnalyisisStage: 0,
      });
  },

  handleToggleOutlierAnalysisMode: () => {
    if (get().OutlierAnalyisisStage >= 1)
      return set({
        dataAnalysisStage: 0,

        OutlierAnalyisisStage: 0,
      });
    if (get().OutlierAnalyisisStage === 0)
      return set({
        dataAnalysisStage: 0,
        OutlierAnalyisisStage: 1,
      });
  },
  handleStartDataAnalysisMode: () => {
    set({
      dataAnalysisStage: 2,
    });
  },
  handleEndDataAnalysisMode: () => {
    set({
      dataAnalysisStage: 0,
    });
  },
  dataAnalysisXAxis: null,
  setDataAnalysisXAxis: (value) => {
    set({ dataAnalysisXAxis: value });
  },
  dataAnalysisYAxis: null,
  setDataAnalysisYAxis: (value) => {
    set({ dataAnalysisYAxis: value });
  },
  setSelectedUniqueValues: (key, value) =>
    set((state) => {
      const updatedValues = Object.keys(state.selectedUniqueValues).includes(
        key,
      )
        ? Array.from(new Set([...state.selectedUniqueValues[key], value]))
        : [value];

      return {
        selectedUniqueValues: {
          ...state.selectedUniqueValues,
          [key]: updatedValues,
        },
      };
    }),
  removeSelectedUniqueValues: (key: string, value: any) =>
    set((state) => {
      if (Object.keys(state.selectedUniqueValues).includes(key)) {
        const updatedValues = state.selectedUniqueValues[key].filter(
          (item: any) => item !== value,
        );
        return {
          selectedUniqueValues: {
            ...state.selectedUniqueValues,
            [key]: updatedValues,
          },
        };
      }
      return state; // Key doesn't exist, return current state unchanged
    }),
}));
