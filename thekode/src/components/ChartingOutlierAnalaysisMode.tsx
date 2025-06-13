import axios from "axios";
import { useFileStore } from "@/components/Zustand";
import { useEffect } from "react";
import React from "react";

export default function OutlierAnalysisSideFlap() {
  const {
    setLoading,
    server,
    setOutlierAnalysisData,
    selectedFeatures,
    target,
    refetchOutlierAnalysisData,
  } = useFileStore();

  useEffect(() => {
    if (refetchOutlierAnalysisData) {
    }
    try {
      setLoading(true);
      const formData = new FormData();
      if (!target || !selectedFeatures) {
        setLoading(false);

        return;
      }
      formData.append("target", target);
      formData.append("features", JSON.stringify(selectedFeatures));
      axios.post(`${server}/Outlier_Analysis`, formData).then((r) => {
        console.log(r.data);
        setOutlierAnalysisData(r.data);
      });
      setLoading(false);
    } catch (error) {
      setOutlierAnalysisData(null);
      console.error("Error processing file:", error);
      setLoading(false);
    }
  }, [
    selectedFeatures,
    server,
    setLoading,
    setOutlierAnalysisData,
    target,
    refetchOutlierAnalysisData,
  ]);

  return <></>;
}
