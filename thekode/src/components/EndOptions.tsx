"use client";
import { parsePythonDataFrame } from "@/lib/xlsx_preprocess";
import { config } from "../../config";
import { Button } from "./ui/button";
import { useFileStore } from "@/components/Zustand";
import { toast } from "sonner";
import { useEffect, useState } from "react";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "./ui/select";
import { Label } from "./ui/label";

export default function EndOptions() {
  const {
			stage,
			setStage,
			target,
			selectedFeatures,
			setCorrelation1Sheet,
			setCorrelation2Sheet,
			setLoading,
			setCurrentSheet,
			setUniqueValues,
			algorithm,
			setLoadingPercent,
			setAnalysisSheet,
			loading,
			correlation1Sheet,
			server,
			setIncludedColsDfDisplay,
			endMode,
			setEndMode,
			handleEndDataAnalysisMode		} = useFileStore();
  const [trainActive, setTrainActive] = useState(false);

  const handleProcessCorrelation = () => {
	handleEndDataAnalysisMode();
    const formData = new FormData();
    formData.append("target", target as string);
    formData.append("features", JSON.stringify(selectedFeatures));
    formData.append("algorithm", JSON.stringify(algorithm));

    fetch(`${server}/SetOptions`, { method: "POST", body: formData })
      .then((r) => r.json())
      .then((d) => {
        setCorrelation1Sheet(parsePythonDataFrame(d.c1) as any);
        setCorrelation2Sheet(parsePythonDataFrame(d.c2) as any);
        setCurrentSheet("Correlation1");
       
     setLoading(false);
        setTrainActive(true);
        toast.success("Successfully Processed Correlation");
      })
      .catch((e) => {
        console.error(e);
        setLoading(false);
      });
  };
  function set_server_data({ attr, value }: { attr: string; value: any }) {
    const formData = new FormData();
    formData.append("value", JSON.stringify(value));
    setLoading(true);
    return fetch(`${server}/SetValue?attr=${attr}`, {
      method: "POST",
      body: formData,
    }).then(() => {
      setLoading(false);
    }).catch((e) => {
      console.error(e);
      setLoading(false);
    });
  }
  const train_model = () => {
    console.log("training");
    setLoading(true);
    setLoadingPercent(true);
    const fd = new FormData();

    fd.append("target", target as string);
    fd.append("selectedFeatures", JSON.stringify(selectedFeatures));
    fd.append("problem_type", algorithm as string);
    fetch(`${server}/Train`, {
      method: "POST",
      body: fd,
    })
      .then((r) => r.json())
      .then((d) => {
        // console.log("model_result", d.model_result);
        setIncludedColsDfDisplay(d.cols_to_display);
        setAnalysisSheet(parsePythonDataFrame(d.model_result) as any);
        setStage(5);
        setCurrentSheet("Analysis");
        toast.success("Training Completed");
        setLoadingPercent(false);
        setLoading(false);
      })
      .catch((e) => {
        console.error(e);
        setLoading(false);
      });
  };

  const handleTrain = () => {
    setTrainActive(false);
    // fetch(`${server}/Get_Unique_Values`)
    //   .then((r) => r.json())
    //   .then((s) => console.log(s));
    if (algorithm !== "Time Series") {
      train_model();
    } else {
      setStage(4);
      setLoading(false);
    }
  };

  const handleEvalTS = () => {
    if (!selectedFeatures) {
      return;
    }
    const fd = new FormData();
    fd.append("features", JSON.stringify(selectedFeatures));
    fetch(`${server}/Set_Features_Data`, {
      method: "POST",
      body: fd,
    }).then(() => {
      fetch(`${server}/Get_Unique_Values`)
        .then((r) => r.json())
        .then((s) => {
          console.log(JSON.parse(s.unique));
          setUniqueValues(JSON.parse(s.unique));
          setStage(4);
        })
        .catch((e) => {
          console.error(e);
          setLoading(false);
        })
        .catch((e) => {
          console.error(e);
          setLoading(false);
        });
    });
  };
  return algorithm === "Time Series" ? (
			<>
				{" "}
				<div
					style={{
						opacity: stage >= 3 ? "1" : "0",
						pointerEvents: stage >= 3 ? "auto" : "none",
						height: stage >= 3 ? "auto" : "0",
					}}
					className=" px-2 pb-2 bg-black/10 rounded-md backdrop-blur-md border border-white/10 mt-1"
				>
					<Button
						disabled={
							target === null ||
							selectedFeatures.length === 0 ||
							correlation1Sheet !== null ||
							loading
						}
						onClick={handleEvalTS}
						className="mt-2 w-full transition-all duration-300"
					>
						Evaluate Time-series data
					</Button>
				</div>
			</>
		) : (
			<>
				<div
					style={{
						opacity: stage >= 3 ? "1" : "0",
						pointerEvents: stage >= 3 ? "auto" : "none",
						height: stage >= 3 ? "auto" : "0",
					}}
					className=" px-2 pb-2 bg-black/10 rounded-md backdrop-blur-md border border-white/10 mt-1"
				>
					<Button
						disabled={
							target === null ||
							selectedFeatures.length === 0 ||
							correlation1Sheet !== null ||
							loading
						}
						onClick={handleProcessCorrelation}
						className="mt-2 mb-1 w-full transition-all duration-300"
					>
						Process Correlation
					</Button>
					
					<Label>Mode</Label>
					<Select
						onValueChange={(
							value: "train&predict" | "predict" | "tune_hyperparameters",
						) => {
							setEndMode(value);
						}}
						value={endMode == null ? "Select Mode" : endMode}
					>
						<SelectTrigger className="w-full bg-white/10 border-white/10  ">
							<SelectValue>
								{endMode == null
									? "Select Mode"
									: endMode === "predict"
										? "Predict Only"
										: endMode === "tune_hyperparameters"
											? "Tune Hyperparameters"
											: "Train & Predict"}
							</SelectValue>
						</SelectTrigger>
						<SelectContent className="bg-white/10 border-white/10 backdrop-blur-md">
							<SelectItem value="train&predict">Train & Predict</SelectItem>
							<SelectItem value="predict">Predict Only</SelectItem>
							{algorithm === "Classification" && (
								<SelectItem value="tune_hyperparameters">
									Tune Hyperparameters
								</SelectItem>
							)}
						</SelectContent>
					</Select>
					{endMode === "train&predict" && (
						<Button
							disabled={
								!trainActive ||
								target === null ||
								selectedFeatures.length === 0 ||
								loading
							}
							onClick={() => {
								console.log("rtest");
								setLoading(true);
								set_server_data({
									attr: "algorithm",
									value: algorithm as string,
								}).then(() => {
									set_server_data({
										attr: "target",
										value: target as string,
									}).then(() => {
										set_server_data({
											attr: "feature_list",
											value: selectedFeatures,
										}).then(() => {
											handleTrain();
										});
									});
								});
							}}
							className="mt-2 w-full transition-all duration-300"
						>
							Train
						</Button>
					)}
				</div>
			</>
		);
}
