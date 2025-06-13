"use client";
import { ChangeEvent, useEffect, useState } from "react";
import { Button } from "./ui/button";
import { Input } from "./ui/input";
import { Label } from "./ui/label";
import { useFileStore } from "@/components/Zustand";
import { config } from "../../config";
import { parsePythonDataFrame } from "@/lib/xlsx_preprocess";
import { toast } from "sonner";
import { useRouter } from "next/router";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "./ui/select";

export default function PredictionOptions() {
  const {
    stage,
    setStage,
    setLoading,
    setLoadingPercent,
    setPredictionFile,
    predictionFile,
    analysisSheet,
    loading,
    setPredictionSheet,
    setCurrentSheet,
    algorithm,
    target,
    selectedFeatures,
    trainingFile,
    aggregator,
    selectedUniqueValues,
    setPredictionGraph,
    server,
    endMode,
    algorithmPos,
  } = useFileStore();

  const [algorithm_rank_pos_prob, set_algorithm_rank_pos_prob] = useState<
    string | null
  >(null);

  const [algorithm_rank_name, set_algorithm_rank_name] = useState<
    string | null
  >(null);

  useEffect(() => {
    const si = algorithmPos?.data.filter((x) => x.Rank === 1)[0];
    if (!si) return;
    set_algorithm_rank_pos_prob(
      `${si.Rank}%::%${si.Position}%::%${si["Probability Flag"]}`,
    );
  }, [algorithmPos]);
  async function get_file_data({
    data,
    file_type,
  }: {
    data: File;
    file_type: string;
  }) {
    const formDataX = new FormData();
    formDataX.append("data", data);
    formDataX.append("file_type", file_type);
    return fetch(`${server}/GetData`, {
      method: "POST",
      body: formDataX,
    })
      .then((r) => r.json())
      .then((d) => {
        if (d.status === 200) {
          setLoading(false);
          return 1;
        }
      })
      .catch((e) => {
        console.error(e);
        setLoading(false);
      });
  }

  async function predict_values({
    file,
    algorithm,
    target,
    features,
  }: {
    file: File;
    algorithm: string;
    target: string;
    features: string[];
  }) {
    const isCompleted = await get_file_data({
      data: file,
      file_type: "Predict",
    });
    if (!isCompleted) {
      return 0;
    }
    setLoading(true);
    setLoadingPercent(true);
    const formData = new FormData();
    formData.append("algorithm", algorithm);
    formData.append("target", target);
    formData.append("features", JSON.stringify(features));
    if (algorithm_rank_pos_prob) {
      formData.append(
        "model_position",
        algorithm_rank_pos_prob?.split("%::%")[1],
      );
      formData.append(
        "model_probability_flag",
        algorithm_rank_pos_prob?.split("%::%")[2],
      );
    }

    fetch(`${server}/Predict`, { method: "POST", body: formData })
      .then((r) => r.json())
      .then((d) => {
        setLoading(false);
        setLoadingPercent(false);
        setPredictionSheet(parsePythonDataFrame(d.Final) as any);
        setCurrentSheet("Prediction");
        toast.success("Prediction Complete");
      })
      .catch((e) => {
        console.error(e);
        setLoading(false);
      });
  }
  async function test_and_predict_data({
    predictionFile,
    trainingFile,
    aggregator,
    selectedUniqueValues,
  }: {
    predictionFile: File;
    trainingFile: File;
    aggregator: string[];
    selectedUniqueValues: { [key: string]: (string | number)[] };
  }) {
    setLoading(true);
    setLoadingPercent(true);
    if (!target) {
      return;
    }
    const formData = new FormData();
    formData.append("predictionFile", predictionFile);
    formData.append("trainingFile", trainingFile);
    formData.append("aggregator", JSON.stringify(aggregator));
    formData.append("features", JSON.stringify(selectedUniqueValues));
    formData.append("target", target);
    await fetch(`${server}/test/`, {
      method: "POST",
      body: formData,
    })
      .then((r) => r.json())
      .then((d) => {
        if (d.Error) {
          if (d.Level === 1) {
            toast.warning(
              `Level 1 Error Detected, will continue with prediction, CODE : ${
                d.Errors.join(
                  ", ",
                )
              }`,
            );
          } else {
            toast.error(
              `Error Detected,stopped execution, CODE : ${
                d.Errors.join(", ")
              } `,
            );
            setLoading(false);
            return;
          }
        }
        predict_values_ts({
          predictionFile: predictionFile,
          trainingFile: trainingFile,
        });
      })
      .catch((e: Error) => {
        toast.warning(
          `Level 3 Error Detected, will continue with prediction, CODE : ${e.message}`,
        );
        setLoading(false);
      });
  }
  async function predict_values_ts({
    predictionFile,
    trainingFile,
  }: {
    predictionFile: File;
    trainingFile: File;
  }) {
    setLoading(true);
    setLoadingPercent(true);
    const formData = new FormData();
    formData.append("predictionFile", predictionFile);
    formData.append("trainingFile", trainingFile);

    await fetch(`${server}/time_series/Predict`, {
      method: "POST",
      body: formData,
    })
      .then((r) => r.json())
      .then((d) => {
        if (Object.keys(d).includes("Error")) {
          alert(`Error With File ${d.Error}`);
          return;
        }
        console.log("Prediction : ", d.Final);
        console.log("Prediction Converted : ", parsePythonDataFrame(d.Final));
        setLoading(false);
        setLoadingPercent(false);
        setPredictionSheet(parsePythonDataFrame(d.Final) as any);
        setCurrentSheet("Prediction");
        setPredictionGraph(JSON.parse(d.graphs));
        toast.success("Prediction Complete");
      })
      .catch((e) => {
        console.error(e);
        setLoading(false);
      });
  }

  const handleFileInputClick = () => {
    setStage(6);
  };

  const handleFileChange = (event: ChangeEvent<HTMLInputElement>) => {
    const selectedFile = event.target.files ? event.target.files[0] : null;
    if (selectedFile) {
      setPredictionFile(selectedFile);
      setLoading(false);
    }
  };
  const handleProcess = () => {
    console.log("processing");
    if (algorithm === "Time Series") {
      if (predictionFile !== null && trainingFile !== null) {
        test_and_predict_data({
          predictionFile: predictionFile as File,
          trainingFile: trainingFile as File,
          aggregator: aggregator as string[],
          selectedUniqueValues: selectedUniqueValues,
        });
      }
    } else {
      if (predictionFile !== null) {
        predict_values({
          algorithm: algorithm as string,
          target: target as string,
          features: selectedFeatures,
          file: predictionFile as File,
        });
        // get_file_data({ data: trainingFile, file_type: "Train" });
      }
    }
  };

  return (
    <>
      {" "}
      <div
        style={{
          display:
            stage >= 5 || (endMode === "predict" && stage >= 3)
              ? "block"
              : "none",
        }}
        className={`${
          stage === 5 ? "rounded-b-md" : ""
        } transition-all  overflow-hidden duration-300 border-t border-x border-b rounded-t-md p-2 mt-1 bg-white/10 backdrop-blur-md border-white/10 relative z-0`}
      >
        <Label>Choose File For Prediction</Label>
        <Input
          className="bg-white/10 backdrop-blur-md border-white/10"
          accept=".xls, .xlsx, .csv"
          type="file"
          disabled={loading}
          onClick={handleFileInputClick}
          onChange={handleFileChange}
        />
        {algorithm === "Time Series" ? (
          <></>
        ) : (
          <>
            <Label>Choose Algorithm to use</Label>

            <Select
              value={
                algorithm_rank_pos_prob
                  ? algorithm_rank_pos_prob
                  : "Select Algorithm"
              }
              disabled={loading}
              onValueChange={(value: string | null) => {
                set_algorithm_rank_pos_prob(value);

                // setStage(3);
              }}
            >
              <SelectTrigger className="w-full bg-white/10 border-white/10 ">
                <SelectValue>
                  {algorithm_rank_pos_prob
                    ? algorithm_rank_pos_prob.split("%::%")[0]
                    : "Select Algorithm"}
                </SelectValue>
              </SelectTrigger>
              <SelectContent className="bg-white/10 border-white/10 backdrop-blur-md h-64">
                {algorithmPos?.data.map((si) => (
                  <SelectItem
                    key={si.Position}
                    value={`${si["Algorithm Name"]}%::%${si.Position}%::%${si["Probability Flag"]}`}
                  >
                    {si["Algorithm Name"]}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </>
        )}

        <Button
          onClick={handleProcess}
          disabled={!predictionFile || loading}
          className="w-full mt-2"
        >
          Perform Prediction
        </Button>
      </div>
    </>
  );
}
