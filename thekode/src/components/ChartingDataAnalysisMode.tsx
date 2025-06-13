import { Button } from "./ui/button";
import { Label } from "./ui/label";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";
import axios from "axios";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { useFileStore } from "@/components/Zustand";
import { useState } from "react";
import React from "react";

export default function DataAnalysisSideFlap() {
  const {
    stage,
    collapsed,
    uniqueValues,
    setSelectedUniqueValues,
    selectedUniqueValues,
    algorithm,
    dataAnalysisStage,
    targets,
    setLoading,
    server,
    setDataAnalysisData,
    dataAnalysisXAxis,
    setDataAnalysisXAxis,
    selectedFeatures,
    target,
    loading,
  } = useFileStore();
  const [askContinue, setAskContinue] = useState<boolean | "NO" | "YES">(false);
  // const [xAxis, setXAxis] = useState(targets ? targets[0] : "");
  // const [dataAnalysisXAxis, dataAnalysisXAxis] = useState(targets ? targets[1] : "");

  const handleClick = async () => {
    setLoading(true);

    try {
      const formData = new FormData();
      if (!target || !dataAnalysisXAxis) {
        setLoading(false);

        return;
      }
      formData.append("target", target);
      formData.append("feature_selected", dataAnalysisXAxis);
      const x = axios.post(`${server}/Summarise_Data`, formData).then((r) => {
        console.log(r.data);
        console.log(r.data);
        setDataAnalysisData(r.data);
      });
      // const result = await x.data;
      // console.log(result);
      // console.log((await result).summary_data);

    } catch (error) {
      setDataAnalysisData(null);
      console.error("Error processing file:", error);
    } finally {
		      setLoading(false);

    }
  };
  return (
    <>
      <AlertDialog open={askContinue === true}>
        {/* <AlertDialogTrigger>Open</AlertDialogTrigger> */}
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Are you absolutely sure?</AlertDialogTitle>
            <AlertDialogDescription>
              This will train on{" "}
              {Object.values(selectedUniqueValues).reduce((acc, arr) => {
                return acc * arr.length;
              }, 1)}{" "}
              combinations
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel onClick={() => setAskContinue("NO")}>
              Cancel
            </AlertDialogCancel>
            <AlertDialogAction
              onClick={() => {
                setAskContinue("YES");
              }}
            >
              Continue
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>

      <div
        className="border bg-white/10 backdrop-blur-md border-white/10 absolute top-0 right-0 transition-all duration-300   -mx-2 rounded-md w-64 p-2  
      "
        style={{
          zIndex: stage > 2 && dataAnalysisStage >= 1 ? 0 : -10,
          opacity: stage > 2 && dataAnalysisStage >= 1 ? "1" : "0",
          pointerEvents: stage > 2 && dataAnalysisStage >= 1 ? "all" : "none",
          translate: stage > 2 && dataAnalysisStage >= 1 ? "94%" : "0%",
        }}
      >
        {/* <div className="group relative w-full -mt-1">
					<Label className="">Target</Label>
					<Select
						disabled={!targets}
						value={dataAnalysisXAxis ? dataAnalysisXAxis : ""}
						onValueChange={setDataAnalysisXAxis}
					>
						<SelectTrigger className="w-full bg-white/10 mt-1 border-white/10 ">
							<SelectValue placeholder="Select Attribute" />
						</SelectTrigger>
						<SelectContent className="bg-white/10 border-white/10 backdrop-blur-md">
							{targets?.map((i) => (
								<SelectItem disabled={i === dataAnalysisXAxis} value={i} key={i}>
									{i}
								</SelectItem>
							))}
						</SelectContent>
					</Select>
				</div> */}
        <div className="group relative w-full mt-2">
          <Label className="">Feature</Label>
          <Select
            disabled={!targets || loading}
            value={dataAnalysisXAxis ? dataAnalysisXAxis : ""}
            onValueChange={setDataAnalysisXAxis}
          >
            <SelectTrigger className="w-full bg-white/10 mt-1 border-white/10 ">
              <SelectValue placeholder="Select Attribute" />
            </SelectTrigger>
            <SelectContent className="bg-white/10 border-white/10 backdrop-blur-md">
              {selectedFeatures.map((i) => (
                <SelectItem
                  disabled={i === dataAnalysisXAxis}
                  value={i}
                  key={i}
                >
                  {i}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>

        <Button
          disabled={dataAnalysisXAxis === null || dataAnalysisXAxis === ""||loading}
          onClick={handleClick}
          className="w-full mt-2"
        >
          Go
        </Button>
      </div>
    </>
  );
}
