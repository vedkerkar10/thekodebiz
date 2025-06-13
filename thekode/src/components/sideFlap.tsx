import { parsePythonDataFrame } from "@/lib/xlsx_preprocess";
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

import { useFileStore } from "@/components/Zustand";
import { useEffect, useState } from "react";
import { PlotValsSelect } from "./PlotValsSelect";
import { toast } from "sonner";

export default function SideFlap() {
  const {
    stage,
    setStage,
    collapsed,
    uniqueValues,
    setSelectedUniqueValues,
    setAnalysisSheet,
    setCurrentSheet,
    selectedUniqueValues,
    target,
    selectedFeatures,
    algorithm,
    setLoading,
    aggregator,
    trainingFile,
    frequency,
    setAnalysisGraphs,
    granularity,
    epochs,
    setModelAnalysisSheets,
    server
  } = useFileStore();
  const [askContinue, setAskContinue] = useState<boolean | "NO" | "YES">(false);
  const train_model = async () => {
    setLoading(true);

    setStage(5);

    // setLoadingPercent(true);
    if (
      askContinue === "NO" ||
      !target ||
      !selectedFeatures ||
      !trainingFile ||
      !frequency ||
      !aggregator ||
      !selectedUniqueValues ||
      !granularity
    ) {
      toast.message("Cancelled Building");
      setLoading(false);
      return;
    }
    toast.message("Building Model");
    const url = `${server}/time_series/start`;
    const form = new FormData();
    form.append("target", target);
    form.append("features", JSON.stringify(selectedFeatures));
    form.append("frequency", frequency);
    form.append("granularity", granularity);
    form.append("agg_date", JSON.stringify(aggregator));
    form.append("uniqueValues", JSON.stringify(selectedUniqueValues));
    form.append("file", trainingFile);
    form.append("epochs", epochs.toString());

    // form.append('trainingFile', 'trainingFile');
    // form.append('predictionFile', 'predictionFile');
    // form.append('aggregator', '["date"]');
    // form.append('features', '{"store":[1],"item":[1]}');

    // const options = {method: 'POST'};

    try {
      const response = await fetch(url, {
        method: "POST",
        body: form,
      })
      const data = await response.json();
      console.log(data.graphs_df)
      setModelAnalysisSheets(parsePythonDataFrame(data.graphs_df) as any);
      // JSON.parse(data.graphs_df).map((i) =>
      // );
      // console.log(parsePythonDataFrame(data.graphs_df));
      setAnalysisSheet(parsePythonDataFrame(data.EVAL) as any);
      setAnalysisGraphs(JSON.parse(data.graphs));
      setLoading(false);
      setCurrentSheet("Analysis");
      toast.success("Model/s Built");
    } catch (error) {
      console.error(error);
    }
  };
  useEffect(() => {
    Object.keys(uniqueValues).map((i) => {
      setSelectedUniqueValues(i, uniqueValues[i][0]);
    });
  }, [setSelectedUniqueValues, uniqueValues]);
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
									train_model();
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
						zIndex:
							stage === 4 && !collapsed && algorithm === "Time Series"
								? 0
								: -10,
						opacity:
							stage === 4 && !collapsed && algorithm === "Time Series"
								? "1"
								: "0",
						pointerEvents:
							stage === 4 && !collapsed && algorithm === "Time Series"
								? "all"
								: "none",
						translate:
							stage === 4 && !collapsed && algorithm === "Time Series"
								? "94%"
								: "0%",
					}}
				>
					<div className="h-[75vh] overflow-y-scroll  scrollbar-thumb-neutral-200/50 scrollbar-thumb-rounded-full  scrollbar-track-transparent scrollbar-thin">
						{Object.keys(uniqueValues).map((i) => (
							<>
								<Label>{i}</Label>
								<PlotValsSelect attr={i} />
							</>
						))}
					</div>

					<Button
						disabled={Object.values(selectedUniqueValues).some(
							(array) => array.length === 0,
						)}
						onClick={() => {
							if (
								Object.values(selectedUniqueValues).reduce((acc, arr) => {
									return acc * arr.length;
								}, 1) < 2
							) {
								train_model();
								return;
							}
							setAskContinue(true);
						}}
						className="w-full mt-2"
					>
						Build Model
					</Button>
				</div>
			</>
		);
}
