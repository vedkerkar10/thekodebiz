"use client";
import { type ChangeEvent, useEffect, useState } from "react";
import { Button } from "./ui/button";
import { Label } from "./ui/label";
import { useFileStore } from "@/components/Zustand";

import { parsePythonDataFrame } from "@/lib/xlsx_preprocess";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "./ui/select";

export default function HyperParameterOptions() {
  const {
			stage,
			setLoading,
			algorithm,
			loading,
			server,
			endMode,
			target,
			selectedFeatures,
			setBestParamsSheet,
			setConfusionMatrixSheet,
			setClassificationReportSheet,
			// hyperParameterFile,
			setHyperParameterFile,
			sethyperTuningEstimates,
			// hyperTuningEstimates,
			setChosenHyperTuningAlgo,
			chosenHyperTuningAlgo,
		} = useFileStore();

		const [hyperTuningAlgos, setHyperTuningAlgos] = useState<Array<{
			algo: string;
			position: number;
		}> | null>(null);
		// const [chosenHyperTuningAlgo, setChosenHyperTuningAlgo] = useState<
		// 	string | null
		// >(null);


		const handleEstimateHyperTuning = () => {
			console.log("processing");
			if (algorithm === "Classification") {
				if (
					algorithm !== null &&
					target !== null &&
					selectedFeatures !== null &&
					chosenHyperTuningAlgo !== null
				) {
					setLoading(true);
					const formData = new FormData();
					formData.append("algorithm", algorithm);
					formData.append(
						"hyper_position",
						chosenHyperTuningAlgo.split("%::%")[1],
					);
					formData.append("target", target);
					formData.append("features", JSON.stringify(selectedFeatures));
					fetch(`${server}/Hypertune_Estimate`, {
						method: "POST",
						body: formData,
					})
						.then((r) => r.json())
						.then((d) => {
							console.log(d);
							sethyperTuningEstimates(d);
							// console.log(d.no_of_fits);
							// console.log(d.estimated_time);

							setLoading(false);
						});
				}
			}
		};
		useEffect(() => {
			if (!algorithm) return;
			const formData = new FormData();
			formData.append("algorithm", algorithm);
			fetch(`${server}/Get_Hyper_Algo_Position`, {
				method: "POST",
				body: formData,
			})
				.then((r) => r.json())
				.then((d) => {
					console.log({ d });
					console.log({ table: parsePythonDataFrame(d.hyper_algo_results) });
					const parsedDF = parsePythonDataFrame(d.hyper_algo_results) as {
						data: Array<{
							"Hyper Algorithm Name": string;
							"Hyper Parameters": string;
							Position: number;
							hadID: boolean;
							id: number;
						}>;
					};
					setHyperTuningAlgos(
						parsedDF.data.map((x) => {
							return { algo: x["Hyper Algorithm Name"], position: x.Position };
						}),
					);
				});
		}, [algorithm, server]);
		return (
			<>
				{" "}
				<div
					style={{
						display:
							stage >= 5 || (endMode === "tune_hyperparameters" && stage >= 3)
								? "block"
								: "none",
					}}
					className={`${
						stage === 5 ? "rounded-b-md" : ""
					} transition-all  overflow-hidden duration-300 border-t border-x border-b rounded-t-md p-2 mt-1 bg-white/10 backdrop-blur-md border-white/10 relative z-0`}
				>
					<>
						<Label>Choose Algorithm for Hyper Tuning</Label>

						<Select
							value={
								chosenHyperTuningAlgo
									? chosenHyperTuningAlgo
									: "Select Algorithm"
							}
							disabled={loading}
							onValueChange={(value: string | null) => {
								if (!value) return;
								setChosenHyperTuningAlgo(value);
							}}
						>
							<SelectTrigger className="w-full bg-white/10 border-white/10 mt-2  ">
								<SelectValue>
									{chosenHyperTuningAlgo
										? chosenHyperTuningAlgo.split("%::%")[0]
										: "Select Algorithm"}
								</SelectValue>
							</SelectTrigger>
							<SelectContent className="bg-white/10 border-white/10 backdrop-blur-md h-64">
								{hyperTuningAlgos?.map((si) => (
									<SelectItem
										key={si.position}
										value={`${si.algo}%::%${si.position}`}
									>
										{si.algo}
									</SelectItem>
								))}
							</SelectContent>
						</Select>
					</>
					<Button
						onClick={handleEstimateHyperTuning}
						disabled={!chosenHyperTuningAlgo || loading}
						className="w-full  mt-2 "
					>
						Hyper Tuning Estimations
					</Button>
					{/* {hyperTuningEstimates && (
						<Button
							onClick={handleProcess}
							disabled={!chosenHyperTuningAlgo || loading}
							className="w-full  mt-2 "
						>
							Tune Hyperparameters
						</Button>
					)} */}
				</div>
			</>
		);
}
