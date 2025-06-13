"use client";
import { parsePythonDataFrame } from "@/lib/xlsx_preprocess";
import { useFileStore } from "./Zustand";
import { Button } from "./ui/button";
import { useEffect, useState } from "react";
import { cn } from "@/lib/utils";
import { ChevronsRight } from "lucide-react";

export function HyperTuningSidebar() {
	const {
		stage,
		setStage,
		setLoading,
		algorithm,
		loading,
		server,
		endMode,
		target,
		selectedFeatures,
		setHyperSheet,
		setBestParamsSheet,
		setConfusionMatrixSheet,
		setClassificationReportSheet,
		hyperParameterFile,
		setHyperParameterFile,
		sethyperTuningEstimates,
		hyperTuningEstimates,
		chosenHyperTuningAlgo,
		resetHyperTuningEstimates,
		collapsed,
	} = useFileStore();
	const [closeDisplay, setCloseDisplay] = useState(false);

	useEffect(() => {
		if (!hyperTuningEstimates) {
			setCloseDisplay(false);
		}
	}, [hyperTuningEstimates]);
	const handleProcess = () => {
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
				fetch(`${server}/Tune_Hyperparameters`, {
					method: "POST",
					body: formData,
				})
					.then((r) => r.json())
					.then((d) => {
						console.log(d);
						console.log(d.best_params_df);
						console.log(
							"best_params_dfbest_params_df",
							parsePythonDataFrame(d.best_params_df),
						);
						console.log(
							"confusion_matrix",
							parsePythonDataFrame(d.confusion_matrix),
						);
						console.log(
							"classification_report",
							parsePythonDataFrame(d.classification_report),
						);

						setBestParamsSheet(parsePythonDataFrame(d.best_params_df));
						setConfusionMatrixSheet(
							// biome-ignore lint/suspicious/noExplicitAny: <explanation>
							parsePythonDataFrame(d.confusion_matrix) as any,
						);
						setClassificationReportSheet(
							// biome-ignore lint/suspicious/noExplicitAny: <explanation>
							parsePythonDataFrame(d.classification_report) as any,
						);
						setCloseDisplay(true);
						setLoading(false);
					});
			}
		}
	};
	const handleCancel = () => {
		setStage(5);
		setHyperParameterFile(null);
		resetHyperTuningEstimates();
	};
	return (
		<div
			className={cn(
				collapsed ? "opacity-0 pointer-events-none" : "",
				closeDisplay
					? "-left-7 bg-transparent "
					: "left-72 z-50 bg-neutral-600/10 backdrop-blur-3xl border border-white/20 shadow-lg",
				" fixed top-22 w-80  transition-all  ease-out     rounded-lg  overflow-hidden",
			)}
		>
			<div className={cn("p-6 relative")}>
				<button
					type="button"
					onClick={() => setCloseDisplay(!closeDisplay)}
					className={cn(
						closeDisplay
							? "bg-neutral-200/20 hover:bg-neutral-100/80 rounded-r-full backdrop-blur-3xl"
							: "bg-neutral-600/0",
						"p-2 absolute top-0 right-0 z-50 cursor-pointer group",
					)}
				>
					<ChevronsRight
						style={{
							rotate: closeDisplay ? "180deg" : "0deg",
						}}
						className=" h-5 group-hover:rotate-180  transition-all group-hover:text-indigo-800"
					/>
				</button>

				{!closeDisplay && (
					<>
						<h2 className="text-xl font-semibold mb-4 text-gray-800 ">
							Hyper Tuning
						</h2>
						{hyperTuningEstimates ? (
							<div className="space-y-1">
								<div className="flex items-center space-x-2">
									<span className="text-sm text-gray-600 ">
										Number of Fits:
									</span>
									<span className="text-sm font-medium text-gray-800 ">
										{hyperTuningEstimates.no_of_fits}
									</span>
								</div>
								<hr className="border-gray-800/10" />
								<div className="flex items-start space-x-2">
									<span className="text-sm text-gray-600 ">
										Estimated Time:
									</span>
									<span className="font-medium text-sm text-gray-800 ">
										{Math.round(hyperTuningEstimates.estimated_time * 100) /
											100}{" "}
										mins
									</span>
								</div>
							</div>
						) : (
							<p className="text-sm text-gray-600 ">No estimates available</p>
						)}
					</>
				)}
			</div>
			<div
				className={cn(
					closeDisplay ? "opacity-0" : "",
					"px-6 py-4 bg-gray-800/20  flex justify-between",
				)}
			>
				<Button
					onClick={handleProcess}
					disabled={loading || !hyperTuningEstimates}
					className="w-full mr-2"
				>
					Start Hyper Tuning
					{/* {isLoading ? "Processing..." : "Start Hyper Tuning"} */}
				</Button>
				<Button
					variant="destructive"
					onClick={handleCancel}
					disabled={loading}
					className="w-full ml-2"
				>
					Cancel
				</Button>
			</div>
		</div>
	);
}