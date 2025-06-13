"use client";
import { ChangeEvent, useEffect, useState } from "react";
import { Button } from "./ui/button";
import { Input } from "./ui/input";
import { Label } from "./ui/label";
import { useFileStore } from "@/components/Zustand";
import * as XLSX from "xlsx";
import { config } from "../../config";
import { ChevronsRight } from "lucide-react";
import { toast } from "sonner";
import { row } from "./types";
import { parsePythonDataFrame } from "@/lib/xlsx_preprocess";
export const FileSelector = () => {
  const {
			setProcessed,
			setStage,
			stage,
			setTrainingFile,
			setLoading,
			trainingFile,
			setOriginalSheet,
			setTargets,
			loading,
			setCurrentSheet,
			setAnalysisSheet,
			setCorrelation1Sheet,
			setCorrelation2Sheet,
			setPredictionFile,
			setPredictionSheet,
			setAggregates,
			setAlgorithm,
			setTarget,
			setGranularity,
			setAggregator,
			setFrequency,
			setSelectedUniqueValues,
			setPredictionGraph,
			server,
			setAnalysisMessage,
			handleEndDataAnalysisMode,
		} = useFileStore();
  const [isProcessed, setIsProcessed] = useState(false);

  useEffect(() => {
    if (trainingFile) {
					// alert(
					//   `You Have Selected a File Of Size ${
					//     Math.round((trainingFile.size / (1024 * 1024)) * 100) / 100
					//   }Mb, The Screen may freeze for a short time `
					// );
					// trainingFile.arrayBuffer().then((fileData) => {
					//   const workbook = XLSX.read(fileData, {
					//     type: "binary",
					//     cellDates: true,
					//   });
					//   const sheetName = workbook.SheetNames[0];
					//   const sheet = workbook.Sheets[sheetName];
					//   const sheetData = XLSX.utils.sheet_to_json(sheet) as any;
					//   const headers = Object.keys(sheetData[0]).map((columnName: string) => ({
					//     field: columnName,
					//     headerName: columnName,
					//     type: typeof sheetData[0][columnName],
					//     width: 100,
					//   }));
					//   const hasOwnProp = Object.prototype.hasOwnProperty; // Renamed to avoid shadowing
					//   for (let index = 0; index < sheetData.length; index++) {
					//     const rowData = sheetData[index];
					//     rowData.hadID = true;
					//     if (!hasOwnProp.call(rowData, "id")) {
					//       rowData.id = index;
					//       rowData.hadID = false;
					//     }
					//     for (const key in rowData) {
					//       if (hasOwnProp.call(rowData, key) && rowData[key] instanceof Date) {
					//         const date = rowData[key] as Date;
					//         const formattedDate = `${date.getDate()}/${
					//           date.getMonth() + 1
					//         }/${date.getFullYear()}`;
					//         rowData[key] = formattedDate;
					//       }
					//     }
					//   }
					//   setOriginalSheet({
					//     headers: headers,
					//     data: sheetData,
					//   });
					//   setCurrentSheet("Original");
					//   setLoading(false);
					//   toast.success("Table Loaded Successfully");
					// });
				}
  }, [setCurrentSheet, setLoading, setOriginalSheet, trainingFile]);

  async function get_file_data({
    data,
    file_type,
  }: {
    data: File;
    file_type: string;
  }) {
    setLoading(true);
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
          // console.log(d.target_features);
          if (file_type === "Train") {
            setTargets(d.target_features);
            setAggregates(d.date_columns);
            setAnalysisMessage(d.data_analysis);
        setOriginalSheet(parsePythonDataFrame(d.display_data as any) as any)
        setCurrentSheet("Original");
          }
          setLoading(false);
          setProcessed(true);
          toast.success("Data Processed Successfully");
          toast.success("Table Loaded Successfully");
          setStage(2);
          return 1;
        }
      }).catch(e=>{
        console.error(e)
        setLoading(false)
      })
  }

  const handleFileInputClick = () => {
    setStage(0);
    reset();
  };
  const reset = () => {
    setIsProcessed(false);
    setAnalysisSheet(null);
    setCorrelation1Sheet(null);
    setCorrelation2Sheet(null);
    setPredictionFile(null);
    setPredictionSheet(null);
    setAlgorithm(null);
    setTarget(null);
    setGranularity(null);
    setAggregator(null);
    setFrequency(null);
    setPredictionGraph(null);
  };
  const handleFileChange = (event: ChangeEvent<HTMLInputElement>) => {
    const selectedFile = event.target.files ? event.target.files[0] : null;
    if (event.target.files?.length === 0) {
      setLoading(false);
    }
    if (selectedFile) {
      reset();
      setTrainingFile(selectedFile);
    }
  };

  const handleProcess = () => {
    setStage(2)
    handleEndDataAnalysisMode();

    if (trainingFile !== null) {
      setAlgorithm(null);
      get_file_data({ data: trainingFile, file_type: "Train" });
      setIsProcessed(true);
    }
  };
  return (
    <div
      className={`${
        stage === 1 ? "rounded-b-md" : ""
      } transition-all duration-300 border-t border-x border-b rounded-t-md p-2  bg-white/10 backdrop-blur-md border-white/10 relative z-0`}
    >
      <Label>Choose File</Label>
      {/* <Label
        className=" mt-1 h-10 rounded-md border pl-5 overflow-hidden flex items-center bg-white/10 backdrop-blur-md border-white/10 hover:border-white/20"
        htmlFor="fs"
      >
        <span className=" whitespace-nowrap">Choose File </span>
        <span
          className={
            "text-xs text-black/50 ml-2 w-full text-ellipsis overflow-hidden  whitespace-nowrap "
          }
        >
          {trainingFile ? trainingFile.name : "No File Chosen"}
        </span>
      </Label> */}
      <Input
        id="fs"
        tabIndex={-1}
        className=" backdrop-blur-md bg-white/10 border-white/10 "
        accept=".xls, .xlsx, .csv"
        type="file"
        onClick={handleFileInputClick}
        onChange={handleFileChange}
      />
      <Button
        onClick={handleProcess}
        disabled={!trainingFile || loading}
        className="w-full mt-2"
      >
        Fetch Training Data
      </Button>
    </div>
  );
};
