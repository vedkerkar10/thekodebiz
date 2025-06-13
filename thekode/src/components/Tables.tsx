"use client";
import { uuidv4 } from "@/lib/utils";
import { format } from "date-fns";

import axios from "axios";
import { ArrowDownCircleIcon, UnplugIcon } from "lucide-react";
import { Dataframe } from "./DataFrame";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "./ui/tabs";
import { useFileStore } from "@/components/Zustand";
import Link from "next/link";
import { Input } from "./ui/input";
import { Label } from "./ui/label";
import { Button } from "./ui/button";
import { config } from "../../config";
import { MaterialSymbolsDownloadRounded } from "@/components/icons";
import { parsePythonDataFrame } from "@/lib/xlsx_preprocess";
import * as XLSX from "xlsx";
import { TimeseriesChart } from "./chart";
import { Checkbox } from "@/components/ui/checkbox";

import {
  Dialog,
  DialogClose,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import { use, useEffect, useState } from "react";
import { create } from "zustand";
import { header_type } from "./types";

interface TableProps {
  name: string;
  sheet: {
    headers: header_type[];
    data: { [key: string]: any; id: number }[];
  };
}
interface TableTabsProps {
  dynamicTable: TableProps[] | null;

  addTable: (value: TableProps) => void;
}

export const useTableTabsProps = create<TableTabsProps>((set, get) => ({
  dynamicTable: null,
  addTable: (value) =>
    set({ dynamicTable: [...(get().dynamicTable ?? []), value] }),
}));

export function Tables() {
  const {
    trainingFile,
    originalSheet,
    correlation1Sheet,
    correlation2Sheet,
    analysisSheet,
    predictionSheet,
    predictionFile,
    setLoading,
    currentSheet,
    setCurrentSheet,
    algorithm,
    analysisGraphs,
    predictionGraph,
    analysisMessage,
    server,
    includedColsDfDisplay,
    confusionMatrixSheet,
    bestParamsSheet,
    classificationReportSheet,
  } = useFileStore();

  const { dynamicTable } = useTableTabsProps();
  useEffect(() => {
    console.log(analysisSheet);
  }, [analysisSheet]);
  const [downloadOriginal, setDownloadOriginal] = useState<
    "indeterminate" | boolean
  >(true);
  const [downloadCorrelation1, setDownloadCorrelation1] = useState<
    "indeterminate" | boolean
  >(true);
  const [downloadCorrelation2, setDownloadCorrelation2] = useState<
    "indeterminate" | boolean
  >(true);
  const [downloadAnalysis, setDownloadAnalysis] = useState<
    "indeterminate" | boolean
  >(true);
  const [downloadBestParams, setDownloadBestParams] = useState<
    "indeterminate" | boolean
  >(true);
  const [downloadModelGraphAnalysis, setDownloadModelGraphAnalysis] = useState<
    "indeterminate" | boolean
  >(true);
  const [downloadConfusionMatrix, setDownloadConfusionMatrix] = useState<
    "indeterminate" | boolean
  >(true);
  const [downloadClassificationReport, setDownloadClassificationReport] =
    useState<"indeterminate" | boolean>(true);
  const [downloadPrediction, setDownloadPrediction] = useState<
    "indeterminate" | boolean
  >(true);

  function downloadServerSheets() {
    setLoading(true);
    const sheets = getSheetsToDownload();
    console.log(sheets);

    const formData = new FormData();
    formData.append("sheets", JSON.stringify(sheets));

    axios
      .post(`${server}/download`, formData, { responseType: "blob" })
      .then((resp) => {
        const file = new Blob([resp.data], {
          type: resp.headers["content-type"],
        });
        const link = document.createElement("a");
        link.href = URL.createObjectURL(file);
        link.download = `Output_${format(new Date(), "yyyyMMddHHmmss")}.zip`;
        link.click();
        setLoading(false);
      })
      .catch((error) => console.error(error));
  }
  function getSheetsToDownload(): string[] {
    const sheets: string[] = [];

    if (downloadOriginal) sheets.push("Original Sheet");
    if (algorithm !== "Time Series") {
      if (downloadCorrelation1) sheets.push("Correlation 1");
      if (downloadCorrelation2) sheets.push("Correlation 2");
    }
    if (algorithm === "Time Series" && downloadModelGraphAnalysis)
      sheets.push("Graphic Analysis");
    if (downloadAnalysis) sheets.push("Analysis");
    if (downloadPrediction) sheets.push("Prediction Sheet");
    if (downloadBestParams) sheets.push("Best Params");
    if (downloadConfusionMatrix) sheets.push("Confusion Matrix");
    if (downloadClassificationReport) sheets.push("Classification Report");

    return sheets;
  }

  useEffect(() => {
    console.log("bestParamsSheet", bestParamsSheet);
    console.log("confusionMatrixSheet", confusionMatrixSheet);
    console.log("classificationReportSheet", classificationReportSheet);
  });
  return (
    <div className=" relative">
      {algorithm === "Time Series" ? (
        <Tabs
          defaultValue="Original"
          value={currentSheet}
          onValueChange={(v) =>
            setCurrentSheet(
              v as
                | "Original"
                | "Correlation1"
                | "Correlation2"
                | "Analysis"
                | "Prediction"
                | string,
            )
          }
          className="w-full"
        >
          <TabsList className="bg-white/10 text-black/50  backdrop-blur-md border border-white/10">
            <TabsTrigger value="Original">Original</TabsTrigger>
            {/* <TabsTrigger
              value="Correlation1"
              disabled={correlation1Sheet === null}
            >
              Correlation 1
            </TabsTrigger>
            <TabsTrigger
              value="Correlation2"
              disabled={correlation2Sheet === null}
            >
              Correlation 2
            </TabsTrigger> */}
            <TabsTrigger value="Analysis" disabled={analysisSheet === null}>
              Analysis
            </TabsTrigger>
            <TabsTrigger disabled={predictionFile === null} value="Prediction">
              Prediction
            </TabsTrigger>
          </TabsList>
          <TabsContent value="Original">
            {trainingFile && (
              <Dataframe
                // hide_footer={false}
                data={originalSheet}
                name={"original"}
              />
            )}
          </TabsContent>
          <TabsContent value="Correlation1">
            {correlation1Sheet ? (
              <Dataframe
                // hide_footer={false}
                data={correlation1Sheet}
                name={"correlation1"}
              />
            ) : (
              <div />
            )}
          </TabsContent>
          <TabsContent value="Correlation2">
            {correlation2Sheet ? (
              <Dataframe data={correlation2Sheet} name={"correlation2"} />
            ) : (
              <div />
            )}{" "}
          </TabsContent>
          <TabsContent className="flex gap-2 w-full  h-full" value="Analysis">
            {analysisSheet ? (
              <>
                <Dataframe
                  className="w-1/2"
                  data={analysisSheet}
                  name={"analysis"}
                />
                <TimeseriesChart
                  className="w-2/3 h-full min-h-96"
                  chardData={analysisGraphs}
                />
              </>
            ) : (
              <div />
            )}{" "}
          </TabsContent>
          <TabsContent className="flex gap-2 w-full  h-full" value="Prediction">
            {predictionSheet ? (
              <>
                <Dataframe
                  className="w-1/2"
                  data={predictionSheet}
                  name={"prediction"}
                />
                <TimeseriesChart
                  className="w-2/3 h-full min-h-96"
                  chardData={predictionGraph}
                />
              </>
            ) : (
              <div />
            )}
          </TabsContent>
        </Tabs>
      ) : (
        <Tabs
          defaultValue="Original"
          value={currentSheet}
          onValueChange={(v) =>
            setCurrentSheet(
              v as
                | "Original"
                | "Correlation1"
                | "Correlation2"
                | "Analysis"
                | "Prediction",
            )
          }
          className="w-full"
        >
          <TabsList className="bg-white/10 text-black/50 max-w-[80rem] overflow-y-visible scrollbar-none h-full scrollbar-track-ring overflow-x-scroll backdrop-blur-md border border-white/10">
            <TabsTrigger value="Original">Original</TabsTrigger>
            <TabsTrigger
              value="Correlation1"
              disabled={correlation1Sheet === null}
            >
              Correlation 1
            </TabsTrigger>
            <TabsTrigger
              value="Correlation2"
              disabled={correlation2Sheet === null}
            >
              Correlation 2
            </TabsTrigger>
            <TabsTrigger value="Analysis" disabled={analysisSheet === null}>
              Analysis
            </TabsTrigger>
            <TabsTrigger disabled={predictionFile === null} value="Prediction">
              Prediction
            </TabsTrigger>

            <TabsTrigger disabled={bestParamsSheet === null} value="BestParams">
              Best Params
            </TabsTrigger>
            <TabsTrigger
              disabled={confusionMatrixSheet === null}
              value="ConfusionMatrix"
            >
              Confusion Matrix
            </TabsTrigger>
            <TabsTrigger
              disabled={classificationReportSheet === null}
              value="ClassificationReport"
            >
              Classification Report
            </TabsTrigger>

            {dynamicTable?.map((x) => (
              <TabsTrigger key={x.name} value={x.name}>
                {x.name}
              </TabsTrigger>
            ))}
          </TabsList>

          <TabsContent className="" value="Original">
            {trainingFile &&
              (analysisMessage ? (
                <div>
                  <Dataframe
                    hide_footer
                    data={originalSheet}
                    name={"original"}
                  />
                </div>
              ) : (
                <Dataframe hide_footer data={originalSheet} name={"original"} />
              ))}
          </TabsContent>
          <TabsContent value="Correlation1">
            {correlation1Sheet ? (
              <Dataframe data={correlation1Sheet} name={"correlation1"} />
            ) : (
              <div />
            )}
          </TabsContent>
          <TabsContent value="Correlation2">
            {correlation2Sheet ? (
              <Dataframe data={correlation2Sheet} name={"correlation1"} />
            ) : (
              <div />
            )}{" "}
          </TabsContent>
          <TabsContent value="Analysis">
            {analysisSheet ? (
              <Dataframe
                show_cols={includedColsDfDisplay}
                data={analysisSheet}
                name={"correlation1"}
              />
            ) : (
              <div />
            )}{" "}
          </TabsContent>
          <TabsContent value="Prediction">
            {predictionSheet ? (
              <Dataframe data={predictionSheet} name={"correlation1"} />
            ) : (
              <div />
            )}
          </TabsContent>

          <TabsContent value="BestParams">
            {bestParamsSheet ? (
              <Dataframe data={bestParamsSheet} name={"bestParams"} />
            ) : (
              <div>{JSON.stringify(bestParamsSheet)}</div>
            )}
          </TabsContent>
          <TabsContent value="ConfusionMatrix">
            {confusionMatrixSheet ? (
              <Dataframe
                data={confusionMatrixSheet}
                name={"confusionMatrixSheet"}
              />
            ) : (
              <div>{JSON.stringify(confusionMatrixSheet)}</div>
            )}
          </TabsContent>
          <TabsContent value="ClassificationReport">
            {classificationReportSheet ? (
              <Dataframe
                data={classificationReportSheet}
                name={"classificationReportSheet"}
              />
            ) : (
              <div>{JSON.stringify(classificationReportSheet)}</div>
            )}
          </TabsContent>

          {dynamicTable?.map((x) => (
            <TabsContent key={x.name} value={x.name}>
              {x.sheet ? (
                <Dataframe data={x.sheet} name={x.name} />
              ) : (
                <div>{JSON.stringify(x.sheet)}</div>
              )}
            </TabsContent>
            // <TabsTrigger  value={x.name}>
            //   {x.name}
            // </TabsTrigger>
          ))}
        </Tabs>
      )}

      <div className=" absolute top-0 right-0">
        <Dialog>
          <DialogTrigger asChild>
            <Button
              className="gap-2"
              disabled={originalSheet === null}

              // onClick={downloadSheet}
            >
              <MaterialSymbolsDownloadRounded />
              Download
            </Button>
          </DialogTrigger>
          <DialogContent className="flex flex-col">
            <DialogHeader>
              <DialogTitle>Download</DialogTitle>
              <DialogDescription>
                Select The Tables You want in your xlsx file
              </DialogDescription>
            </DialogHeader>
            <div>
              <div className="items-top flex space-x-2">
                <Checkbox
                  onCheckedChange={setDownloadOriginal}
                  checked={downloadOriginal}
                />
                <div className="grid gap-1.5 leading-none">
                  <label className="text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70">
                    Original Sheet
                  </label>
                </div>
              </div>
              {algorithm !== "Time Series" && (
                <>
                  <div className="items-top flex space-x-2 mt-4">
                    <Checkbox
                      onCheckedChange={setDownloadCorrelation1}
                      checked={downloadCorrelation1}
                    />
                    <div className="grid gap-1.5 leading-none">
                      <label className="text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70">
                        Correlation 1
                      </label>
                    </div>
                  </div>
                  <div className="items-top flex space-x-2 mt-4">
                    <Checkbox
                      onCheckedChange={setDownloadCorrelation2}
                      checked={downloadCorrelation2}
                    />
                    <div className="grid gap-1.5 leading-none">
                      <label className="text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70">
                        Correlation 2
                      </label>
                    </div>
                  </div>
                </>
              )}
              <div className="items-top flex space-x-2 mt-4">
                <Checkbox
                  onCheckedChange={setDownloadAnalysis}
                  checked={downloadAnalysis}
                />
                <div className="grid gap-1.5 leading-none">
                  <label className="text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70">
                    Analysis Sheet
                  </label>
                </div>
              </div>

              <div className="items-top flex space-x-2 mt-4">
                <Checkbox
                  onCheckedChange={setDownloadBestParams}
                  checked={downloadBestParams}
                />
                <div className="grid gap-1.5 leading-none">
                  {/* biome-ignore lint/a11y/noLabelWithoutControl: <explanation> */}
                  <label className="text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70">
                    Best Params
                  </label>
                </div>
              </div>

              <div className="items-top flex space-x-2 mt-4">
                <Checkbox
                  onCheckedChange={setDownloadConfusionMatrix}
                  checked={downloadConfusionMatrix}
                />
                <div className="grid gap-1.5 leading-none">
                  <label className="text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70">
                    Confusion Matrix
                  </label>
                </div>
              </div>

              <div className="items-top flex space-x-2 mt-4">
                <Checkbox
                  onCheckedChange={setDownloadClassificationReport}
                  checked={downloadClassificationReport}
                />
                <div className="grid gap-1.5 leading-none">
                  <label className="text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70">
                    Classification Report
                  </label>
                </div>
              </div>

              {algorithm === "Time Series" && (
                <div className="items-top flex space-x-2 mt-4">
                  <Checkbox
                    onCheckedChange={setDownloadModelGraphAnalysis}
                    checked={downloadModelGraphAnalysis}
                  />
                  <div className="grid gap-1.5 leading-none">
                    <label className="text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70">
                      Model Graph Analysis Sheet
                    </label>
                  </div>
                </div>
              )}
              <div className="items-top flex space-x-2 mt-4">
                <Checkbox
                  onCheckedChange={setDownloadPrediction}
                  checked={downloadPrediction}
                />
                <div className="grid gap-1.5 leading-none">
                  <label className="text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70">
                    Prediction Sheet
                  </label>
                </div>
              </div>
            </div>
            <DialogClose asChild>
              <Button
                type="button"
                onClick={downloadServerSheets}
                className=" mt-3 flex gap-3"
              >
                Confirm And Download <ArrowDownCircleIcon />
              </Button>
            </DialogClose>
          </DialogContent>
        </Dialog>
      </div>
    </div>
  );
}
