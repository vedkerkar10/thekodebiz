import { Check, Cross, LucideFireExtinguisher, X } from "lucide-react";
import { Button } from "./ui/button";
import { Label } from "./ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "./ui/select";
import { useFileStore } from "@/components/Zustand";
import { useEffect, useState } from "react";
import { config } from "../../config";
import { parsePythonDataFrame } from "@/lib/xlsx_preprocess";

export const AlgorithmSelector = () => {
  const [algorithmLock, setAlgorithmLock] = useState(false);

  const { processed, stage, setStage, setAlgorithm, algorithm ,loading,server,setAlgorithmPos} =
    useFileStore();

  useEffect(() => {
    if (algorithm === null) {
      setAlgorithmLock(false);
    }
  }, [algorithm]);
  const handleAlgorithmChange = (
    value: "Classification" | "Regression" | "Time Series" | null
  ) => {
    // const handleAlgorithmChange = (value: string | null) => {
    setAlgorithm(value);
    setStage(3);
  };

  // const handleSetAlgorithm = () => {
  //   if (stage >= 3) {
  //     setStage(2);
  //     // setAlgorithmLock(false);
  //   } else {
  //     setStage(3);
  //     // setAlgorithmLock(true);
  //   }
  // };
  return (
    <div
      className={
        "rounded-b-md border-x border-b px-2 pb-2 transition-all  ease-in-out duration-300 relative -z-10  border-white/10 bg-white/10  backdrop-blur-md "
      }
      style={{
        // translate: processed ? "0px 0px" : "0px -50px",
        opacity: processed && stage > 0 ? 1 : 0,
        zIndex: processed && stage > 0 ? 0 : -10,
        pointerEvents: processed && stage > 0 ? "auto" : "none",
      }}
    >
      <Label>Choose Method</Label>
      <div className="flex gap-2">
        <Select
          value={algorithm ? algorithm : "Select Algorithm"}
          disabled={algorithmLock || loading}
          onValueChange={(value: string | null) => {
            const fd = new FormData();
            fd.append("algorithm", value as string);

            fetch(`${server}/Get_Algo_Position`, {
              method: "POST",
              body: fd,
            })
              .then((response) => {
                if (!response.ok) {
                  throw new Error(`HTTP error! Status: ${response.status}`);
                }
                return response.json();
              })
              .then((data) => {
                setAlgorithmPos(parsePythonDataFrame(data.algo_results) as any);
              })
              .catch((error) => {
                console.error("Error:", error.message);
              });
            setAlgorithm(
              value as "Classification" | "Regression" | "Time Series"
            );
            setStage(3);
          }}
        >
          <SelectTrigger className="w-full bg-white/10 border-white/10 ">
            <SelectValue>
              {algorithm ? algorithm : "Select Algorithm"}
            </SelectValue>
          </SelectTrigger>
          <SelectContent className="bg-white/10 border-white/10 backdrop-blur-md">
            <SelectItem value="Classification">Classification</SelectItem>
            <SelectItem value="Regression">Regression</SelectItem>
            <SelectItem value="Time Series">Time Series</SelectItem>
          </SelectContent>
        </Select>
        {/* <Button
          disabled={algorithm === null || algorithm === undefined}
          onClick={handleSetAlgorithm}
          className="w-10 h-10 p-1 bg-white/10 border-white/5 border font-thin hover:bg-white/20"
          variant={"secondary"}
        >
          {algorithmLock ? <X /> : <Check />}

        </Button> */}
      </div>
    </div>
  );
};
