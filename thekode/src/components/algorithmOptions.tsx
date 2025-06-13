import { Input } from "./ui/input";
import { Label } from "./ui/label";
import { useFileStore } from "@/components/Zustand";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "./ui/select";
import { TimeSeriesOptions } from "./algorithmOptions/timeSeries";
import { ClassificationOptions } from "./algorithmOptions/classification";
import { RegressionOptions } from "./algorithmOptions/regression";

export function AlgorithmOptions() {
  const { stage, algorithm } = useFileStore();
  return (
    <div
      // transition-all duration-300 border-t border-x border-b rounded-t-md   bg-white relative z-0
      className="border  rounded-md relative transition-all duration-300 p-2  bg-white/10 border-white/10 backdrop-blur-md mt-1"
      style={{
        zIndex: stage === 3 ? 0 : -10,
        opacity: stage >= 3 ? "1" : "0",
        // translate: stage >= 3 ? "0 8px" : "0px -100px",
      }}
    >
      <Label>{algorithm}</Label>
      {algorithm === "Regression" && <RegressionOptions />}
      {algorithm === "Time Series" && <TimeSeriesOptions />}
      {algorithm === "Classification" && <ClassificationOptions />}
    </div>
  );
}

