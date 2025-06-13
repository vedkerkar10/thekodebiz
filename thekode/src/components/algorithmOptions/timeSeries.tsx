import { useState } from "react";
import { FeatureSelect } from "../FeatureSelect";
import { Input } from "../ui/input";
import { Label } from "../ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "../ui/select";
import { useFileStore } from "@/components/Zustand";
import { Slider } from "../ui/slider";

export const TimeSeriesOptions = () => {
  // const [epochs, setEpochs] = useState(0);
  const {
    epochs,
    setEpochs,
    targets,
    setTarget,
    target,
    aggregates,
    aggregator,
    setAggregator,
    setFrequency,
    frequency,
    setPercentageAccuracy,
    percentageAccuracy,
    setGranularity,
    granularity,
    loading,
  } = useFileStore();
  const freqKeyMap: { [key: string]: string } = {
    D: "Daily",
    M: "Monthly",
    Q: "Quarterly",
    H: "Half-Yearly",
    Y: "Yearly",
  };
  return (
    <div>
      <Label>Select Target</Label>

      <Select
        value={target !== null ? target : ""}
        onValueChange={(e) => setTarget(e)}
        disabled={targets === null || loading}
      >
        <SelectTrigger className="w-full bg-white/10 border-white/10 ">
          <SelectValue>Target : {target ? target : "Not Chosen"}</SelectValue>
        </SelectTrigger>
        <SelectContent className="bg-white/10 border-white/10 backdrop-blur-md">
          {targets !== null ? (
            targets.map((i) => (
              <SelectItem value={i} key={i}>
                {i}
              </SelectItem>
            ))
          ) : (
            <></>
          )}
        </SelectContent>
      </Select>

      <Label>Select Features</Label>
      <FeatureSelect />

      <Label>Select Aggregator</Label>

      <Select
        disabled={loading}
        value={aggregator ? aggregator[0] : ""}
        onValueChange={(v) => setAggregator([v])}
      >
        <SelectTrigger className="w-full bg-white/10 border-white/10 ">
          <SelectValue>
            Aggregator : {aggregator ? aggregator : "Not Chosen"}
          </SelectValue>
        </SelectTrigger>
        <SelectContent className="bg-white/10 border-white/10 backdrop-blur-md">
          {aggregates !== null ? (
            aggregates.map((i) => (
              <SelectItem value={i} key={i}>
                {i}
              </SelectItem>
            ))
          ) : (
            <></>
          )}
        </SelectContent>
      </Select>
      <Label>Select Frequency</Label>
      <Select disabled={loading} onValueChange={(v) => setFrequency(v)}>
        <SelectTrigger className="w-full bg-white/10 border-white/10 ">
          <SelectValue>
            Frequency : {frequency ? freqKeyMap[frequency] : "Not Chosen"}{" "}
          </SelectValue>
        </SelectTrigger>
        <SelectContent className="bg-white/10 border-white/10 backdrop-blur-md">
          <SelectItem value="D">Daily</SelectItem>
          <SelectItem value="M">Monthly</SelectItem>
          <SelectItem value="Q">Quarterly</SelectItem>
          <SelectItem value="H">Half-Yearly</SelectItem>
          <SelectItem value="Y">Yearly</SelectItem>
        </SelectContent>
      </Select>
      <Label>Select Granularity</Label>
      <Select disabled={loading} onValueChange={(v) => setGranularity(v)}>
        <SelectTrigger className="w-full bg-white/10 border-white/10 ">
          <SelectValue>
            Granularity : {granularity ? freqKeyMap[granularity] : "Not Chosen"}{" "}
          </SelectValue>
        </SelectTrigger>
        <SelectContent className="bg-white/10 border-white/10 backdrop-blur-md">
          <SelectItem
            disabled={
              frequency === "M" ||
              frequency === "Q" ||
              frequency === "H" ||
              frequency === "Y"
            }
            value="D"
          >
            Daily
          </SelectItem>
          <SelectItem
            disabled={
              frequency === "Q" || frequency === "H" || frequency === "Y"
            }
            value="M"
          >
            Monthly
          </SelectItem>
          <SelectItem
            disabled={frequency === "H" || frequency === "Y"}
            value="Q"
          >
            Quarterly
          </SelectItem>
          <SelectItem disabled={frequency === "Y"} value="H">
            Half-Yearly
          </SelectItem>
          <SelectItem  value="Y">
            Yearly
          </SelectItem>
        </SelectContent>
      </Select>
      <div className="flex items-center  gap-2 ">
        <Label className="">Training Cycles : </Label>
        <Input
          disabled={loading}
          type="number"
          inputMode="numeric"
          onChange={(e) => setEpochs(Math.max(parseInt(e.target.value), 1))}
          className="w-20 text-center bg-transparent active:bg-white/10 border-white/0 active:border-white/10 font-mono"
          value={epochs}
        />
      </div>
      <Slider
        disabled={loading}
        className=" pt-2 pb-3 "
        value={[epochs]}
        onValueChange={(v: number[]) => setEpochs(v[0])}
        min={1}
        defaultValue={[20]}
        max={100}
        step={1}
      />
      {/* <Input onChange={(e)=>setSelectEpochs(e.target.value)} value={epochs}  max={15} min={1} type="range" /> */}
    </div>
  );
};
