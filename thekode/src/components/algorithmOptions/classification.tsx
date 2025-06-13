import { FeatureSelect } from "../FeatureSelect";
import { Label } from "../ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "../ui/select";
import { useFileStore } from "@/components/Zustand";

export const ClassificationOptions = () => {
  const { targets,setTarget,target,loading } = useFileStore();

  return (
    <div>
      <Label>Select Target</Label>
      <Select value={target as string} onValueChange={(e)=>setTarget(e)} disabled={targets === null||loading}>
        <SelectTrigger className="w-full bg-white/10 border-white/10">
          <SelectValue>{target?`Selected : ${target}`:"Select Target"}</SelectValue>
        </SelectTrigger>
        <SelectContent className="bg-white/10 border-white/10 backdrop-blur-lg">
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
      <div>

      <FeatureSelect />
      </div>
    </div>
  );
};
