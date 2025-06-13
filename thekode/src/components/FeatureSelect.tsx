import { cn } from "@/lib/utils";
import { Check } from "lucide-react";
import { Button } from "./ui/button";
import {
  Command,
  CommandEmpty,
  CommandGroup,
  CommandInput,
  CommandItem,
  CommandList,
} from "./ui/command";
import { Popover, PopoverContent, PopoverTrigger } from "./ui/popover";
import { useFileStore } from "@/components/Zustand";
import { useEffect } from "react";
export function FeatureSelect() {
  const { targets, target, setSelectedFeatures, selectedFeatures,setCorrelation1Sheet,setCorrelation2Sheet ,setCurrentSheet,setAnalysisSheet,setPredictionSheet,loading} =
    useFileStore();
  // biome-ignore lint/correctness/useExhaustiveDependencies: <explanation>
useEffect(() => {
    setCorrelation1Sheet(null)
    setCorrelation2Sheet(null)
    setCurrentSheet('Original')
    setAnalysisSheet(null)
    setPredictionSheet(null)
  }, [selectedFeatures, setAnalysisSheet, setCorrelation1Sheet, setCorrelation2Sheet, setCurrentSheet, setPredictionSheet]);
  useEffect(() => {
    if (target) {
      setSelectedFeatures([]);
    }
  }, [setSelectedFeatures, target]);
  return (
    <div className="w-full">
      <Popover >
        <PopoverTrigger
          asChild
          className="w-full bg-white/10 border-white/10 hover:bg-sky-100/20"
        >
          <Button
          disabled={loading}
            className="w-full  text-left  justify-start overflow-clip text-ellipsis"
            variant={"outline"}
          >
            {selectedFeatures.length === 0
              ? "Select Features"
              : `${
                  selectedFeatures ? selectedFeatures.length : 0
                } features selected`}
          </Button>
        </PopoverTrigger>
        <PopoverContent className="w-56 bg-transparent border-none p-0.5">
          <Command className=" bg-white/10 border-white/10 border backdrop-blur-xl ">
            <CommandInput className="" placeholder="Search..." />
            <CommandList className="">
              <CommandEmpty>No results found.</CommandEmpty>
              <CommandGroup heading="Actions">
                <CommandItem
                  onSelect={() => {
                    setSelectedFeatures(
                      selectedFeatures.length === targets.length - 1 &&
                        selectedFeatures.every((feature) =>
                          targets.includes(feature)
                        )
                        ? []
                        : [...targets.filter((i) => i !== target)]
                    );
                  }}
                >
                  {selectedFeatures.length === targets.length - 1 &&
                  selectedFeatures.every((feature) => targets.includes(feature))
                    ? "Deselect All"
                    : "Select All"}
                </CommandItem>
              </CommandGroup>
              <CommandGroup heading="Features">
                {/* {JSON.stringify(features)} */}

                {targets
                  .filter((i) => i !== target)
                  .map((i) => {
                    const isActive = selectedFeatures.includes(i);
                    return (
                      <CommandItem
                        onSelect={() => {
                          setSelectedFeatures(
                            selectedFeatures.includes(i)
                              ? selectedFeatures.filter((x) => x !== i)
                              : [...selectedFeatures, i]
                          );
                        }}
                        key={i}
                      >
                        <Check
                          className={cn(
                            "mr-2 h-4 w-4",
                            isActive ? "opacity-100" : "opacity-0"
                          )}
                        />
                        {i}
                      </CommandItem>
                    );
                  })}
              </CommandGroup>
            </CommandList>
          </Command>
        </PopoverContent>
      </Popover>
    </div>
  );
}
