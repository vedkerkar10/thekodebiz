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
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "./ui/select";
export function PlotValsSelect({ attr }: { attr: any }) {
  const {
    targets,
    target,
    setSelectedFeatures,
    selectedFeatures,
    setCorrelation1Sheet,
    setCorrelation2Sheet,
    setCurrentSheet,
    setAnalysisSheet,
    setPredictionSheet,
    selectedUniqueValues,
    setSelectedUniqueValues,
    uniqueValues,
    removeSelectedUniqueValues,
  } = useFileStore();

  return (
    <div className="w-full">
 
      <Popover>
        <PopoverTrigger
          asChild
          className="w-full bg-white/10 border-white/10 hover:bg-sky-100/20"
        >
          <Button
            className="w-full  text-left  justify-start overflow-clip text-ellipsis"
            variant={"outline"}
          >
            {selectedUniqueValues[attr] && selectedUniqueValues[attr].length > 0
              ? selectedUniqueValues[attr].length > 5?`${selectedUniqueValues[attr].length} features selected`:selectedUniqueValues[attr].join(", ")
              : "No Items Selected"}
        
          </Button>
        </PopoverTrigger>
        <PopoverContent className="w-56 bg-transparent border-none p-0.5">
          <Command className=" bg-white/10 border-white/10 border backdrop-blur-xl ">
            <CommandInput className="" placeholder="Search..." />
            <CommandList className="">
              <CommandEmpty>No results found.</CommandEmpty>
              {/* <CommandGroup heading="Actions">
                <CommandItem
                  onSelect={() => {}}
                >
                Select All
                </CommandItem>
              </CommandGroup> */}
              <CommandGroup heading="Features">
                {/* {JSON.stringify(features)} */}

                {uniqueValues[attr].map((i) => {
                  const isActive = selectedUniqueValues[attr]
                    ? selectedUniqueValues[attr].includes(i)
                    : false;
                  return (
                    <CommandItem
                      onSelect={() => {
                        console.log(attr, i);
                        if (isActive) {
                          removeSelectedUniqueValues(attr, i);
                        } else {
                          setSelectedUniqueValues(attr, i);
                        }
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
