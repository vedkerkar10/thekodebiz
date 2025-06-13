"use client";

import * as React from "react";
import { Check } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Command,
  CommandEmpty,
  CommandGroup,
  CommandInput,
  CommandItem,
  CommandList,
} from "@/components/ui/command";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";

type Option = {
  label: string;
  value: string;
};

type MultiSelectProps = {
  options: Option[];
  selected: string[];
  onChange: (selected: string[]) => void;
  placeholder?: string;
};

export function MultiSelect({
  options,
  selected,
  onChange,
  placeholder = "Select Features",
}: MultiSelectProps) {
  const toggleSelectAll = () =>
    onChange(
      selected.length === options.length ? [] : options.map((o) => o.value)
    );

  const handleToggle = (value: string) =>
    onChange(
      selected.includes(value)
        ? selected.filter((item) => item !== value)
        : [...selected, value]
    );

  return (
    <Popover>
      <PopoverTrigger asChild>
        <Button
          variant="outline"
          className="w-full bg-white/10 border-white/10 hover:bg-sky-100/20 text-left justify-start overflow-clip text-ellipsis"
        >
          {selected.length === 0
            ? placeholder
            : selected.length === 1
            ? selected[0]
            : `${selected.length} features selected`}
        </Button>
      </PopoverTrigger>
      <PopoverContent className="w-56 bg-transparent border-none p-0.5">
        <Command className="bg-white/10 border-white/10 border backdrop-blur-xl">
          <CommandInput placeholder="Search..." />
          <CommandList>
            <CommandEmpty>No results found.</CommandEmpty>
            <CommandGroup heading="Actions">
              <CommandItem onSelect={toggleSelectAll}>
                {selected.length === options.length
                  ? "Deselect All"
                  : "Select All"}
              </CommandItem>
            </CommandGroup>
            <CommandGroup heading="Features">
              {options.map((option) => {
                const isActive = selected.includes(option.value);
                return (
                  <CommandItem
                    key={option.value}
                    onSelect={() => handleToggle(option.value)}
                  >
                    <Check
                      className={`mr-2 h-4 w-4 ${
                        isActive ? "opacity-100" : "opacity-0"
                      }`}
                    />
                    {option.label}
                  </CommandItem>
                );
              })}
            </CommandGroup>
          </CommandList>
        </Command>
      </PopoverContent>
    </Popover>
  );
}
