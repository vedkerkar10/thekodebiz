"use client";
import type React from "react";
import { useEffect, useRef, useState } from "react";
import { Group } from "@visx/group";
import { AxisBottom, AxisLeft } from "@visx/axis";
import { scaleLinear } from "@visx/scale";
import { useTooltip, useTooltipInPortal } from "@visx/tooltip";
import { localPoint } from "@visx/event";
import { Button } from "./ui/button";
import { ArrowBigLeft, ArrowBigRight, LoaderCircle } from "lucide-react";
import { useFileStore } from "./Zustand";

export type ChartDataPoint = {
  [key: string]: number;
};

type Props = {
  data: ChartDataPoint[];
  width: number;
  height: number;
};

type TooltipData = {
  x: number;
  y: number;
  key: string;
  value: number;
  color: string;
};

export function OutlierChart({
  data,
  type = "Scatter Plot",
}: {
  data: ChartDataPoint[][] | null;
  type?: "Scatter Plot";
}) {
  const refContainer = useRef<HTMLDivElement | null>(null);
  const [dimensions, setDimensions] = useState({
    width: 0,
    height: 0,
  });
  const [chartIndex, setChartIndex] = useState(0);
  const {toggleRefetchOutlierAnalysisData}=useFileStore()

  useEffect(() => {
    if (refContainer.current) {
      setDimensions({
        width: refContainer.current.offsetWidth,
        height: refContainer.current.offsetHeight,
      });
    }
  }, []);

  return (
    <div className="flex flex-col bg-white/10 border relative border-neutral-700 p-2 rounded-md backdrop-blur-md">
      <div>
        <Button onClick={toggleRefetchOutlierAnalysisData}>
          Refetch
        </Button>
      </div>
      <div className="flex bg-white/10 backdrop-blur-lg absolute z-10 top-0 right-0 flex-col mb-2 p-3">
        <div className="flex space-x-2">
          <Button
            onClick={() => setChartIndex((prev) => Math.max(prev - 1, 0))}
          >
            <ArrowBigLeft />
          </Button>
          <Button
            onClick={() =>
              setChartIndex((prev) =>
                Math.min(prev + 1, (data?.length || 1) - 1)
              )
            }
          >
            <ArrowBigRight />
          </Button>
        </div>
        <div className="text-black mt-1">
          {data ? `${chartIndex + 1} of ${data.length}` : "0 of 0"}
        </div>
      </div>
      <div
        ref={refContainer}
        className="md:col-span-4 h-full w-full flex items-start justify-start p-2"
      >
        {!data ? (
          <div className="w-full min-h-32 flex items-center justify-center border p-4 rounded-md border-white/10 bg-white/10 backdrop-blur-md ">
            <LoaderCircle
              className="animate-spin"
              size={24}
              strokeWidth={2}
              aria-hidden="true"
            />
          </div>
        ) : (
          <ScatterPlot
            data={data[chartIndex]}
            width={dimensions.width - 50}
            height={500}
          />
        )}
      </div>
    </div>
  );
}

const ScatterPlot = ({ data, width, height }: { data?: ChartDataPoint[]; width: number; height: number }) => {
  const { showTooltip, hideTooltip, tooltipData, tooltipLeft, tooltipTop } = useTooltip<TooltipData>();
  const { containerRef, TooltipInPortal } = useTooltipInPortal();

  if (!data) return <></>;

  const xAxis = Object.keys(data[0]).find((key) => key !== "count") || "X";
  const yAxis = "count";

  const xMax = width - 100;
  const yMax = height - 100;

  const xScale = scaleLinear({
    domain: [
      Math.min(...data.map((d) => d[xAxis])),
      Math.max(...data.map((d) => d[xAxis]))
    ],
    range: [0, xMax],
    nice: true,
  });

  const yScale = scaleLinear({
    domain: [0, Math.max(...data.map((d) => d[yAxis]))],
    range: [yMax, 0],
    nice: true,
  });

  return (
    <div ref={containerRef} style={{ position: "relative" }}>
      <svg width={width} height={height}>
        <Group top={30} left={70}>
          {data.map((d, i) => {
            const cx = xScale(d[xAxis]);
            const cy = yScale(d[yAxis]);
            return (
              <circle
                key={`point-${i}`}
                cx={cx}
                cy={cy}
                r={4}
                fill="blue"
                onMouseEnter={(event) => {
                  const coords = localPoint(event);
                  showTooltip({
                    tooltipLeft: coords?.x,
                    tooltipTop: coords?.y,
                    tooltipData: { x: d[xAxis], y: d[yAxis], key: xAxis, value: d[xAxis], color: "blue" },
                  });
                }}
                onMouseLeave={hideTooltip}
              />
            );
          })}
          <AxisBottom top={yMax} scale={xScale} label={xAxis} labelOffset={40} />
          <AxisLeft scale={yScale} label={yAxis} labelOffset={40} />
        </Group>
      </svg>
      {tooltipData && (
        <TooltipInPortal left={tooltipLeft} top={tooltipTop}>
          <div className="bg-white p-2 shadow-md text-black rounded-md">
            {`${tooltipData.key}: ${tooltipData.value}`}
          </div>
        </TooltipInPortal>
      )}
    </div>
  );
};