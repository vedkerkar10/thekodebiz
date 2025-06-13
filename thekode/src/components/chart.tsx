"use client";

import { toPng } from "html-to-image";
import download from "downloadjs";
import { cn } from "@/lib/utils";

import { ResponsiveLine } from "@nivo/line";
import { ChevronLeft, ChevronRight, Download } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { Button } from "./ui/button";

export function TimeseriesChart({
  chardData,
  className,
}: {
  chardData: {
    label: string;
    y: Array<number>;
    values: string;
    x: number[];
  }[][];
  className?: string;
}) {
  const chart = useRef<HTMLDivElement | null>(null);
  const [currentCharts, setCurrentCharts] = useState(0);
  useEffect(() => {
    console.log(chardData);
  }, [chardData]);
  return (
    <div
      className={cn(
        className,
        " h-[26rem] p-2 bg-white rounded-t-md border relative "
      )}
    >
      <div
        className="bg-white "
        style={{ height: "100%", width: "100%" }}
        ref={chart}
      >
        <ResponsiveLine
          data={chardData[currentCharts].map((d) => {
            return {
              id: d.label,
              // color:'#00da88',
              data: d.y.map((val: number, idx: number) => {
                const date = new Date(d.x[idx]);
                return {
                  x: `${date.getFullYear()}-${(date.getMonth() + 1)
                    .toString()
                    .padStart(2, "0")}-${date
                    .getDate()
                    .toString()
                    .padStart(2, "0")}`,
                  y: val,
                };
              }),
            };
          })}
          legends={[
            {
              anchor: "top-right",
              direction: "column",
              justify: false,
              translateX: -10,
              translateY: 0,
              itemWidth: 120,
              itemHeight: 20,
              itemsSpacing: 4,
              symbolSize: 10,
              symbolShape: "circle",
              itemDirection: "left-to-right",
              itemTextColor: "#777",
              effects: [
                {
                  on: "hover",
                  style: {
                    itemBackground: "rgba(0, 0, 0, .03)",
                    itemOpacity: 1,
                  },
                },
              ],
            },
          ]}
          // curve=""
          useMesh={true}
          enableArea
          enablePoints={true}
          isInteractive={true}
          margin={{ top: 10, right: 20, bottom: 100, left: 40 }}
          xScale={{
            type: "time",
            format: "%Y-%m-%d",
            useUTC: false,
            precision: "day",
          }}
          xFormat="time:%Y-%m-%d"
          yScale={{
            type: "linear",
            min: 0,
            max: "auto",
          }}
          colors={[
            "#FF204E",
            "#007F73",
            "#6420AA",
            "#1D2B53",
            "#211951",
            "#B6FFFA",
            "#F8FF95",
            "#F94C10",
          ]}
          crosshairType="cross"
          axisBottom={{
            tickRotation: -45,
            tickSize: 0,
            tickPadding: 16,
            format: "%b %d %y",
            // tickValues: 'every month',
            legendOffset: -12,
          }}
          axisLeft={{
            tickSize: 0,
            tickValues: 5,
            tickPadding: 16,
          }}
          // useMesh={false}
          //   gridYValues={24}
          //   gridXValues={64}
          theme={{
            tooltip: {
              chip: {
                borderRadius: "9999px",
              },
              container: {
                fontSize: "12px",
                textTransform: "capitalize",
                borderRadius: "6px",
              },
            },
            grid: {
              line: {
                stroke: "#f3f4f622",
              },
            },
          }}
          // role="application"
        />
      </div>
      <div className=" absolute  left-0 right-0  mt-3 w-full  rounded-b-md bg-white p-4 border flex items-center justify-between">
        <Button
          variant={"ghost"}
          onClick={async () => {
            if (!chart.current) {
              return;
            }

            const dataUrl = await toPng(chart.current, { quality: 100 });

            download(dataUrl, "chart.png");
          }}
        >
          Download Chart
        </Button>

        <div className=" flex gap-4 items-center  ">
          <div className="text-sm"> {chardData[currentCharts][0].values}</div>
          <div className="text-sm">
            {currentCharts + 1} of {chardData.length}
          </div>
          <Button
            onClick={() =>
              setCurrentCharts((x) => Math.max((x - 1) % chardData.length, 0))
            }
            variant={"secondary"}
          >
            <ChevronLeft className="w-6" />
          </Button>
          <Button
            onClick={() => setCurrentCharts((x) => (x + 1) % chardData.length)}
            variant={"secondary"}
          >
            <ChevronRight className="w-6" />
          </Button>
        </div>
      </div>
    </div>
  );
}
