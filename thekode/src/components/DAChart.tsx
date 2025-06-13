"use client";
import type React from "react";
import { useEffect, useMemo, useRef, useState } from "react";
import { BarStack } from "@visx/shape";
import { Group } from "@visx/group";
import { AxisBottom, AxisLeft } from "@visx/axis";
import { scaleBand, scaleLinear, scaleOrdinal } from "@visx/scale";
import { LegendOrdinal } from "@visx/legend";
import { useTooltip, useTooltipInPortal } from "@visx/tooltip";
import { localPoint } from "@visx/event";
import { interpolateTurbo } from "d3-scale-chromatic";
import type { SeriesPoint } from "@visx/shape/lib/types";
import { Switch } from "@/components/ui/switch";
import { Label } from "@/components/ui/label";
import { ArrowLeft, ArrowRight, LoaderCircle } from "lucide-react";
import { useFileStore } from "@/components/Zustand";
import { Button } from "./ui/button";

export type ChartDataPoint = {
	[key: string]: number;
};

type Props = {
	data: ChartDataPoint[];
	width: number;
	height: number;
};

let tooltipTimeout: number;

type TooltipData = {
	bar: SeriesPoint<ChartDataPoint>;
	key: string;
	index: number;
	height: number;
	width: number;
	x: number;
	y: number;
	color: string;
};

const StackedBarChart: React.FC<Props> = ({ data, width, height }) => {
	const {
		tooltipOpen,
		tooltipLeft,
		tooltipTop,
		tooltipData,
		hideTooltip,
		showTooltip,
	} = useTooltip<TooltipData>();
	const { dataAnalysisXAxis, target } = useFileStore();
	const [clampInt, setClampInt] = useState(true);
	const [isPercentage, setIsPercentage] = useState(true);
	const [offset, setOffset] = useState(0);
	const range = 15;
	const [xAxis, setXAxis] = useState("");

	const { containerRef, TooltipInPortal } = useTooltipInPortal({
		scroll: true,
	});
	useEffect(() => {
		console.log(dataAnalysisXAxis);
		if (dataAnalysisXAxis === null) {
			setXAxis("");
			return;
		}
		setXAxis(dataAnalysisXAxis);
	}, [dataAnalysisXAxis]);
	useEffect(() => {
		if (data!==null) {
			setOffset(0);
		}
	}, [data]);

	
	const croppedData = useMemo(
		() => data.slice(offset, offset + range),
		[offset, data],
	);

	useEffect(()=>{

		console.log('data',data)
		console.log('croppedData',croppedData)
	},[croppedData,data])
	const keys = useMemo(
		() =>
			Object.keys(croppedData[0] ? croppedData[0] : []).filter(
				(key) => key !== xAxis,
			),
		[xAxis, croppedData],
	);
	const xMax = width - 100;
	const yMax = height - 100;

	const dateScale = useMemo(
		() =>
			scaleBand<string>({
				domain: croppedData.map((d) => `${d[xAxis]}`),
				padding: 0.2,
				range: [0, xMax],
			}),
		[croppedData, xMax, xAxis],
	);

	const valueScale = useMemo(
		() =>
			scaleLinear<number>({
				domain: [
					0,
					Math.max(
						...croppedData.map((d) =>
							keys.reduce((sum, key) => sum + (d[key] as number), 0),
						),
					),
				],
				range: [yMax, 0],
				nice: true,
			}),
		[croppedData, keys, yMax],
	);

	const colorScale = useMemo(
		() =>
			scaleOrdinal<string, string>({
				domain: keys,
				range: keys.map((_, index) => interpolateTurbo(index / keys.length)),
			}),
		[keys],
	);

	return (
		<div
			ref={containerRef}
			className="border  rounded-md border-white/10 bg-white/10 backdrop-blur-md  relative  "
		>
			<svg width={width+20} className=" p-2" height={height+30}>
				<title>Stacked Bar Chart</title>
				<Group top={30} left={70} >
					<BarStack
					data={croppedData}
						keys={keys}
						x={(d) => `${d[xAxis]}`}
						xScale={dateScale}
						yScale={valueScale}
						color={colorScale}
					>
						{(barStacks) =>
							barStacks.map((barStack) =>
								barStack.bars.map((bar) => {
									const barId = `bar-${barStack.index}-${bar.index}`;
									return (
										<rect
											key={barId}
											x={bar.x}
											y={bar.y}
											width={Math.max(bar.width - 3, 1)}
											height={bar.height}
											fill={bar.color}
											onMouseLeave={() => {
												tooltipTimeout = window.setTimeout(() => {
													hideTooltip();
												}, 300);
											}}
											onMouseMove={(event) => {
												if (tooltipTimeout) clearTimeout(tooltipTimeout);
												const eventSvgCoords = localPoint(event);
												const left = bar.x + bar.width / 2;
												showTooltip({
													tooltipData: bar,
													tooltipTop: eventSvgCoords?.y,
													tooltipLeft: left,
												});
											}}
											style={{
												outline: "1px solid black",
												position: "relative",
												cursor: "pointer",
											}}
										/>
									);
								}),
							)
						}
					</BarStack>

					<AxisBottom
						label={xAxis}
						labelOffset={40}
						top={yMax}
						scale={dateScale}
						
						tickTransform=""
						labelClassName=" text-2xl "
						tickClassName=""
						tickLabelProps={() => ({ fontSize: 11, textAnchor: "end", angle:-45})}
					/>
					<AxisLeft
						label={target ? target : ""}
						labelOffset={40}
						scale={valueScale}
						tickLabelProps={() => ({ fontSize: 11, textAnchor: "end",alignmentBaseline:"middle" })}
					/>
				</Group>
			</svg>
			{tooltipOpen && tooltipData && (
				<TooltipInPortal
					top={tooltipTop}
					left={tooltipLeft}
					applyPositionStyle
					style={{padding:0,paddingInline:10,backgroundColor:"#FFFFFFEE",border:'1px solid #000',borderRadius:2,backdropFilter:"blur(10px)"}}
					// className="bg-neutral-900 text-white p-20 border border-black "
				>
					<div className="" style={{ color: colorScale(tooltipData.key) }}>
						<strong>{tooltipData.key}</strong>
					</div>
					<div className="text-xs text-black/50">
						{(() => {
							const filteredData = { ...croppedData[tooltipData.index] };
							delete filteredData[xAxis];
							// const { ...filteredData } = ;
							// console.log([xAxis]);
							const sum = Object.values(filteredData).reduce(
								(acc, val) => (acc as number) + (val as number),
								0,
							);
							let f = Number.parseFloat(
								filteredData[tooltipData.key] as unknown as string,
							);

							if (isPercentage) {
								f =
									Math.round(
										(Number.parseFloat(
											filteredData[tooltipData.key] as unknown as string,
										) /
											Number.parseFloat(sum as unknown as string)) *
											10000,
									) / 100;
							}

							if (clampInt) {
								f = Math.round(f);
							}
							return `${f}${isPercentage ? "%" : ""}`;
						})()}
					</div>
				</TooltipInPortal>
			)}

			<div className="absolute right-0 top-0 border-s border-b rounded-bl-md overflow-hidden  backdrop-blur-md bg-white opacity-5  transition-all border-neutral-400/80 duration-700 hover:blur-0 hover:opacity-100">
				<div className="max-h-32 overflow-y-scroll scrollbar-thumb-neutral-200/50 scrollbar-thumb-rounded-full  scrollbar-track-transparent scrollbar-thin">
					<LegendOrdinal
						shape={"rect"}
						className="text-xs p-3 px-4  "
						scale={colorScale}
						labelFormat={(label) => label}
					/>
				</div>

				<div className="flex flex-col p-4    text-xs gap-2 border-t border-neutral/20">
					<div className="flex items-center gap-2">
						<Switch
							className="h-5 w-8 [&_span]:size-4 [&_span]:data-[state=checked]:translate-x-3 rtl:[&_span]:data-[state=checked]:-translate-x-3"
							checked={clampInt}
							onCheckedChange={setClampInt}
						/>
						<Label className="text-xs">{clampInt ? "Integer" : "Float"}</Label>
					</div>
					<div className="flex items-center gap-2">
						<Switch
							className="h-5 w-8 [&_span]:size-4 [&_span]:data-[state=checked]:translate-x-3 rtl:[&_span]:data-[state=checked]:-translate-x-3"
							checked={isPercentage}
							onCheckedChange={setIsPercentage}
						/>
						<Label className="text-xs">
							{isPercentage ? "Percentage" : "Absolute"}
						</Label>
					</div>
				</div>
				<div className=" flex flex-col p-4    text-xs gap-2 border-t border-neutral/20">
					<div className="flex w-full  justify-between gap-2">
						<Button
							disabled={offset === 0}
							onClick={() =>
								setOffset((x) => {
									const step = Math.min(Math.floor(data?.length * 0.2) || 1, x); // Step size starts at 20%, reduces near lower bound
									return Math.max(x - step, 0);
								})
							}
							className="w-full"
							size={"icon"}
						>
							<ArrowLeft />
						</Button>{" "}
						<Button
							disabled={offset + range >= (data?.length || 0)}
							onClick={() =>
								setOffset((x) => x+range)
							}
							className="w-full"
							size={"icon"}
						>
							<ArrowRight />
						</Button>
					</div>
					<h1 className="text-end">
					 {offset + 1}-{Math.min(offset + range, data?.length)} of{" "}
                        {data?.length}
					</h1>
				</div>
			</div>
		</div>
	);
};

export function Chart({
	data,

	type = "Stacked Bar Chart",
}: {
	data: ChartDataPoint[] | null;

	type?: "Stacked Bar Chart";
}) {
	const refContainer = useRef<HTMLDivElement | null>(null); // Proper ref type
	const [dimensions, setDimensions] = useState({
		width: 0,
		height: 0,
	});
	useEffect(() => {
		if (refContainer.current) {
			setDimensions({
				width: refContainer.current.offsetWidth,
				height: refContainer.current.offsetHeight,
			});
		}
	}, []);
	return (
		<div className=" flex ">
			<div
				ref={refContainer}
				className=" md:col-span-4 h-full    w-full flex items-start justify-start p-2"
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
					<>
						{type === "Stacked Bar Chart" ? (
							<StackedBarChart
								data={data}
								width={dimensions.width - 50}
								height={500}
							/>
						) : (
							<></>
						)}
					</>
				)}
			</div>
		</div>
	);
}



