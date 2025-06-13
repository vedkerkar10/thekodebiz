"use client";
import type React from "react";
import { useMemo, useState } from "react";
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
import axios from "axios";
import {
	Select,
	SelectContent,
	SelectItem,
	SelectTrigger,
	SelectValue,
} from "@/components/ui/select";
import { Input } from "@/components/ui/input";
import { LoaderCircle } from "lucide-react";
// import { CONFIG } from "@/lib/config";
import { Button } from "@/components/ui/button";
import { useFileStore } from "@/components/Zustand";

export type DataPoint = {
	[key: string]: number | string;
};

type Props = {
	data: DataPoint[];
	width: number;
	height: number;
	xAxis: string;
};

let tooltipTimeout: number;

type TooltipData = {
	bar: SeriesPoint<DataPoint>;
	key: string;
	index: number;
	height: number;
	width: number;
	x: number;
	y: number;
	color: string;
};

const StackedBarChart: React.FC<Props> = ({ data, width, height, xAxis }) => {
	const {
		tooltipOpen,
		tooltipLeft,
		tooltipTop,
		tooltipData,
		hideTooltip,
		showTooltip,
	} = useTooltip<TooltipData>();
	const [clampInt, setClampInt] = useState(true);
	const [isPercentage, setIsPercentage] = useState(true);
	const { containerRef, TooltipInPortal } = useTooltipInPortal({
		scroll: true,
	});
	const keys = useMemo(
		() => Object.keys(data[0]).filter((key) => key !== xAxis),
		[xAxis, data],
	);
	const xMax = width - 100;
	const yMax = height - 100;

	const dateScale = useMemo(
		() =>
			scaleBand<string>({
				domain: data.map((d) => d[xAxis] as string),
				padding: 0.2,
				range: [0, xMax],
			}),
		[data, xMax, xAxis],
	);

	const valueScale = useMemo(
		() =>
			scaleLinear<number>({
				domain: [
					0,
					Math.max(
						...data.map((d) =>
							keys.reduce((sum, key) => sum + (d[key] as number), 0),
						),
					),
				],
				range: [yMax, 0],
				nice: true,
			}),
		[data, keys, yMax],
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
			className="border p-4 rounded-md border-white/10 bg-white/10 backdrop-blur-md  relative w-full  "
		>
			<svg width={width} height={height}>
				<title>Stacked Bar Chart</title>
				<Group top={50} left={100}>
					<BarStack
						data={data}
						keys={keys}
						x={(d) => d[xAxis] as string}
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
											width={bar.width - 3}
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
						top={yMax}
						scale={dateScale}
						tickLabelProps={() => ({ fontSize: 11, textAnchor: "middle" })}
					/>
					<AxisLeft
						scale={valueScale}
						tickLabelProps={() => ({ fontSize: 11, textAnchor: "end" })}
					/>
				</Group>
			</svg>
			{tooltipOpen && tooltipData && (
				<TooltipInPortal
					top={tooltipTop}
					left={tooltipLeft}
					className="bg-neutral-900 text-white p-20 border border-black "
				>
					<div style={{ color: colorScale(tooltipData.key) }}>
						<strong>{tooltipData.key}</strong>
					</div>
					<div className="text-xs text-black/50">
						{(() => {
							const filteredData = { ...data[tooltipData.index] };
							delete filteredData[xAxis];
							// const { ...filteredData } = ;
							console.log([xAxis]);
							const sum = Object.values(filteredData).reduce(
								(acc, val) => (acc as number) + (val as number),
								0,
							);
							let f = Number.parseFloat(
								filteredData[tooltipData.key] as string,
							);

							if (isPercentage) {
								f =
									Math.round(
										(Number.parseFloat(
											filteredData[tooltipData.key] as string,
										) /
											Number.parseFloat(sum as string)) *
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

			<div className="absolute right-0 top-0 border-s border-b rounded-bl-md overflow-hidden hover:border-white/10 backdrop-blur-md bg-white opacity-5  transition-all border-black/20 duration-700 hover:blur-0 hover:opacity-100">
				<LegendOrdinal
					shape={"rect"}
					className="text-xs p-3 px-4  "
					scale={colorScale}
					labelFormat={(label) => label}
				/>
				<div className="flex flex-col p-4    text-xs gap-2 border-t border-white/20">
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
			</div>
		</div>
	);
};

export default function Page() {
	const { server } = useFileStore();
	const [data, setData] = useState<DataPoint[] | null>(null);
	const [xAxis, setXAxis] = useState("date");
	const [sheetFile, setSheetFile] = useState<File[] | null>(null);
	const [stage, setStage] = useState();
	const [isLoading, setIsLoading] = useState<boolean>(false);

	// useEffect(() => {
	// 	const fetchData = async () => {
	// 		try {
	// 			const response = await axios.get("/api/data");
	// 			// const response = await axios.post(`${server}/Summarise_Data`);
	// 			const result = await response.data;
	// 			const parsedData = result.map((item: DataPoint) => ({
	// 				...item,
	// 				date: item.date,
	// 			}));
	// 			setData(parsedData);
	// 			console.log(parsedData);
	// 		} catch (error) {
	// 			console.error("Error fetching data:", error);
	// 		}
	// 	};

	// 	fetchData();
	// }, []);

	const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
		const files = event.target.files;
		setSheetFile(files ? Array.from(files) : null);
	};

	const handleClick = async () => {
		try {
			setIsLoading(true);
			const formData = new FormData();
			if (!sheetFile) return;
			formData.append("file", sheetFile[0]);
			const x = await axios.post(`${server}/atomic_Summarise_Data`, formData);
			const result = await x.data;
			console.log(result);
			setData(result);
			setIsLoading(false);
			// setTimeout(() => {
			// 	setIsLoading(false);
			// }, 1000);
		} catch (error) {
			setData(null);
			console.error("Error processing file:", error);
			setIsLoading(false);
		}
	};

	return (
		<div className=" flex  h-screen">
			<div className="p-2">
				<div className="flex  flex-col  w-full justify-start p-4 pt-2 border border-white/10 bg-white/10  backdrop-blur-md  rounded-md ">
					<div className="space-y-2 w-full group relative">
						<Label>Sheet</Label>
						<Input
							onChange={handleFileChange}
							className=" backdrop-blur-md bg-white/10 border-white/10 "
							accept=".xls, .xlsx, .csv"
							type="file"
						/>
						{/* {JSON.stringify(sheetFile?.map((x) => x.name))} */}
					</div>

					<div className="mt-2">
						<Button
							onClick={handleClick}
							disabled={isLoading}
							data-loading={isLoading}
							className="group relative disabled:opacity-100 w-full "
						>
							<span className="group-data-[loading=true]:text-transparent">
								Process
							</span>
							{isLoading && (
								<div className="absolute inset-0 flex items-center justify-center">
									<LoaderCircle
										className="animate-spin"
										size={16}
										strokeWidth={2}
										aria-hidden="true"
									/>
								</div>
							)}
						</Button>
					</div>

					<div className="group relative w-full mt-2">
						<Label className="">X Axis</Label>
						<Select disabled={!data} onValueChange={setXAxis}>
							<SelectTrigger className="w-full bg-white/10 mt-1 border-white/10 ">
								<SelectValue placeholder="Select Attribute" />
							</SelectTrigger>
							<SelectContent className="bg-white/10 border-white/10 backdrop-blur-md">
								{Object.keys(data ? data[0] : {}).map((item) => {
									return (
										<SelectItem
											disabled={item === xAxis}
											key={item}
											value={item}
										>
											{item}
										</SelectItem>
									);
								})}
							</SelectContent>
						</Select>
					</div>
				</div>
			</div>

			<div className="col-span-2 md:col-span-4 h-full   w-full flex items-start justify-start p-2">
				{isLoading || !data ? (
					<></>
				) : (
					<StackedBarChart xAxis={xAxis} data={data} width={200} height={500} />
				)}
			</div>
		</div>
	);
}



