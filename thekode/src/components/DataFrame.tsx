"use client";
import {
  Box,
  ListItemIcon,
  ListItemText,
  MenuItem,
  Theme,
  ownerWindow,
  styled,
} from "@mui/material";
import {
  DataGrid,
  GridRowsProp,
  GridColDef,
  GridCell,
  GridColumnMenuContainer,
  GridColumnMenu,
  GridColumnMenuProps,
  GridColumnMenuSortItem,
  GridColumnMenuFilterItem,
  GridColumnMenuHideItem,
  GridColumnMenuManageItem,
  GridColumnMenuItemProps,
  GridToolbarColumnsButton,
  GridToolbarContainer,
  GridToolbarDensitySelector,
  GridToolbarExport,
  GridToolbarFilterButton,
  RowPropsOverrides,
  GridCellParams,
} from "@mui/x-data-grid";
import { useTheme } from "next-themes";
// import { MaterialSymbolsLightDownloadRounded } from "./icons";
import { Button } from "./ui/button";
import * as XLSX from "xlsx";
import { MouseEvent, useEffect, useState } from "react";
import { Paintbrush, Star } from "lucide-react";
// import { RightOptions } from "./RightOptions";
import {
  ConditionalFormatting,
  StyleRule,
  TableName,
  divideIntoParts,
  generateClassNames,
  generateStyleRules,
} from "@/lib/color";
import { useFileStore } from "@/components/Zustand";
import { header_type } from "./types";
import { cn } from "@/lib/utils";
// import { useStore, useTableData } from "@/components/Zustand";

export function Dataframe({
		data,
		name,
		className,
		show_cols,
		hide_footer,
	}: {
		data: {
			headers: header_type[];
			data: { [key: string]: any; id: number }[];
		} | null;
		name: TableName;
		className?: string;
		show_cols?: string[];
		hide_footer?: boolean;
	}) {
		const { theme } = useTheme();
		console.log("data", name, data);

		if (data === null) {
			console.log("NO DATA in", name);
			return <></>;
		}

		const preprocess_data = data.data.map(
			(item: Record<string, any>, idx: number) => {
				if (!("id" in item)) {
					return { ...item, id: idx }; // Ensure 'item' is treated as an object
				}
				return item;
			},
		);

		const filteredData =
			!show_cols || show_cols.length === 0
				? data.data
				: (data.data.map((item) =>
						Object.fromEntries(
							Object.entries(item).filter(([key]) => show_cols.includes(key)),
						),
					) as { [key: string]: any; id: number }[]);

		const filteredHeaders =
			!show_cols || show_cols.length === 0
				? data.headers
				: data.headers.filter((header) => show_cols.includes(header.field));

		console.log("show_cols", show_cols);
		console.log("filteredHeaders", filteredHeaders);

		data = {
			data: filteredData.map((item: Record<string, any>, idx: number) => ({
				...item,
				id: "id" in item ? item.id : idx,
			})),
			headers: filteredHeaders,
		};

		if (!Object.keys(data.data[0]).includes("id")) {
			data = {
				data: filteredData.map((v, idx) => {
					v.id = idx;
					return v;
				}),

				headers: filteredHeaders,
			};
			console.log("data", data);
			return <>No ID</>;
		}
		console.log("final data", data);

		return (
			<div className={cn("h-[calc(100vh-9rem)] relative", className)}>
				<DataGrid
					rowHeight={24}
					// rowCount={10}
					hideFooter={hide_footer}
					// paginationMode="server"
					style={{
						backdropFilter: "blur(10px)",
						color: theme === "dark" ? "white" : "black",
						borderColor: theme === "dark" ? "#555" : "#777",
					}}
					sx={{
						"*": {
							transition: "backgroundColor",
							transitionDuration: 500,

							color: theme === "dark" ? "white" : "black",
						},
						".MuiDataGrid-topContainer": {
							backgroundColor: "#9DD9FF55",
						},
						".Mui-even": {
							backgroundColor: theme === "dark" ? "#00131F" : "#9DD9FF55",
							filter: "saturate(120%)",
						},
						".Mui-even:hover": {
							backgroundColor: theme === "dark" ? "#00131F" : "#9DD9FF88",
						},
						".MuiDataGrid-columnHeaders, .MuiDataGrid-toolbarContainer": {
							// borderRadius:'0',
							backgroundColor: theme === "dark" ? "#001105" : "#00A934",
						},
						".MuiDataGrid-columnHeaders": {
							backgroundColor: "#9DD9FF55",
							borderBottom:
								theme === "dark" ? "1px dotted white" : "1px dotted black",
							borderRadius: "0",
						},
						".NA_Val:after": {
							content: '"N/A"',
						},

						// ...getColumnStyles(),
					}}
					getCellClassName={(params: any) => {
						if (params.value == null) {
							return "NA_Val";
						}
						return "";
					}}
					getRowClassName={(params) =>
						params.indexRelativeToCurrentPage % 2 === 0 ? "Mui-even" : "Mui-odd"
					}
					slotProps={{
						cell: {
							showRightBorder: true,
							style: { borderColor: "#444" },
						},
					}}
					slots={{
						columnMenu: CustomColumnMenuComponent,
						// toolbar: CustomToolbar,
					}}
					columns={data.headers as any}
					rows={data.data}
				/>
			</div>
		);
	}

export function CustomColumnMenuComponent(props: GridColumnMenuProps) {
  const { hideMenu, colDef, color, ...other } = props;
  const { theme } = useTheme();

  return (
    <>
      <Box sx={{ svg: { color: theme === "dark" ? "#f0f0f066" : "#0f0f0f" } }}>
        <GridColumnMenu
          className="dark:bg-neutral-900 dark:text-white [svg]:[color:'white']"
          hideMenu={hideMenu}
          colDef={colDef}
          {...other}
        />
      </Box>
    </>
  );
}
