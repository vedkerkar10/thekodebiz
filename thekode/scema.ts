interface BaseOptions {
	display_name?: string;
	id: string;
}
interface BooleanOptions extends BaseOptions {
	type: "boolean";
	default: boolean;
}
interface NumberOptions extends BaseOptions {
	type: "number";
	range: { min: number; max: number };
	default: number;
}

interface TemplateOptions extends BaseOptions {
	type: "template";
	forEach?: string;
	ref?: string;
	name: string;
}
interface SelectOptions extends BaseOptions {
	type: "select";
	select: "single" | "multiple";
	options: Array<{ display_name?: string; id: string }>;
}
interface LabelOptions extends Omit<BaseOptions, "id"> {
	type: "title";
}

type ui_schema_options =
	| BaseOptions
	| SelectOptions
	| LabelOptions
	| NumberOptions
	| BooleanOptions
	| TemplateOptions;

interface ui_schema_type {
	flow: string;
	completion: {
		button_text: string;
		action: "fetch";
		method: "POST";
		endpoint: "/api/flow";
	};
	position: 0 | 1 | 2 | 3;
	options: ui_schema_options[];
}

interface BlockOption {
	name: string;
	options: ui_schema_options[];
}

interface BlockConfig {
	name: string;
	default?: string;
	available_options: BlockOption[];
}

const block_config: BlockConfig[] = [
	{
		name: "outlier_options",
		default: "iqr",
		available_options: [
			{
				name: "iqr",
				options: [
					{
						type: "number",
						id: "q1",
						range: { min: 0, max: 1 },
						default: 0,
					},
					{
						type: "number",
						id: "q3",
						range: { min: 0, max: 1 },
						default: 0,
					},
					{
						type: "number",
						id: "outlier_coefficient",
						range: { min: 0, max: 10 },
						default: 1.5,
					},
				],
			},
			{
				name: "z_score",
				options: [
					{
						type: "number",
						id: "threshold",
						range: { min: 0, max: 10 },
						default: 0,
					},
				],
			},
			{
				name: "remove_percentile",
				options: [
					{
						type: "number",
						id: "lower_percentile",
						display_name: "Lower Percentile",
						range: { min: 0, max: 10 },
						default: 0,
					},
					{
						type: "number",
						id: "upper_percentile",
						display_name: "Upper Percentile",
						range: { min: 0, max: 10 },
						default: 0,
					},
				],
			},
		],
	},
	{
		name: "scaling_options",
		available_options: [
			{
				name: "default",
				options: [
					{
						type: "boolean",
						id: "scaling",
						default: false,
					},
					{
						type: "select",
						id: "scaling_columns",
						select: "multiple",
						options: [
							{ id: "age", display_name: "Age" },
							{ id: "emi", display_name: "EMI" },
							{ id: "income", display_name: "Income" },
						],
					},
				],
			},
		],
	},
	{
		name: "feature_selection_options",
		available_options: [
			{
				name: "default",
				options: [
					{
						type: "select",
						select: "single",
						id: "feature_selection_method1",
						display_name: "Feature Selection Method",
						options: [
							{ id: "RFECV", display_name: "RFECV" },
							{ id: "RFE", display_name: "RFE" },
							{ id: "SelectFdr", display_name: "SelectFdr" },
							{ id: "SelectFpr", display_name: "SelectFpr" },
							{ id: "SelectPercentile", display_name: "SelectPercentile" },
						],
					},
					{
						type: "select",
						id: "scaling_columns",
						select: "multiple",
						options: [
							{ id: "age", display_name: "Age" },
							{ id: "emi", display_name: "EMI" },
							{ id: "income", display_name: "Income" },
						],
					},
				],
			},
		],
	},
];

const ui_schema_options: ui_schema_type[] = [
	{
		flow: "preprocessing",
		position: 1,
		completion: {
			button_text: "Perform Preprocessing",
			action: "fetch",
			method: "POST",
			endpoint: "/api/flow",
		},
		options: [
			{
				type: "boolean",
				id: "remove_outliers",
				display_name: "Remove Outliers",
			},
			{
				type: "select",
				select: "multiple",
				id: "outlier_columns",
				display_name: "Outlier Columns",
				options: [
					{ id: "age", display_name: "Age" },
					{ id: "emi", display_name: "EMI" },
					{ id: "amount", display_name: "amount" },
				],
			},
			{
				type: "template",
				id: "outlier_options",
				forEach: "outlier_options",
			},
			{
				type: "template",
				id: "scaling_options_config",
				name: "scaling_options",
			},
		],
	},
	{
		flow: "feature_selection",
		position: 2,
		completion: {
			button_text: "Perform Feature Selection",
			action: "fetch",
			method: "POST",
			endpoint: "/api/flow",
		},
		options: [
			{
				type: "boolean",
				id: "feature_selection",
			},
		
			{
				type: "template",
				name: "feature_selection_options",
				id: "feature_selection_options",
				ref: "feature_selection_method1",
			},
		],
	},
];

// const ui_schema_options: ui_schema_type[] = [
// 	{
// 		flow: "preprocessing",
// 		position: 1,
// 		options: [
// {
// 	type: "boolean",
// 	id: "remove_outliers",
// 	display_name: "Remove Outliers",
// },
// {
// 	type: "select",
// 	select: "multiple",
// 	id: "outlier_columns",
// 	display_name: "Outlier Columns",
// 	options: [
// 		{ id: "age", display_name: "Age" },
// 		{ id: "emi", display_name: "EMI" },
// 		{ id: "income", display_name: "Income" },
// 	],
// },
// 			{
// 				type: "title",
// 				display_name: "Select Method",
// 			},
// 			{
// 				type: "select",
// 				id: "outlier_columns",
// 				display_name: "Outlier Columns",
// 				options: [
// 					{ id: "age", display_name: "Age" },
// 					{ id: "emi", display_name: "EMI" },
// 					{ id: "income", display_name: "Income" },
// 				],
// 			},
// 			{
// 				type: "number",
// 				id: "q1",
// 				range: { min: 0, max: 1 },
// 				default: 0,
// 			},
// 			{
// 				type: "number",
// 				id: "q3",
// 				range: { min: 0, max: 1 },
// 				default: 0,
// 			},
// 			{
// 				type: "number",
// 				id: "outlier_coefficient",
// 				display_name: "Outlier Coefficient",
// 				range: { min: 0, max: 1 },
// 				default: 0,
// 			},
// 		],
// 	},
// ];
