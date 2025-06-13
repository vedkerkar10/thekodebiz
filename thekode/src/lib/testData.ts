export const TDATA = {
	outlier_removal: {
		feature_names: [
			"Age",
			"EMI Amount",
			"Pincode",
			"Region",
			"Gross Disbursal",
			"Loan Amount",
		],
		functions: [
			{
				name: "iqr",
				parameters: [
					{
						type: "number",
						id: "q1",
						range: {
							min: 0,
							max: 50,
						},
						default: 25,
					},
					{
						type: "number",
						id: "q3",
						range: {
							min: 51,
							max: 100,
						},
						default: 75,
					},
					{
						type: "number",
						id: "outlier_coefficient",
						range: {
							min: 1,
							max: 2,
						},
						default: 1.5,
					},
				],
			},
			{
				name: "zscore",
				parameters: [
					{
						type: "number",
						id: "threshold",
						range: {
							min: 2,
							max: 6,
						},
						default: 3,
					},
				],
			},
			{
				name: "remove_percentile",
				parameters: [
					{
						type: "number",
						id: "lower_percentile",
						range: {
							min: 0,
							max: 50,
						},
						default: 1,
					},
					{
						type: "number",
						id: "upper_percentile",
						range: {
							min: 50,
							max: 100,
						},
						default: 99,
					},
				],
			},
		],
	},
	scaling: {
		feature_names: [
			"Age",
			"EMI Amount",
			"Pincode",
			"Region",
			"Gross Disbursal",
			"Loan Amount",
		],
		functions: [
			{
				name: "MinMaxScaler()",
				parameters: [],
			},
			{
				name: "StandardScaler()",
				parameters: [],
			},
		],
	},
	sampling: {
		feature_names: [
			"Age",
			"EMI Amount",
			"Pincode",
			"Region",
			"Gross Disbursal",
			"Loan Amount",
		],
		functions: [
			{
				name: "SMOTE()",
				parameters: [
					{
						type: "number",
						id: "random_state",
						range: {
							min: 0,
							max: 42,
						},
						default: 42,
					},
					{
						type: "string",
						id: "sampling_strategy",
						range: ["auto"],
						default: "auto",
					},
				],
			},
			{
				name: "NearMiss()",
				parameters: [
					{
						type: "string",
						id: "sampling_strategy",
						range: ["auto"],
						default: "auto",
					},
				],
			},
		],
	},
	feature_selection: {
		functions: [
			{
				name: "RFECV()",
				parameters: [
					{
						type: "string",
						id: "estimator",
						range: "RandomForestClassifier(random_state=42)",
					},
					{
						type: "string",
						id: "cv",
						range: "StratifiedKFold(n_splits=5, shuffle=True, random_state=0)",
					},
					{
						type: "string",
						id: "scoring",
						range:
							"make_scorer(recall_score, average='\"micro\"', labels=class_labels)",
					},
					{
						type: "number",
						id: "verbose",
						range: {
							min: 0,
							max: 2,
						},
						default: 1,
					},
				],
			},
			{
				name: "RFECV()",
				parameters: [
					{
						type: "string",
						id: "estimator",
						range:
							"LogisticRegression(penalty='l2', solver='liblinear', random_state=42)",
					},
					{
						type: "string",
						id: "cv",
						range: "StratifiedKFold(n_splits=5, shuffle=True, random_state=0)",
					},
					{
						type: "string",
						id: "scoring",
						range:
							"make_scorer(recall_score, average='\"micro\"', labels=class_labels)",
					},
					{
						type: "number",
						id: "verbose",
						range: {
							min: 0,
							max: 2,
						},
						default: 1,
					},
				],
			},
		],
	},
};