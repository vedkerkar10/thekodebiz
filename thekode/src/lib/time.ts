export function formatSmallTime(nanoseconds: number): string {
	const levels = [
		{ unit: "ps", value: 1e-3 },
		{ unit: "ns", value: 1 },
		{ unit: "µs", value: 1e3 },
		{ unit: "ms", value: 1e6 },
		{ unit: "seconds", value: 1e9 },
		{ unit: "minutes", value: 60 * 1e9 },
	];

	for (let i = 0; i < levels.length - 1; i++) {
		const current = levels[i];
		const next = levels[i + 1];

		if (nanoseconds < next.value) {
			return `${(nanoseconds / current.value).toFixed(2)} ${current.unit}`;
		}
	}

	return `${(nanoseconds / levels[levels.length - 1].value).toFixed(2)} minutes`;
}