export function getGradientStops(
  colorA: string,
  colorB: string,
  colorC: string,
  parts: number,
): string[] {
  // Parse hex color codes to RGB values
  const rgbA = hexToRgb(colorA);
  const rgbB = hexToRgb(colorB);
  const rgbC = hexToRgb(colorC);

  // Calculate the step size for color components
  const stepRed = (rgbB.red - rgbA.red) / (parts - 1);
  const stepGreen = (rgbB.green - rgbA.green) / (parts - 1);
  const stepBlue = (rgbB.blue - rgbA.blue) / (parts - 1);

  const stops: string[] = [];

  // Generate color stops
  for (let i = 0; i < parts; i++) {
    let red = Math.floor(rgbA.red + i * stepRed);
    let green = Math.floor(rgbA.green + i * stepGreen);
    let blue = Math.floor(rgbA.blue + i * stepBlue);

    // Handle transition to color C (if provided) in the middle
    if (colorC && i >= parts / 2) {
      const stepRedC = (rgbC.red - rgbB.red) / (parts - parts / 2);
      const stepGreenC = (rgbC.green - rgbB.green) / (parts - parts / 2);
      const stepBlueC = (rgbC.blue - rgbB.blue) / (parts - parts / 2);

      red = Math.floor(rgbB.red + (i - parts / 2) * stepRedC);
      green = Math.floor(rgbB.green + (i - parts / 2) * stepGreenC);
      blue = Math.floor(rgbB.blue + (i - parts / 2) * stepBlueC);
    }

    // Clamp RGB values to 0-255
    red = Math.max(0, Math.min(255, red));
    green = Math.max(0, Math.min(255, green));
    blue = Math.max(0, Math.min(255, blue));

    // Convert back to hex format
    stops.push(
      `#${red.toString(16).padStart(2, "0")}${green
        .toString(16)
        .padStart(2, "0")}${blue.toString(16).padStart(2, "0")}`,
    );
  }

  return stops;
}

type Range = [number, number]; // Define type for range pair

export function generateClassNames(column: string, ranges: number[]): string[] {
  return ranges.map((val) => `${column}-${val}`); // Use start+1 for class naming
}

// Helper function to convert hex color code to RGB object
function hexToRgb(hex: string): { red: number; green: number; blue: number } {
  const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
  return result
    ? {
        red: parseInt(result[1], 16),
        green: parseInt(result[2], 16),
        blue: parseInt(result[3], 16),
      }
    : { red: 0, green: 0, blue: 0 };
}

// Example usage
//   const stops = getGradientStops('#FF0000', '#00FF00', '#0000FF', 5);

export function divideIntoParts(a: number, b: number, c: number, x: number) {
  if (x <= 0) {
    throw new Error("Number of parts must be positive");
  }

  const totalSum = c - a;
  const partSize = totalSum / x;
  const ranges = [];
  let start = a;

  for (let i = 0; i < x; i++) {
    const end = start + partSize;
    ranges.push({ i: [start, end] });
    start = end;
  }

  return ranges;
}

export function generateStyleRules({
  min,
  mid,
  max,
  colorA,
  colorB,
  colorC,
  n,
}: {
  min: number;
  mid: number;
  max: number;
  colorA: string;
  colorB: string;
  colorC: string;
  n: number;
}): StyleRule[] {
  if (n <= 0) {
    throw new Error("Number of parts must be positive");
  }

  // Ensure min <= mid <= max
  if (min > mid || mid > max) {
    throw new Error("Values must be in ascending order: min <= mid <= max");
  }

  const ranges = divideIntoParts(min, mid, max, n);

  // console.log(ranges);
  // Generate color stops
  const stops = getGradientStops(colorA, colorB, colorC, n);

  return ranges.map((range, index) => ({
    className: `${index + 1}`, // Generate unique class names
    backgroundColor: stops[index],
    range: [range.i[0], range.i[1]],
  }));
}

// export function divideIntoParts(
//   a: number,
//   b: number,
//   c: number,
//   x: number
// ): number[][] {
//   if (x <= 0) {
//     throw new Error("Number of parts must be positive");
//   }

//   const totalSum = a + b + c;
//   const partSize = totalSum / x;
//   const ranges: number[][] = [];
//   let start = 0;

//   for (let i = 0; i < x; i++) {
//     const end = Math.min(start + partSize, totalSum);
//     ranges.push([start, end]);
//     start = end;
//   }

//   return ranges;
// }

export type ConditionalFormatting = {
  table: string;
  columns: Array<{
    name: string;
    enabled?: boolean | undefined | null;
    format: FormattingOptions; // Separate type for format options
  }>;
};

export type FormattingOptions = [];

export type formattingModel = {
  Column: string;
  classes: {
    [key: number]: {
      color: number;
    };
  };
};

export type StyleRule = {
  className: string; // Class name for applying the style (required)
  backgroundColor: string; // Background color in hex format (required)
  range: [number, number]; // Range of values for this style (required)
};

export type TableName =
  | "original"
  | "analysis"
  | "correlation1"
  | "correlation2"
  | "prediction"
  | "hyper"
  | "confusionMatrixSheet"
  | "classificationReportSheet"
  | "bestParams"
  | string;
