"use client";

export const parsePythonDataFrame = (d: {
  hadID?: boolean | null;
  data: Array<{ [key: string]: object }>;
  schema: { fields: Array<{ name: string; type: string }> };
}) => {
  let data = d;
  if (typeof d === "string") {
    data = JSON.parse((d as string).replace("NaN", "null"));
  }

  const dates = data.schema.fields
    .filter((i) => i.type.toLowerCase().includes("date"))
    .map((i) => {
      return i.name;
    });
  const numbers = data.schema.fields
    .filter((i) => i.type.toLowerCase().includes("integer"))
    .map((i) => {
      return i.name;
    });

  return {
    headers: data.schema.fields.map((i) => {
      return {
        // width:
        //   data.schema.fields.length <= 4 ||
        //   i.type.toLowerCase().includes("date")
        //     ? i.type.toLowerCase().includes("date")
        //       ? 150
        //       : 200
        //     : 100,
        minWidth: 100,
        flex: 1,
        field: i.name,
        headerName: i.name,
        type: i.type.replace("integer", "number").replace("datetime", "Date"),
      };
    }),
    // biome-ignore lint/suspicious/noExplicitAny: <explanation>
    data: data.data.map((i: any, idx: number) => {
      if (!Object.hasOwn(i, "id")) {
        i.id = idx;
        i.hadID = false;
      } else {
        i.hadID = true;
      }
      for (const iterator of dates) {
        const currentDate = new Date(i[iterator]);
        const currentStringDate = `${currentDate.getDate()}/${currentDate.getMonth()+1}/${currentDate.getFullYear()}`;

        i[iterator] = currentStringDate;
      }
      for (const iterator of numbers) {
        i[iterator] = i[iterator] === null ? "N/A" : i[iterator];
      }
      return i;
    }),
  };
};

export const DataFrameToUniqueValues = (d: {
  hadID?: boolean | null;
  data: { [key: string]: string | number | boolean | object }[];
  schema: { fields: { name: string; type: string }[] };
}): { name: string; values: (string | number)[] }[] => {
  const fields = d.schema.fields;
  const data = d.data;

  const output: { name: string; values: (string | number)[] }[] = [];

  for (const field of fields) {
    const fieldName = field.name;
    const values: (string | number)[] = [];

    // Iterate over each row in data and extract values for the current field
    for (const row of data) {
      values.push(row[fieldName] as string | number);
    }

    // Push field name and corresponding values to the output array
    output.push({ name: fieldName, values });
  }
  console.log(output);
  return output;
};
