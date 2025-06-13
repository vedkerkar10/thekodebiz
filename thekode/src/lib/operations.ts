import { config } from "../../config";

export function cleanData(obj: any) {
  return obj.data.map((x: { hadID: any; id: any }) => {
    if (x.hadID === false) {
      delete x.id;
      delete x.hadID;
    }
    return x;
  });
}


