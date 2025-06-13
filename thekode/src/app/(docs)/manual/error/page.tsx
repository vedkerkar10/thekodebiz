import { FileWarningIcon, TriangleAlert } from "lucide-react";
import {
  Breadcrumb,
  BreadcrumbItem,
  BreadcrumbLink,
  BreadcrumbList,
  BreadcrumbPage,
  BreadcrumbSeparator,
} from "@/components/ui/breadcrumb";

export default function Page() {
  return (
    <div className="p-8 h-full">
      <Breadcrumb>
        <BreadcrumbList>
          <BreadcrumbItem>
            <BreadcrumbLink href="/manual">Manual</BreadcrumbLink>
          </BreadcrumbItem>
          <BreadcrumbSeparator />
          <BreadcrumbItem>
            <BreadcrumbLink href="/error">Error Codes</BreadcrumbLink>
          </BreadcrumbItem>
        </BreadcrumbList>
      </Breadcrumb>

      <main className="prose ">
        <h1>Error Codes</h1>
        <h3 className=" flex items-center gap-2">
          001 <TriangleAlert className=" text-red-500" />
        </h3>
        <i>Missing/Incorrect Data</i>
        <p>
          Prediction File has columns not present in the file processed during
          Training
        </p>
        <h3 className=" flex items-center gap-2">
          002 <TriangleAlert className=" text-yellow-500" />
        </h3>
        <i>Missing/Incorrect Data</i>
        <p>Trained on different features than present in the prediction file</p>
        <h3 className=" flex items-center gap-2">
          003 <TriangleAlert className=" text-red-500" />
        </h3>
        <i>Missing/Incorrect Data</i>
        <p>Aggregator not present in the prediction file</p>
      </main>
    </div>
  );
}
