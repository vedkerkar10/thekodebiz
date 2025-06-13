import { FileWarningIcon, TriangleAlert } from "lucide-react";
import {
  Breadcrumb,
  BreadcrumbItem,
  BreadcrumbLink,
  BreadcrumbList,
  BreadcrumbPage,
  BreadcrumbSeparator,
} from "@/components/ui/breadcrumb";
import Link from "next/link";

export default function Page() {
  return (
    <div className="p-8">
      <Breadcrumb>
        <BreadcrumbList>
          <BreadcrumbItem>
            <BreadcrumbLink href="/manual">Manual</BreadcrumbLink>
          </BreadcrumbItem>
        </BreadcrumbList>
      </Breadcrumb>

      <main className="prose ">
        <h1>Manual</h1>
        <Link href={"/manual/error"} className="flex items-center gap-2">
          Error Codes
        </Link>

        <p className="mt-1">A compilation of potential error codes.</p>
      </main>
    </div>
  );
}
