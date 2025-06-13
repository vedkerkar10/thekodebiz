import { HomeIcon, Table } from "lucide-react";
import Link from "next/link";

export default function SideBar() {
  return (
    <div className="w-16 p-2 border-r border-gray-300 h-screen flex flex-col ">
      <Link
        href={"/"}
        className="p-2 flex items-center justify-center aspect-square"
      >
        <Table />
      </Link>
      {/* <Link href={"/"}>Home</Link> */}
    </div>
  );
}
