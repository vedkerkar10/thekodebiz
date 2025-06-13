"use client";

// import { io } from "socket.io-client";
import { useFileStore } from "@/components/Zustand";
import { useState } from "react";

export function LoaderDisplay() {
  // Destructure loading state from the Zustand store
  // const socket = io(`${config.server}`);
  const [_loadingPercent, _setLoadingPercent] = useState(0);
  const { loading, loadingPercent } = useFileStore();

  // socket.on("progress", (v) => {
  //   console.log(v);
  //   _setLoadingPercent(v);
  //   if (v > 0.9) {
  //     _setLoadingPercent(1);
  //   }
  //   // if(v.Type==='TrainProgress'){
  //   //   console.log('Training');
  //   //   _setLoadingPercent(v.value)
  //   // }
  // });

  return (
    <>
      {loading?(
      <div className="absolute w-full h-0.5 z-[1000000000]   pointer-events-none overflow-hidden">
        <div className="h-10 w-56 bg-gradient-to-r from-indigo-800 backdrop-blur-xl to-indigo-500 animate-loading-bar relative" />
      </div>
      ):(<></>)}
      {/* {_loadingPercent}
      {loading && !loadingPercent ? (
        <div className="absolute w-screen h-screen z-[1000000000]   pointer-events-none overflow-hidden">
          <div className="h-0.5 w-56 bg-gradient-to-r from-sky-200/10 backdrop-blur-xl to-indigo-500 animate-loading-bar relative" />
        </div>
      ) : loading && loadingPercent ? (
        <div className="absolute w-screen h-screen z-[1000000000] pointer-events-none">
          <div
            style={{ width: `${_loadingPercent * 100}%` }}
            className="h-1  transition-all bg-slate-800  relative"
          />
        </div>
      ) : null} */}
    </>
  );
}

{
  /* <div className="fixed w-screen h-screen z-[1000000000]   pointer-events-none">
  <div
    style={{ width: `${_loadingPercent*100}%` }}
    className="h-1  transition-all duration-300 bg-slate-800  relative  "
  />
</div> */
}
